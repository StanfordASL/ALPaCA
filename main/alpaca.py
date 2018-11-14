import tensorflow as tf
import numpy as np
import time
from copy import deepcopy
from tensorflow.python.ops.parallel_for import gradients


class ALPaCA:
    def __init__(self, config, sess, graph=None, preprocess=None, f_nom=None):
        self.config = deepcopy(config)
        self.lr = config['lr']
        self.x_dim = config['x_dim']
        self.phi_dim = config['nn_layers'][-1]
        self.y_dim = config['y_dim']
        self.sigma_eps = self.config['sigma_eps']

        self.updates_so_far = 0
        self.sess = sess
        self.graph = graph if graph is not None else tf.get_default_graph()
        
        # y = K^T phi( preprocess(x) ) + f_nom(x)
        self.preprocess = preprocess
        self.f_nom = f_nom 

    def construct_model(self):
        with self.graph.as_default():
            last_layer = self.config['nn_layers'][-1]

            if self.sigma_eps is list:
                self.SigEps = tf.diag( np.array(self.sigma_eps) )
            else:
                self.SigEps = self.sigma_eps*tf.eye(self.y_dim)
            self.SigEps = tf.reshape(self.SigEps, (1,1,self.y_dim,self.y_dim))
            
            # try making it learnable
            #self.SigEps = tf.get_variable('sigma_eps', initializer=self.SigEps )

            # Prior Parameters of last layer
            self.K = tf.get_variable('K_init',shape=[last_layer,self.y_dim]) #\bar{K}_0

            self.L_asym = tf.get_variable('L_asym',shape=[last_layer,last_layer]) # cholesky decomp of \Lambda_0
            self.L = self.L_asym @ tf.transpose(self.L_asym) # \Lambda_0
            
            # x: query points (M, N_test, x_dim)
            # y: target points (M, N_test, y_dim) ( what K^T phi(x) should approximate )
            self.x = tf.placeholder(tf.float32, shape=[None,None,self.x_dim], name="x")
            self.y = tf.placeholder(tf.float32, shape=[None,None,self.y_dim], name="y")

            # Points used to compute posterior using BLR
            # context_x: x points available for context (M, N_context, x_dim)
            # context_y: y points available for context (M, N_context, y_dim)
            self.context_x = tf.placeholder(tf.float32, shape=[None,None,self.x_dim], name="cx")
            self.context_y = tf.placeholder(tf.float32, shape=[None,None,self.y_dim], name="cy")

            # num_updates: number of context points from context_x,y to use when computing posterior. size (M,)
            self.num_models = tf.shape(self.context_x)[0]
            self.max_num_context = tf.shape(self.context_x)[1]*tf.ones((self.num_models,), dtype=tf.int32)
            self.num_context = tf.placeholder_with_default(self.max_num_context, shape=(None,))

            # Map input to feature space
            with tf.variable_scope('phi',reuse=None):
                # self.phi is (M, N_test, phi_dim)
                self.phi = tf.map_fn( lambda x: self.basis(x),
                                 elems=self.x,
                                 dtype=tf.float32)

            # Map context input to feature space
            with tf.variable_scope('phi', reuse=True):
                # self.context_phi is (M, N_context, phi_dim)
                self.context_phi = tf.map_fn( lambda x: self.basis(x),
                                              elems=self.context_x,
                                              dtype=tf.float32)
                
            # Evaluate f_nom if given, else use 0
            self.f_nom_cx = tf.zeros_like(self.context_y)
            self.f_nom_x = 0 #tf.zeros_like(self.y)
            if self.f_nom is not None:
                self.f_nom_cx = self.f_nom(self.context_x)
                self.f_nom_x = self.f_nom(self.x)
                
            # Subtract f_nom from context points before BLR
            self.context_y_blr = self.context_y - self.f_nom_cx

            # Compute posterior weights from context data
            with tf.variable_scope('blr', reuse=None):
                # posterior_K is (M, phi_dim, y_dim), posterior_L_inv is (M, phi_dim, phi_dim)
                self.posterior_K, self.posterior_L_inv = tf.map_fn( lambda x: self.batch_blr(*x),
                                                                    elems=(self.context_phi, self.context_y_blr, self.num_context),
                                                                    dtype=(tf.float32, tf.float32) )


            # Compute posterior predictive distribution, and evaluate targets self.y under this distribution
            self.mu_pred, self.Sig_pred, self.predictive_nll = self.compute_pred_and_nll()
            
            # The loss function is expectation of this predictive nll.
            self.total_loss = tf.reduce_mean(self.predictive_nll)
            tf.summary.scalar('total_loss', self.total_loss)

            self.optimizer = tf.train.AdamOptimizer(self.lr)

            global_step = tf.Variable(0, trainable=False, name='global_step')
            self.train_op = self.optimizer.minimize(self.total_loss,global_step=global_step)

            self.train_writer = tf.summary.FileWriter('summaries/'+str(time.time()), self.sess.graph, flush_secs=10)
            self.merged = tf.summary.merge_all()

            self.saver = tf.train.Saver()

            self.sess.run(tf.global_variables_initializer())
    
    # ----  TF operations ---- #
    def basis(self,x,name='basis'):
        layer_sizes = self.config['nn_layers']
        activations = {
            'relu': tf.nn.relu,
            'tanh': tf.nn.tanh,
            'sigmoid': tf.nn.sigmoid
        }
        activation = activations[ self.config['activation'] ]

        if self.preprocess is None:
            inp = x
        else:
            inp = self.preprocess(x)
        with tf.variable_scope(name,reuse=tf.AUTO_REUSE):
            for units in layer_sizes:
                inp = tf.layers.dense(inputs=inp, units=units,activation=activation)

        return inp

    def batch_blr(self,X,Y,num):
        X = X[:num,:]
        Y = Y[:num,:]
        Ln_inv = tf.matrix_inverse(tf.transpose(X) @ X + self.L)
        Kn = Ln_inv @ (tf.transpose(X) @ Y + self.L @ self.K)
        return  tf.cond( num > 0, lambda : (Kn,Ln_inv), lambda : (self.K, tf.linalg.inv(self.L)) )
    
    def compute_pred_and_nll(self):
        """
        Uses self.posterior_K and self.posterior_L_inv and self.f_nom_x to generate the posterior predictive.
        Returns:
            mu_pred = posterior predictive mean at query points self.x
                        shape (M, T, y_dim)
            Sig_pred = posterior predictive variance at query points self.x 
                        shape (M, T, y_dim, y_dim)
            predictive_nll = negative log likelihood of self.y under the posterior predictive density
                        shape (M, T) 
        """
        mu_pred = batch_matmul(tf.matrix_transpose(self.posterior_K), self.phi) + self.f_nom_x
        spread_fac = 1 + batch_quadform(self.posterior_L_inv, self.phi)
        Sig_pred = tf.expand_dims(spread_fac, axis=-1)*tf.reshape(self.SigEps, (1,1,self.y_dim,self.y_dim))
        
        # Score self.y under predictive distribution to obtain loss
        with tf.variable_scope('loss', reuse=None):
            logdet = self.y_dim*tf.log(spread_fac) + tf.linalg.logdet(self.SigEps)
            Sig_pred_inv = tf.linalg.inv(Sig_pred)
            quadf = batch_quadform(Sig_pred_inv, (self.y - mu_pred))

        predictive_nll = tf.squeeze(logdet + quadf, axis=-1)
        
        # log stuff for summaries
        self.rmse_1 = tf.reduce_mean( tf.sqrt( tf.reduce_sum( tf.square(mu_pred - self.y)[:,0,:], axis=-1 ) ) )
        self.mpv_1 = tf.reduce_mean( tf.matrix_determinant( Sig_pred[:,0,:,:]) )
        tf.summary.scalar('RMSE_1step', self.rmse_1)
        tf.summary.scalar('MPV_1step', self.mpv_1)
        
        return mu_pred, Sig_pred, predictive_nll

    
    # ---- Train and Test functions ------ #
    def train(self, dataset, num_train_updates):
        batch_size = self.config['meta_batch_size']
        horizon = self.config['data_horizon']
        test_horizon = self.config['test_horizon']

        #minimize loss
        for i in range(num_train_updates):
            x,y = dataset.sample(n_funcs=batch_size, n_samples=horizon+test_horizon)

            feed_dict = {
                    self.context_y: y[:,:horizon,:],
                    self.context_x: x[:,:horizon,:],
                    self.y: y[:,horizon:,:],
                    self.x: x[:,horizon:,:],
                    self.num_context: np.random.randint(horizon+1,size=batch_size)
                    }

            summary,loss, _ = self.sess.run([self.merged,self.total_loss,self.train_op],feed_dict)

            if i % 50 == 0:
                print('loss:',loss)

            self.train_writer.add_summary(summary, self.updates_so_far)
            self.updates_so_far += 1

    # x_c, y_c, x are all [N, n]
    # returns mu_pred, Sig_pred
    def test(self, x_c, y_c, x):
        feed_dict = {
            self.context_y: y_c,
            self.context_x: x_c,
            self.x: x
        }
        mu_pred, Sig_pred = self.sess.run([self.mu_pred, self.Sig_pred], feed_dict)
        return mu_pred, Sig_pred

    # convenience function to use just the encoder on numpy input
    def encode(self, x):
        feed_dict = {
            self.x: x
        }
        return self.sess.run(self.phi, feed_dict)

    # ---- Save and Restore ------
    def save(self, model_path):
        save_path = self.saver.save(self.sess, model_path)
        print('Saved to:', save_path)

    def restore(self, model_path):
        self.saver.restore(self.sess, model_path)
        print('Restored model from:', model_path)



# given mat [a,b,c,...,N,N] and batch_v [a,b,c,...,M,N], returns [a,b,c,...,M,N]
def batch_matmul(mat, batch_v, name='batch_matmul'):
    with tf.name_scope(name):
        return tf.matrix_transpose(tf.matmul(mat,tf.matrix_transpose(batch_v)))

# works for A = [...,n,n] or [...,N,n,n]
# (uses the same matrix A for all N b vectors in the first case)
# assumes b = [...,N,n]
# returns  [...,N,1]
def batch_quadform(A, b):
    A_dims = A.get_shape().ndims
    b_dims = b.get_shape().ndims
    b_vec = tf.expand_dims(b, axis=-1)
    if A_dims == b_dims + 1:
        return tf.squeeze( tf.matrix_transpose(b_vec) @ A @ b_vec, axis=-1)
    elif A_dims == b_dims:
        Ab = tf.expand_dims( tf.matrix_transpose( A @ tf.matrix_transpose(b) ), axis=-1) # ... x N x n x 1
        return tf.squeeze( tf.matrix_transpose(b_vec) @ Ab, axis = -1) # ... x N x 1
    else:
        raise ValueError('Matrix size of %d is not supported.'%(A_dims))

# takes in y = (..., y_dim)
#          x = (..., x_dim)
# returns dydx = (..., y_dim, x_dim), the jacobian of y wrt x
def batch_2d_jacobian(y, x):
    shape = tf.shape(y)
    y_dim = y.get_shape().as_list()[-1]
    x_dim = x.get_shape().as_list()[-1]
    batched_y = tf.reshape(y, (-1, y_dim))
    batched_x = tf.reshape(x, (-1, x_dim))

    batched_dydx = gradients.batch_jacobian(y, x)

    dydx = tf.reshape(batched_dydx, tf.concat( (shape, [x_dim]), axis=0 ))
    return dydx

#------------ END General ALPaCA -------------#

def blr_update_np(K,L,X,Y):
    Ln_inv = np.linalg.inv( X.T @ X + L )
    Kn = Ln_inv @ (X.T @ Y + L @ K)
    return Kn, Ln_inv

def sampleMN(K, L_inv, Sig):
    mean = np.reshape(K.T, [-1])
    cov = np.kron(Sig, L_inv)
    K_vec = np.random.multivariate_normal(mean,cov)
    return np.reshape(K_vec, K.T.shape).T

def tp(x):
    return np.swapaxes(x, -1,-2)

def extract_x(xu, x_dim):
    xu_shape = tf.shape(xu)
    begin = tf.zeros_like(xu_shape)
    size = tf.concat( [ -1*tf.ones_like(xu_shape, dtype=tf.int32)[:-1], [x_dim] ], axis=0)
    x = tf.slice(xu, begin, size)
    return x


class AdaptiveDynamics(ALPaCA):
    def __init__(self, config, sess, graph=None, uncertainty_prop=False, preprocess=None, f_nom=None):
        if f_nom == None:
           f_nom = lambda xu: extract_x(xu, config['y_dim'])
                               
        super(AdaptiveDynamics, self).__init__(config, sess, graph, preprocess, f_nom)
        self.uncertainty_prop = uncertainty_prop

    def construct_model(self, uncertainty_prop=None):
        super(AdaptiveDynamics, self).construct_model()
        with self.graph.as_default():
            self.dphi_dxu = batch_2d_jacobian(self.phi, self.x)
            self.df_nom_dxu = batch_2d_jacobian(self.f_nom_x, self.x)

    def compute_pred_and_nll(self):    
        # Set up graph to do uncertainty prop / multistep predicitions if requested
        if self.uncertainty_prop:            
            x_dim = self.y_dim
            u = self.x[:,:,x_dim:] 
            initial_sig = tf.zeros((tf.shape(self.x)[0],x_dim,x_dim))

            mu_pred, Sig_pred = self.rollout_scan(self.x[:,0,:x_dim], initial_sig, u, resampling=True)

            # loss computation
            with tf.variable_scope('loss', reuse=None):
                logdet = tf.linalg.logdet(Sig_pred)
                Sig_pred_inv = tf.linalg.inv(Sig_pred)
                quadf = batch_quadform(Sig_pred_inv, (self.y - mu_pred))

                predictive_nll = logdet + quadf[:,:,0]
                
            # log stuff for summaries
            for t in range(self.config['test_horizon']):
                if t % 2 == 0:
                    rmse = tf.reduce_mean( tf.sqrt( tf.reduce_sum( tf.square(mu_pred - self.y)[:,t,:], axis=-1 ) ) )
                    mpv = tf.reduce_mean( tf.matrix_determinant( Sig_pred[:,t,:,:]) )
                    tf.summary.scalar('RMSE_'+str(t+1)+'_step', rmse)
                    tf.summary.scalar('MPV_'+str(t+1)+'_step', mpv)
        
            return mu_pred, Sig_pred, predictive_nll
        
        else:
            return super(AdaptiveDynamics, self).compute_pred_and_nll()
    
    # ---- TF utilities for prediction over multiple timesteps with uncertainty propagation ------
    def linearized_step(self, x, S, u, proc_noise=True,resampling=True):
        K = self.posterior_K
        Linv = self.posterior_L_inv

        xu = tf.concat([x,u],axis=-1) # (M, xdim+u_dim)
        
        f_nom = tf.zeros_like(x)
        if self.f_nom is not None:
            f_nom = self.f_nom(xu)
        
        with tf.variable_scope('phi', reuse=True):
            phi = self.basis(xu) # (M, phi_dim)
        phi_ed = tf.expand_dims(phi,axis=1) # (M, 1, phi_dim)
        
        xp = f_nom + batch_matmul( tf.matrix_transpose(K), phi_ed)[:,0,:] 
        
        dxp_dx = tf.stop_gradient( batch_2d_jacobian(xp, x) )
        
        # do we need f_nom_x? or can we work with dxp_dx only?
        f_nom_x = tf.stop_gradient( batch_2d_jacobian(f_nom, x) )
        phi_x = tf.stop_gradient( batch_2d_jacobian(phi,x) )
        
        pxK = tf.matrix_transpose(phi_x) @ K
        pxK_fnomx = pxK + tf.matrix_transpose(f_nom_x)
                
        Sp = dxp_dx @ S @ tf.matrix_transpose(dxp_dx) #tf.matrix_transpose(pxK_fnomx) @ S @ pxK_fnomx

        
        if proc_noise:
            Sp += self.SigEps[0,:,:,:] # check size on this        

        if resampling:
            # change shape of trace term so multiply works
            trace_term = tf.trace(tf.matrix_transpose(phi_x) @ Linv @ phi_x @ S)
            trace_term = tf.reshape(trace_term,[-1,1,1])            
            Sp += trace_term * self.SigEps[0,:,:,:]
            
            # same for quadform term
            quadf_term = batch_quadform(Linv, phi_ed)
            Sp += quadf_term * self.SigEps[0,:,:,:]
        
        return xp, Sp
    
    def rollout_scan(self, x0, S0, u, resampling=True):
        u_tp = tf.transpose(u,perm=[1,0,2])

        init_h = (x0,S0)
            
        def state_update(ht,ut):
            x,S = ht 
            return self.linearized_step(x,S,ut,resampling=resampling)      
            
        xs, Ss = tf.scan(state_update,u_tp,initializer=init_h)
        
        xs = tf.transpose(xs,perm=[1,0,2])
        Ss = tf.transpose(Ss,perm=[1,0,2,3])

        return (xs,Ss)
            
    # ---- Overloading ALPaCA methods to add gradients and extract useful quanitites from the TF graph ------      
    def encode(self, x, return_grad=False):
        if not return_grad:
            return self.sess.run( self.phi , {self.x: x})
        else:
            return self.sess.run( (self.phi, self.dphi_dxu) , {self.x: x})

    def train(self,dataset,num_train_updates):
        super(AdaptiveDynamics, self).train(dataset,num_train_updates)
        
        # obtain parameters of belief over K as np arrays
        self.K0,self.L0,self.sigeps = self.sess.run( (self.K, self.L, self.SigEps[0,0,:,:]) )
        self.reset_to_prior()

    def restore(self, model_path):
        super(AdaptiveDynamics, self).restore(model_path)
        
        # obtain parametrs of belief over K as np arrays
        self.K0,self.L0,self.sigeps = self.sess.run( (self.K, self.L, self.SigEps[0,0,:,:]) )
        self.reset_to_prior()

    # ---- Utilities for downstream use of the dynamics model -------
    def gen_predict_func(self):
        K = self.Ln_inv @ self.Qn
            
        def predict(x,u):
            x_inp = np.reshape( np.concatenate((x,u)), (1,1,-1) )
            phi = self.encode(x_inp, return_grad=False)
            phi = phi[0,0,:]
            f_nom = self.sess.run( self.f_nom_x, {self.x: x_inp} )
            f_nom = f_nom[0,0,:]
            return f_nom + K.T @ phi
        
        return predict
    
    # TODO: include f_nom rather than assuming f_nom = x
    def gen_linearize_func(self):
        K = self.Ln_inv @ self.Qn
        
        def linearize(x,u):
            batch_size = x.shape[0]
            x_dim = x.shape[-1]
            u_dim = u.shape[-1]
            x_inp = np.reshape( np.concatenate((x,u), axis=-1), (-1,1,x_dim+u_dim) )
            phi, dphi_dxu = self.encode(x_inp, return_grad=True)
            phi = phi[:,0,:]
            Ap = dphi_dxu[:,0,:,:x_dim]
            Bp = dphi_dxu[:,0,:,x_dim:]
            
            S = np.zeros((batch_size, 1+x_dim+u_dim, 1+x_dim+u_dim))
            Sig = np.zeros((batch_size, x_dim, x_dim))
            A = np.zeros((batch_size, x_dim, x_dim))
            B = np.zeros((batch_size, x_dim, u_dim))

            for i in range(batch_size):
                s = (1 + phi[i,:].T @ self.Ln_inv @ phi[i,:])
                Sx = Ap[i,:,:].T @ self.Ln_inv @ phi[i,:]
                Su = Bp[i,:,:].T @ self.Ln_inv @ phi[i,:]
                Sxx = Ap[i,:,:].T @ self.Ln_inv @ Ap[i,:,:]
                Suu = Bp[i,:,:].T @ self.Ln_inv @ Bp[i,:,:]
                Sux = Bp[i,:,:].T @ self.Ln_inv @ Ap[i,:,:]

                S[i,0,0] = s
                S[i,1:1+x_dim,0] = Sx
                S[i,0,1:1+x_dim] = Sx.T
                S[i,1+x_dim:,0] = Su
                S[i,0,1+x_dim:] = Su.T
                S[i,1:1+x_dim,1:1+x_dim] = Sxx
                S[i,1:1+x_dim,1+x_dim:] = Sux.T
                S[i,1+x_dim:,1:1+x_dim] = Sux
                S[i,1+x_dim:,1+x_dim:] = Suu

                A[i,:,:] = K.T @ Ap[i,:,:]
                B[i,:,:] = K.T @ Bp[i,:,:]


            Sig[:,:,:] = np.reshape(self.sigeps, (1,x_dim,x_dim))

            return Sig, S, A + np.eye(x_dim), B
        
        return linearize

    # TODO: include f_nom  rather than assuming f_nom = x (or remove gen_dynamics if not used)
    def gen_dynamics(self,sample_model=False):
        K = self.Ln_inv @ self.Qn
        if sample_model:
            K = sampleMN(K, self.Ln_inv, self.sigeps)

        def dynamics(x,u):
            x_dim = x.shape[-1]
            u_dim = u.shape[-1]
            x_inp = np.reshape( np.concatenate((x,u)), (1,1,-1) )
            phi, dphi_dxu = self.encode(x_inp, return_grad=True)
            phi = phi[0,0,:]
            A = dphi_dxu[0,0,:,:x_dim]
            B = dphi_dxu[0,0,:,x_dim:]
            S = np.zeros((1+x_dim+u_dim, 1+x_dim+u_dim))
            if sample_model:
                S[0,0] = 1.
            else:
                s = (1 + phi.T @ self.Ln_inv @ phi)
                Sx = A.T @ self.Ln_inv @ phi
                Su = B.T @ self.Ln_inv @ phi
                Sxx = A.T @ self.Ln_inv @ A
                Suu = B.T @ self.Ln_inv @ B
                Sux = B.T @ self.Ln_inv @ A

                S[0,0] = s
                S[1:1+x_dim,0] = Sx
                S[0,1:1+x_dim] = Sx.T
                S[1+x_dim:,0] = Su
                S[0,1+x_dim:] = Su.T
                S[1:1+x_dim,1:1+x_dim] = Sxx
                S[1:1+x_dim,1+x_dim:] = Sux.T
                S[1+x_dim:,1:1+x_dim] = Sux
                S[1+x_dim:,1+x_dim:] = Suu


            Sig = self.sigeps
            xp = x + K.T @ phi

            return xp, Sig, S, K.T @ A + np.eye(x_dim), K.T @ B

        return dynamics

    # returns the error of a transition prediction under each model K
    def error(self, x, u, xp, Ks):
        x_inp = np.reshape( np.concatenate((x,u)), (1,1,-1) )
        phi = self.encode(x_inp, return_grad=False)
        phi = phi.reshape((1,-1,1))
        xp_preds = np.squeeze( tp(Ks) @ phi, axis=-1 )
        mses = np.sum( (xp_preds - xp.reshape((1,-1)))**2, axis = -1)
        return mses

    # returns a list of N = num_models model parameter samples K1 .. KN
    def sample_dynamics_matrices(self, n_models=10):
        return np.array( [sampleMN(self.Ln_inv @ self.Qn, self.Ln_inv, self.sigeps) for _ in range(n_models)] )
    
    
    # applies actions starting at x0, simulating the mean dynamics under each
    # model parameter setting K in Ks
    # returns:
    #  x_pred, [N_samples, T+1, x_dim], mean trajectory under each mdoel
    #  A, [N_samples, T, x_dim, x_dim], (dx'/dx) evaluated at (x_pred[t],
    #         action[t]) for every t but the last and every model
    #  B, [N_samples, T, x_dim, u_dim], (dx'/du)
    def sample_rollout(self, x0, actions, Ks, resample=False):
        T, u_dim = actions.shape
        x_dim = x0.shape[0]
        N_samples = len(Ks)

        actions = np.tile(np.expand_dims(actions,axis=0), (N_samples, 1, 1))
        x0 = np.expand_dims(x0, axis=0)

        x_pred = np.zeros( (N_samples, T+1, x_dim) )
        x_pred[:,0,:] = x0

        A = np.zeros( (N_samples, T, x_dim, x_dim) )
        B = np.zeros( (N_samples, T, x_dim, u_dim) )

        for t in range(0, T):
            x_inp = np.concatenate( (x_pred[:,t:t+1,:], actions[:,t:t+1,:]), axis=2 )
            phi, dphi_dxu = self.encode(x_inp, return_grad=True)
            for j in range(N_samples):
                K = Ks[j]
                if resample:
                    K = sampleMN(self.Ln_inv @ self.Qn, self.Ln_inv, self.sigeps)
                A[j,t,:,:] = K.T @ dphi_dxu[j,0,:,:x_dim]
                B[j,t,:,:] = K.T @ dphi_dxu[j,0,:,x_dim:]
                x_pred[j,t+1,:] = x_pred[j,t,:] + np.squeeze( phi[j,:,:] @ K )

        return x_pred #, A, B
    
    def reset_to_prior(self):
        self.Ln_inv = np.linalg.inv(self.L0)
        self.Qn = self.L0 @ self.K0

    # TODO: need to use f_nom here, not assume f_nom = x
    def incorporate_transition(self,x,u,xp):
        # perform RLS update to Kn, Ln
        x_inp = np.reshape( np.concatenate( (x,u), axis=0 ), (1,1,-1) )
        phi = self.encode(x_inp)[0,:,:].T
        y = np.reshape(xp - x, (-1,1))

        phiLamphi = (phi.T @ self.Ln_inv @ phi)[0,0]
        LninvPhi = self.Ln_inv @ phi
        self.Ln_inv = self.Ln_inv - 1./(1. + phiLamphi) * LninvPhi @ LninvPhi.T
        self.Qn = phi @ y.T + self.Qn

    # TODO: change to assume Y is yhat + f_nom
    def incorporate_batch(self, X, Y):
        Phi = self.encode(X)
        Kn,Ln_inv = blr_update_np(self.K0,self.L0,Phi[0,:,:],Y[0,:,:])
        self.Ln_inv = Ln_inv
        self.Kn = Kn
        self.Qn = np.linalg.inv(self.Ln_inv) @ self.Kn
