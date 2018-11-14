import numpy as np
from matplotlib import rc
## for Palatino and other serif fonts use:
rc('font',**{'family':'serif','serif':['Computer Modern Roman']})
rc('text', usetex=True)
import matplotlib.pyplot as plt
import time


#NLL plotting
def nll_plot(nll_mean_list1,nll_var_list1,nll_mean_list2,nll_var_list2,nll_mean_list3,nll_var_list3,N_test,legend=False,last_legend_label=r'GPR'):
    
    legend_label = []
    if nll_mean_list1 is not None:
        plt.gca().set_prop_cycle(None)

        conf_list1 = [1.96*np.sqrt(s)/np.sqrt(N_test) for s in nll_var_list1]
        upper1 = [y + c for y,c in zip(nll_mean_list1,conf_list1)]
        lower1 = [y - c for y,c in zip(nll_mean_list1,conf_list1)]
        plt.fill_between(range(0,len(nll_mean_list1)), upper1, lower1, alpha=.2)
        plt.plot(range(0,len(nll_mean_list1)),nll_mean_list1)
        legend_label.append(r'ALPaCA')
        plt.ylabel('Negative Log Likelihood')
        
    if nll_mean_list2 is not None:
        conf_list2 = [1.96*np.sqrt(s)/np.sqrt(N_test) for s in nll_var_list2]
        upper2 = [y + c for y,c in zip(nll_mean_list2,conf_list2)]
        lower2 = [y - c for y,c in zip(nll_mean_list2,conf_list2)]
        plt.fill_between(range(0,len(nll_mean_list2)), upper2, lower2, alpha=.2)
        plt.plot(range(0,len(nll_mean_list2)),nll_mean_list2)
        legend_label.append(r'ALPaCA (no meta)')
        
    if nll_mean_list3 is not None:
        conf_list3 = [1.96*np.sqrt(s)/np.sqrt(N_test) for s in nll_var_list3]
        upper3 = [y + c for y,c in zip(nll_mean_list3,conf_list3)]
        lower3 = [y - c for y,c in zip(nll_mean_list3,conf_list3)]
        plt.fill_between(range(0,len(nll_mean_list3)), upper3, lower3, alpha=.2)
        plt.plot(range(0,len(nll_mean_list3)),nll_mean_list3)
        legend_label.append(last_legend_label)
    
    if legend==True:
        plt.legend(legend_label)
        
    plt.xlabel('Timesteps')
    
def mse_plot(nll_mean_list1,nll_var_list1,nll_mean_list2,nll_var_list2,nll_mean_list3,nll_var_list3,N_test,legend=False):
    
    legend_label = []
    if nll_mean_list1 is not None:
        plt.gca().set_prop_cycle(None)

        conf_list1 = [1.96*np.sqrt(s)/np.sqrt(N_test) for s in nll_var_list1]
        upper1 = [y + c for y,c in zip(nll_mean_list1,conf_list1)]
        lower1 = [y - c for y,c in zip(nll_mean_list1,conf_list1)]
        plt.fill_between(range(0,len(nll_mean_list1)), upper1, lower1, alpha=.2)
        l1 = plt.plot(range(0,len(nll_mean_list1)),nll_mean_list1,label=r'ALPaCA')
        legend_label.append(r'ALPaCA')
        plt.ylabel('MSE')
        
    if nll_mean_list2 is not None:
        conf_list2 = [1.96*np.sqrt(s)/np.sqrt(N_test) for s in nll_var_list2]
        upper2 = [y + c for y,c in zip(nll_mean_list2,conf_list2)]
        lower2 = [y - c for y,c in zip(nll_mean_list2,conf_list2)]
        plt.fill_between(range(0,len(nll_mean_list2)), upper2, lower2, alpha=.2)
        l2 = plt.plot(range(0,len(nll_mean_list2)),nll_mean_list2, label=r'MAML (1 step)')
        legend_label.append(r'MAML (1 step)')
        
    if nll_mean_list3 is not None:
        conf_list3 = [1.96*np.sqrt(s)/np.sqrt(N_test) for s in nll_var_list3]
        upper3 = [y + c for y,c in zip(nll_mean_list3,conf_list3)]
        lower3 = [y - c for y,c in zip(nll_mean_list3,conf_list3)]
        plt.fill_between(range(0,len(nll_mean_list3)), upper3, lower3, alpha=.2)
        plt.plot(range(0,len(nll_mean_list3)),nll_mean_list3, label=r'MAML (5 step)')
        legend_label.append(r'GPR')
    
    if legend==True:
        plt.legend()
        
    plt.xlabel('Timesteps')
    
    
def time_plot(nll_mean_list1,nll_var_list1,nll_mean_list2,nll_var_list2,nll_mean_list3,nll_var_list3,N_test,legend=False):
    #same arguments cause I'm lazy
    
    legend_label = []
    if nll_mean_list1 is not None:
        plt.gca().set_prop_cycle(None)

        conf_list1 = [1.96*np.sqrt(s)/np.sqrt(N_test) for s in nll_var_list1]
        upper1 = [y + c for y,c in zip(nll_mean_list1,conf_list1)]
        lower1 = [y - c for y,c in zip(nll_mean_list1,conf_list1)]
        plt.fill_between(range(0,len(nll_mean_list1)), upper1, lower1, alpha=.2)
        plt.plot(range(0,len(nll_mean_list1)),nll_mean_list1)
        legend_label.append(r'ALPaCA')
        plt.ylabel(r'Time (s)')
        
    if nll_mean_list2 is not None:
        conf_list2 = [1.96*np.sqrt(s)/np.sqrt(N_test) for s in nll_var_list2]
        upper2 = [y + c for y,c in zip(nll_mean_list2,conf_list2)]
        lower2 = [y - c for y,c in zip(nll_mean_list2,conf_list2)]
        plt.fill_between(range(0,len(nll_mean_list2)), upper2, lower2, alpha=.2)
        plt.plot(range(0,len(nll_mean_list2)),nll_mean_list2)
        legend_label.append(r'ALPaCA (no meta)')
        
    if nll_mean_list3 is not None:
        conf_list3 = [1.96*np.sqrt(s)/np.sqrt(N_test) for s in nll_var_list3]
        upper3 = [y + c for y,c in zip(nll_mean_list3,conf_list3)]
        lower3 = [y - c for y,c in zip(nll_mean_list3,conf_list3)]
        plt.fill_between(range(0,len(nll_mean_list3)), upper3, lower3, alpha=.2)
        plt.plot(range(0,len(nll_mean_list3)),nll_mean_list3)
        legend_label.append(r'GPR')
    
    if legend==True:
        plt.legend(legend_label)
        
    plt.xlabel('Timesteps')
    
    
def sinusoid_plot(freq,phase,amp,x_list,sigma_list,y_list,X_update, Y_update,sampling_density=101,legend_labels=['Ours', 'True']):
    """
    x,y,sigma should be lists
    """

    #plot given data
    conf_list = [1.96*np.sqrt(s) for s in sigma_list]
    upper = [y + c for y,c in zip(y_list,conf_list)]
    lower = [y - c for y,c in zip(y_list,conf_list)]
    plt.fill_between(x_list, upper, lower, alpha=.5)
    plt.plot(x_list,y_list)
    
    
    
    #plot true sinusoid
    yr_list = [amp*np.sin(freq*x + phase) for x in x_list]
    plt.plot(x_list,yr_list,color='r')
    
    # plot update points
    plt.plot(X_update[0,:,0],Y_update[0,:,0],'+',color='k',markersize=10)
    plt.xlim([np.min(x_list), np.max(x_list)])
    
    #legend
    if legend_labels:
        plt.legend(legend_labels + ['sampled points'])
    
def gen_sin_fig(agent, X,Y,freq,phase,amp,upper_x=5,lower_x=-5,point_every=0.1, label=None):
    y_list = []
    x_list = []
    s_list = []
    for p in np.arange(lower_x,upper_x,0.1):
        y, s = agent.test(X, Y, [[[p]]])
        y_list.append(y[0,0,0])
        x_list.append(p)
        if s:
            s_list.append(s[0,0,0,0])
        else:
            s_list.append(0)
    legend_labels = None
    if label:
        legend_labels = [label, 'True']
    sinusoid_plot(freq,phase,amp,x_list,s_list,y_list,X,Y, legend_labels=legend_labels)

def gen_sin_gp_fig(agent, X,Y,freq,phase,amp,upper_x=5,lower_x=-5,point_every=0.1, label=None):
    x_test = np.reshape( np.arange(lower_x,upper_x,0.1), [1,-1,1] )
    y,s = agent.test(X,Y,x_test)
    y = y[0,:,0]
    s = s[0,:]**2
    legend_labels = None
    if label:
        legend_labels = [label, 'True']
    sinusoid_plot(freq,phase,amp,x_test[0,:,0],s,y,X,Y,legend_labels=legend_labels)
    
def plot_bases(x,y,indices):
    x = x[0,:,0]
    y = y[0,:,:]
    for i in indices:
        plt.figure()
        plt.plot(x,y[:,i])
        plt.legend([r"$\phi_{"+ str(i) +r"}(x)$"])
        plt.show()
        
def gen_sin_bases_fig(agent, sess, x, n_bases):
    phi = sess.run( agent.phi, {agent.x: x} )
    plot_bases(x, phi, np.random.choice(agent.config['nn_layers'][-1],n_bases))
    
        
def plot_sample_fns(x,phi,K,L,SigEps,n_samples):
    x = x[0,:,0]
    phi = phi[0,:,:]
    
    mean = np.reshape(K, [-1])
    cov = np.kron(SigEps, np.linalg.inv(L))
    K_vec = np.random.multivariate_normal(mean,cov,n_samples)
    plt.figure()
    for i in range(n_samples):
        K = np.reshape(K_vec[i,:], K.shape)
        y = np.squeeze(phi @ K)
        plt.plot(x,y)
    plt.show()

# STEP FUNCTIONS
def step_plot(x_jump,x_list,sigma_list,y_list,X_update, Y_update,sampling_density=101,legend_labels=['Ours', 'True']):
    """
    x,y,sigma should be lists
    """

    #plot given data
    conf_list = [1.96*np.sqrt(s) for s in sigma_list]
    upper = [y + c for y,c in zip(y_list,conf_list)]
    lower = [y - c for y,c in zip(y_list,conf_list)]
    plt.fill_between(x_list, upper, lower, alpha=.5)
    plt.plot(x_list,y_list)
    
    
    
    #plot true step
    yr_list = [0.5 + 0.5*np.sign(x-x_jump) for x in x_list]
    plt.plot(x_list,yr_list,color='r')
    
    # plot update points
    plt.plot(X_update[0,:,0],Y_update[0,:,0],'+',color='k',markersize=10)
    plt.xlim([np.min(x_list), np.max(x_list)])
    plt.ylim([-1,2])
    
    #legend
    if legend_labels:
        plt.legend(legend_labels + ['sampled points'])
        
def multistep_plot(pt_list,x_list,sigma_list,y_list,X_update, Y_update,sampling_density=101,legend_labels=['Ours', 'True']):
    """
    x,y,sigma should be lists
    """

    #plot given data
    conf_list = [1.96*np.sqrt(s) for s in sigma_list]
    upper = [y + c for y,c in zip(y_list,conf_list)]
    lower = [y - c for y,c in zip(y_list,conf_list)]
    plt.fill_between(x_list, upper, lower, alpha=.5)
    plt.plot(x_list,y_list)
    
    
    
    #plot true step
    #yr_list = []
    x = np.reshape(x_list,[1,-1])
    step_pts = np.reshape(pt_list,[-1,1])
    y = 2.*np.logical_xor.reduce( x > step_pts, axis=0) - 1.
    yr_list = y
    
#     for x in x_list:
#         for i in range(len(pt_list)):

#             if x<pt_list[0]:
#                 yr_list.append(((i)%2)*2-1.0)
#                 break
                
#             if i==(len(pt_list)-1) and x>pt_list[-1]:
# #                 print('ok')
#                 yr_list.append(((i+1)%2)*2-1.0)
#                 break
                
#             if x>pt_list[i] and x<pt_list[i+1]:
#                 yr_list.append(((i+1)%2)*2-1.0)
#                 break
                
    plt.plot(x_list,yr_list,color='r')
    
    # plot update points
    plt.plot(X_update[0,:,0],Y_update[0,:,0],'+',color='k',markersize=10)
    plt.xlim([np.min(x_list), np.max(x_list)])
    plt.ylim([-2,2])
    
    #legend
    if legend_labels:
        plt.legend(legend_labels + ['sampled points'])
        
        
#do plotting
def gen_step_fig(agent,X,Y,x_jump,upper_x=5,lower_x=-5,point_every=0.1, label=None):
    y_list = []
    x_list = []
    s_list = []
    for p in np.arange(lower_x,upper_x,0.1):
        y, s = agent.test(X, Y, [[[p]]])
        y_list.append(y[0,0,0])
        x_list.append(p)
        if s:
            s_list.append(s[0,0,0,0])
        else:
            s_list.append(0)
        
    legend_labels = None
    if label:
        legend_labels = [label, 'True']
    step_plot(x_jump,x_list,s_list,y_list,X,Y, legend_labels=legend_labels)
    

def gen_step_gp_fig(agent, X, Y, x_jump, upper_x=5,lower_x=-5,point_every=0.1, label=None):
    x_test = np.reshape( np.arange(lower_x,upper_x,0.1), [1,-1,1] )
    y,s = agent.test(X,Y,x_test)
    y = y[0,:,0]
    s = s[0,:]**2
    legend_labels = None
    if label:
        legend_labels = [label, 'True']
    step_plot(x_jump,x_test[0,:,0],s,y,X,Y,legend_labels=legend_labels)

def gen_multistep_fig(agent, X,Y,x_jump,upper_x=5,lower_x=-5,point_every=0.1, label=None):
    y_list = []
    x_list = []
    s_list = []
    for p in np.arange(lower_x,upper_x,0.1):
        y, s = agent.test(X, Y, [[[p]]])
        y_list.append(y[0,0,0])
        x_list.append(p)
        if s:
            s_list.append(s[0,0,0,0])
        else:
            s_list.append(0)
        
    legend_labels = None
    if label:
        legend_labels = [label, 'True']
    multistep_plot(x_jump,x_list,s_list,y_list,X,Y, legend_labels=legend_labels)

def gen_multistep_gp_fig(agent, X, Y, x_jump, upper_x=5,lower_x=-5,point_every=0.1, label=None):
    x_test = np.reshape( np.arange(lower_x,upper_x,0.1), [1,-1,1] )
    y,s = agent.test(X,Y,x_test)
    y = y[0,:,0]
    s = s[0,:]**2
    legend_labels = None
    if label:
        legend_labels = [label, 'True']
    multistep_plot(x_jump,x_test[0,:,0],s,y,X,Y,legend_labels=legend_labels)

    
# PENDULUM
def plot_trajectory(X,Y,Y_pred,Sig_pred):
    t = np.arange(Y.shape[1] + 1)
    Nu = Y.shape[1] - Y_pred.shape[1] 
    dims = [0,1]
    colors = ['b','r']
    h_list = []
    for i in dims:
        x = np.concatenate( ( X[0,0:1,i], X[0,:,i] + Y[0,:,i] ) )
        x_pred = np.concatenate( (X[0,:Nu+1,i], X[0,Nu:,i] + Y_pred[0,:,i] ) )
        s = np.concatenate( ( np.zeros([Nu+1]), Sig_pred[0,:,i,i] ) )
        c = 1.96*np.sqrt(s)
        u = x_pred + c
        l = x_pred - c
        
        plt.fill_between(t, u, l, alpha=.2, color=colors[i])
        h,  = plt.plot(t, x_pred, color=colors[i],label=r"$x_{"+str(i+1)+"}$", zorder=1)
        h_list.append(h)
        plt.scatter(t, x, marker='+', color=colors[i],zorder=1)
    
    plt.legend(handles=h_list)
    ax = plt.gca()
    ylim = ax.get_ylim()
    plt.plot([Nu, Nu],ylim,linestyle=':',color='k', alpha=0.8)
    plt.fill_between([0,Nu],[ylim[1],ylim[1]],[ylim[0], ylim[0]],color='white',alpha=0.7,zorder=2)
    plt.xlim([t[0],t[-1]])
    plt.ylim(ylim)
    plt.show()
    
def gen_pendulum_onestep_fig(agent,sess,X,Y,Nu,T=None):
    ux = X[:,:Nu,:]
    uy = Y[:,:Nu,:]
    y_pred, sig_pred = agent.test(sess, ux, uy, X[:,Nu:T+1,:])
    
    plot_trajectory(X[:,:T+1,:],Y[:,:T+1,:],y_pred,sig_pred)
    

def sampleMN(K, L_inv, Sig):
    mean = np.reshape(K.T, [-1])
    cov = np.kron(Sig, L_inv)
    K_vec = np.random.multivariate_normal(mean,cov)
    return np.reshape(K_vec, K.T.shape).T
    
def gen_pendulum_sample_fig(agent, X,Y,Nu,N_samples=10,T=None, T_rollout=10,no_update=False):
    if not T:
        T = Y.shape[1]
        
    tt = np.arange(T+1)
    
    ux = X[0:1,:Nu,:]
    uy = Y[0:1,:Nu,:]
    
    K0 = sess.run(agent.K)
    L0 = sess.run(agent.L)
    SigEps = sess.run(agent.SigEps)
    
    Phi = sess.run( agent.phi, {agent.x: X} )
    uPhi = Phi[0:1,:Nu,:]
    
    Kn = K0
    Ln = L0
    Ln_inv = np.linalg.inv(Ln)
    if Nu > 0 and not no_update:
        Kn,Ln_inv = agent.batch_update_np(K0,L0,uPhi[0,:,:],uy[0,:,:])
        Ln = np.linalg.inv(Ln_inv)
    
    x = np.concatenate( ( X[0,0:1,:2], X[0,:T,:2] + Y[0,:T,:] ) )
    x_pred = np.zeros([N_samples, T+1, X.shape[2]])
    print(np.shape(x_pred[:,:Nu+1,:]))
    print(np.shape(x_pred))
    print(np.shape(X[0:1, :Nu+1, :]))
    x_pred[:,:Nu+1,:] = X[0:1, :Nu+1, :]
    print(np.shape(x_pred))
    for j in range(N_samples):
        K = sampleMN(Kn,Ln_inv,SigEps)
#         print(K)
        for t in range(Nu,Nu+T_rollout):
            phi_t = sess.run( agent.phi, {agent.x: x_pred[j:j+1, t:t+1, :]})
            x_pred[j,t+1,:2] = x_pred[j,t,:2] + np.squeeze( phi_t[0,:,:] @ K )
    
    dims = [0,1]
    colors = ['b','r']
    styles=['-',':']
    for i in dims:
        for j in range(N_samples):
            plt.plot(tt[Nu:Nu+T_rollout], x_pred[j,Nu:Nu+T_rollout,i], color=colors[i], alpha=5.0/N_samples)
        plt.plot(tt, x[:,i], linestyle=styles[i], color='k')
    
    ax = plt.gca()
    ylim = [np.min(x)-2,np.max(x)+2]
    #plt.plot([Nu, Nu],ylim,linestyle=':',color='k', alpha=0.8)
    #plt.fill_between([0,Nu],[ylim[1],ylim[1]],[ylim[0], ylim[0]],color='white',alpha=0.7,zorder=2)
    plt.xlim([tt[0],tt[-1]])
    plt.ylim(ylim)
    #plt.show()
    
    
def gen_pendulum_rollout_fig(agent, xu, xp, Nu, N_samples=50, T=None, T_rollout=10, update=True):
    if not T:
        T = xp.shape[1]
    
    tt = np.arange(T+1)
    
    x_dim = xp.shape[-1]
    x = xu[:,:,:x_dim]
    u = xu[:,:,x_dim:]
    
    agent.reset_to_prior()
    if update:
        for t in range(Nu):
            agent.incorporate_transition(x[0,t,:], u[0,t,:], xp[0,t,:])

    Ks = agent.sample_dynamics_matrices(N_samples)
    x_pred = agent.sample_rollout(x[0,Nu,:],u[0,Nu:Nu+T_rollout,:],Ks)
    
    dims = [0,1]
    colors = ['b','r']
    styles=['-',':']
    for i in dims:
        for j in range(N_samples):
            plt.plot(tt[Nu:Nu+T_rollout+1], x_pred[j,:,i], color=colors[i], alpha=5.0/N_samples)
        plt.plot(tt, x[0,:T+1,i], linestyle=styles[i], color='k')
        
    ax = plt.gca()
    ylim = [np.min(x)-2,np.max(x)+2]
    plt.xlim([tt[0],tt[-1]])
    plt.ylim(ylim)        
        
def gen_pendulum_sample_fig_gp(X,Y,Nu,N_samples=10,T=None, T_rollout=10):
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, ConstantKernel
    
    2.5**2*RBF(0.5, (1e-1, 1e0))
    
    if not T:
        T = Y.shape[1]
        
    tt = np.arange(T+1)
    


    kernel = 2.5**2*RBF(0.5, (1e-1, 1e0))
    
    x = np.concatenate( ( X[0,0:1,:2], X[0,:T,:2] + Y[0,:T,:] ) )
    x_pred = np.zeros([N_samples, T+1, X.shape[2]])
    x_pred[:,:Nu+1,:] = X[0:1, :Nu+1, :]
    for j in range(N_samples):
        ux = X[0,:Nu,:]
        uy = Y[0,:Nu,:]
        for t in range(Nu,Nu+T_rollout):
            xt = x_pred[j,t,:]
            
            gp0 = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=0)
            if ux.shape[0] > 0:
                gp0.fit(ux, uy[:,0])
            gp1 = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=0)
            if ux.shape[0] > 0:
                gp1.fit(ux, uy[:,1])
            
            y_samp = np.zeros((1, 2))
            y_samp[0,0] = gp0.sample_y(x_pred[j,t:t+1,:],random_state=None)
            y_samp[0,1] = gp1.sample_y(x_pred[j,t:t+1,:],random_state=None)
            ux = np.concatenate( (ux, [xt]), axis=0 )
            uy = np.concatenate( (uy, y_samp), axis=0 )
            
            x_pred[j,t+1,:2] = xt[:2] + y_samp
    
    dims = [0,1]
    colors = ['b','r']
    styles=['-',':']
    for i in dims:
        for j in range(N_samples):
            plt.plot(tt[Nu:Nu+T_rollout], x_pred[j,Nu:Nu+T_rollout,i], color=colors[i], alpha=5.0/N_samples)
        plt.plot(tt, x[:,i], linestyle=styles[i], color='k')
    
    ax = plt.gca()
    ylim = [np.min(x)-2,np.max(x)+2]
    #plt.plot([Nu, Nu],ylim,linestyle=':',color='k', alpha=0.8)
    #plt.fill_between([0,Nu],[ylim[1],ylim[1]],[ylim[0], ylim[0]],color='white',alpha=0.7,zorder=2)
    plt.xlim([tt[0],tt[-1]])
    plt.ylim(ylim)
    #plt.show()
    
def test_adaptive_dynamics(agent, xu, xp, N_samples, Nu, T_rollout=30):
    agent.reset_to_prior()
    T = xp.shape[1]
    x_dim = xp.shape[2]
    u_dim = xu.shape[2] - x_dim
    
    tt = np.arange(T)
    
    u = xu[0,:,x_dim:]
    x = xu[0,:,:x_dim]

    for t in range(Nu):
        agent.incorporate_transition(x[t,:], u[t,:], xp[0,t,:])
    
    Ks = agent.sample_dynamics_matrices(N_samples)
    x_pred = agent.sample_rollout(x[Nu,:], u[Nu:Nu+T_rollout,:], Ks)
    
    dims = range(x_dim)
    colors = ['b','r','g']
    styles=['-',':','-.']
    N_dims = len(dims)
    for i,d in enumerate(dims):
        plt.subplot(int( np.ceil( N_dims*1./2 )), 2, i+1)
        for j in range(N_samples):
            plt.plot(range(Nu, Nu+T_rollout+1), x_pred[j,:,d], color='C0', alpha=5.0/N_samples)
        plt.plot(tt, x[:T,d], linestyle=':', color='k')
        ylims = np.min(x[:,d]) + (np.max(x[:,d]) - np.min(x[:,d]))*np.array([-0.08, 1.08])
        plt.ylabel(r"$x_{" + str(d+1) + r"}(k)$")
        plt.ylim(ylims)
        plt.xlim([tt[0],tt[-1]])
        ax  = plt.gca()
        #ax.yaxis.set_label_coords(-0.1, 0.5)
        if i % 2 == 1:
            ax.yaxis.tick_right()
            ax.yaxis.set_label_position("right")
            #ax.yaxis.set_label_coords(1.12, 0.5)
        if i >= N_dims - 2:
            plt.xlabel(r"$k$")
        else:
            plt.setp(ax.get_xticklabels(), visible=False)
    