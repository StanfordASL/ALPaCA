import numpy as np
import gym
import tqdm

class Dataset:
    def __init__(self):
        pass

    # draw n_sample (x,y) pairs drawn from n_func functions
    # returns (x,y) where each has size [n_func, n_samples, x/y_dim]
    def sample(self, n_funcs, n_samples):
        raise NotImplementedError

class PresampledDataset(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
        self.x_dim = X.shape[-1]
        self.y_dim = Y.shape[-1]
        self.N = X.shape[0]
        self.T = X.shape[1]
    
    def sample(self, n_funcs, n_samples):
        x = np.zeros((n_funcs, n_samples, self.x_dim))
        y = np.zeros((n_funcs, n_samples, self.y_dim))
        
        for i in range(n_funcs):
            j = np.random.randint(self.N)
            if n_samples > self.T:
                raise ValueError('You are requesting more samples than are in the dataset.')
 
            inds_to_keep = np.random.choice(self.T, n_samples)
            x[i,:,:] = self.X[j,inds_to_keep,:]
            y[i,:,:] = self.Y[j,inds_to_keep,:]
        
        return x,y
        
class PresampledTrajectoryDataset(Dataset):
    def __init__(self, trajs, controls):
        self.trajs = trajs
        self.controls = controls
        self.o_dim = trajs[0].shape[-1]
        self.u_dim = controls[0].shape[-1]
        self.N = len(trajs)
    
    def sample(self, n_funcs, n_samples):
        o_dim = self.o_dim
        u_dim = self.u_dim
        x_dim = o_dim + u_dim
        y_dim = o_dim
        x = np.zeros((n_funcs, n_samples, x_dim))
        y = np.zeros((n_funcs, n_samples, y_dim))
        
        for i in range(n_funcs):
            j = np.random.randint(self.N)
            T = self.controls[j].shape[0]
            if n_samples > T:
                raise ValueError('You are requesting more samples than are in this trajectory.')
            start_ind = 0
            if T > n_samples:
                start_ind = np.random.randint(T-n_samples)
            inds_to_keep = np.arange(start_ind, start_ind+n_samples)
            x[i,:,:self.o_dim] = self.trajs[j][inds_to_keep]
            x[i,:,self.o_dim:] = self.controls[j][inds_to_keep]
            y[i,:,:] = self.trajs[j][inds_to_keep+1] #- self.trajs[j][inds_to_keep]
        
        return x,y

class SinusoidDataset(Dataset):
    def __init__(self, config, noise_var=None, rng=None):
        self.amp_range = config['amp_range']
        self.phase_range = config['phase_range']
        self.freq_range = config['freq_range']
        self.x_range = config['x_range']
        if noise_var is None:
            self.noise_std = np.sqrt( config['sigma_eps'] )
        else:
            self.noise_std = np.sqrt( noise_var )
            
        self.np_random = rng
        if rng is None:
            self.np_random = np.random

    def sample(self, n_funcs, n_samples, return_lists=False):
        x_dim = 1
        y_dim = 1
        x = np.zeros((n_funcs, n_samples, x_dim))
        y = np.zeros((n_funcs, n_samples, y_dim))

        amp_list = self.amp_range[0] + self.np_random.rand(n_funcs)*(self.amp_range[1] - self.amp_range[0])
        phase_list = self.phase_range[0] + self.np_random.rand(n_funcs)*(self.phase_range[1] - self.phase_range[0])
        freq_list = self.freq_range[0] + self.np_random.rand(n_funcs)*(self.freq_range[1] - self.freq_range[0])
        for i in range(n_funcs):
            x_samp = self.x_range[0] + self.np_random.rand(n_samples)*(self.x_range[1] - self.x_range[0])
            y_samp = amp_list[i]*np.sin(freq_list[i]*x_samp + phase_list[i]) + self.noise_std*self.np_random.randn(n_samples)

            x[i,:,0] = x_samp
            y[i,:,0] = y_samp

        if return_lists:
            return x,y,freq_list,amp_list,phase_list

        return x,y
    
class MultistepDataset(Dataset):
    def __init__(self, config, noise_var=None, rng=None):
        self.step_min = config['step_min']
        self.step_max = config['step_max']
        self.num_steps = config['num_steps']
        self.x_range = config['x_range']
        if noise_var is None:
            self.noise_std = np.sqrt( config['sigma_eps'] )
        else:
            self.noise_std = np.sqrt( noise_var )
            
        self.np_random = rng
        if rng is None:
            self.np_random = np.random
            
    def sample(self, n_funcs, n_samples, return_lists=False):
        x_dim = 1
        y_dim = 1
        x = np.zeros((n_funcs, n_samples, x_dim))
        y = np.zeros((n_funcs, n_samples, y_dim))
        
        step_mat = np.zeros((n_funcs, self.num_steps))
        
        for i in range(n_funcs):
            step_pts = self.step_min + self.np_random.rand(self.num_steps)* (self.step_max - self.step_min)
            step_mat[i,:] = step_pts
            
            x_samp = self.x_range[0] + self.np_random.rand(n_samples)*(self.x_range[1] - self.x_range[0])
            y_samp = self.multistep(x_samp, step_pts)

            x[i,:,0] = x_samp
            y[i,:,0] = y_samp

        if return_lists:
            return x,y,step_mat

        return x,y
    
    def multistep(self, x, step_pts):
        x = x.reshape([1,-1])
        step_pts = step_pts.reshape([-1,1])
        y = 2.*np.logical_xor.reduce( x > step_pts, axis=0) - 1.
        y += self.noise_std*self.np_random.randn(x.shape[1])
        return y

# Assumes env has a forward_dynamics(x,u) function
class GymUniformSampleDataset(Dataset):
    def __init__(self, env):
        self.env = env
        self.o_dim = env.observation_space.shape[-1]
        self.u_dim = env.action_space.shape[-1]

    def sample(self, n_funcs, n_samples):
        o_dim = self.o_dim
        u_dim = self.u_dim
        x_dim = o_dim + u_dim
        y_dim = o_dim
        x = np.zeros((n_funcs, n_samples, x_dim))
        y = np.zeros((n_funcs, n_samples, y_dim))

        for i in range(n_funcs):
            self.env.reset()
            for j in range(n_samples):
                s = self.env.get_ob_sample()
                a = self.env.action_space.sample()#get_ac_sample()
                sp = self.env.forward_dynamics(s,a)
                
                x[i,j,:o_dim] = s
                x[i,j,o_dim:] = a
                y[i,j,:] = sp

        return x,y

# wraps a gym env + policy as a dataset
# assumes that the gym env samples parameters from the prior upon reset
class GymDataset(Dataset):
    def __init__(self, env, policy, state_dim=None):
        self.env = gym.wrappers.TimeLimit(env, max_episode_steps=100)
        self.policy = policy
        self.use_state = False
        self.o_dim = env.observation_space.shape[-1]
        if state_dim is not None:
            self.use_state = True
            self.o_dim = state_dim
        self.u_dim = env.action_space.shape[-1]

    def sample(self, n_funcs, n_samples, shuffle=False, verbose=False):
        o_dim = self.o_dim
        u_dim = self.u_dim
        x_dim = o_dim + u_dim
        y_dim = o_dim
        x = np.zeros((n_funcs, n_samples, x_dim))
        y = np.zeros((n_funcs, n_samples, y_dim))

        
        pbar = tqdm.tqdm(disable=(not verbose), total=n_funcs)
        for i in range(n_funcs):
            # sim a trajectory
            x_traj = []
            u_traj = []
            xp_traj = []

            ob = self.env.reset()
            if self.use_state:
                s = self.env.unwrapped.state
            done = False
            while not done:
                ac = self.policy(ob)
                obp, _, done, _ = self.env.step(ac)
                
                if self.use_state:
                    sp = self.env.unwrapped.state
                    x_traj.append(s)
                    u_traj.append(ac)
                    xp_traj.append(sp)
                else:
                    x_traj.append(ob)
                    u_traj.append(ac)
                    xp_traj.append(obp)

                ob = obp
                if self.use_state:
                    s = sp

            T = len(x_traj)
            if T < n_samples:
                print('episode did not last long enough')
                #n_samples = T-1
                i -= 1
                continue

            if shuffle:
                inds_to_keep = np.random.choice(T, n_samples)
            else:
                start_ind = 0 #np.random.randint(T-n_samples)
                inds_to_keep = range(start_ind, start_ind+n_samples)
            x[i,:,:o_dim] = np.array(x_traj)[inds_to_keep,:]
            x[i,:,o_dim:] = np.array(u_traj)[inds_to_keep,:]
            y[i,:,:] = np.array(xp_traj)[inds_to_keep,:]
            
            pbar.update(1)

        pbar.close()
        return x,y

    
class Randomizer(gym.Wrapper):
    def __init__(self, env, prereset_fn):
        super(Randomizer, self).__init__(env)
        self.prereset_fn = prereset_fn
    
    def reset(self):
        self.prereset_fn(self.unwrapped)
        return self.env.reset()
        
    
