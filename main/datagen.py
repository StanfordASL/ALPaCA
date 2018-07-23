import gym
import numpy as np
import time
import os
from tqdm import tqdm
import glob
import h5py
import sys
import random

def pi_zero(state):
    return np.array([0.0])

def pi_random(state):
    return np.random.randn((1))

def sin_gen(freq, phase,amp,noise=0.1):

    x = np.random.uniform(-5, 5)
    eps = np.random.randn()*np.sqrt(noise)
    return x, amp*np.sin(freq*x + phase) + eps

def step_gen(pt,noise=0.0):
    x = np.random.uniform(-5, 5)
    eps = np.random.randn()*noise
    if x<pt:
        return x,0.0+eps
    return x,1.0+eps

def multistep_gen(pt_list,noise=0.1):
    #pt list must be sorted 
    assert(all(pt_list == np.sort(pt_list)))
    
    #currently only doing three pts
    
    x = np.random.uniform(-5, 5)
    eps = np.random.randn()*noise
    for i in range(len(pt_list)):
        if x<pt_list[0]:
            return x, ((i)%2)*2 + eps -1.0
        
        if i==(len(pt_list)-1) and x>pt_list[-1]:
            return x, ((i+1)%2)*2 + eps -1.0
        
        if x>pt_list[i] and x<pt_list[i+1]:
            return x, ((i+1)%2)*2 + eps -1.0

def build_list(min_,max_,N):
    assert(min_ < max_)
    return [np.random.uniform(min_,max_) for _ in range(N)] # can do this in a batch

class DataGenerator:
    """
    Generates data for use in ALPaCA body. This should be written with better code reuse when James has more time.
    """
    
    def __init__(self,config,env,env_name):
        self.env = env
        self.env_name = env_name
        self.config = config
    
    
    
    def sample_trajectories(self,policy,num_samples,num_tasks,return_lists=False,sigma_eps = None, lists=None):
        #take as input policy pi, function that returns action for a state
        
        x_dim = self.config['x_dim']
        y_dim = self.config['y_dim']
        y_matrix = np.zeros((num_tasks,num_samples,y_dim))
        x_matrix = np.zeros((num_tasks,num_samples,x_dim))
            
        
        if self.env_name == 'Sinusoid':
            if lists is None:
                phase_min = self.config['sinusoid_phase_min']
                phase_max = self.config['sinusoid_phase_max']
                phase_list = build_list(phase_min,phase_max,num_tasks)

                freq_min = self.config['sinusoid_freq_min']
                freq_max = self.config['sinusoid_freq_max']
                freq_list = build_list(freq_min,freq_max,num_tasks)

                # add amplitude variation
                amp_min = self.config['sinusoid_amp_min']
                amp_max = self.config['sinusoid_amp_max']
                amp_list = build_list(amp_min,amp_max,num_tasks)
            
            else:
                phase_list,freq_list,amp_list = lists
            
            for i in range(num_tasks):
                
                p = phase_list[i]
                f = freq_list[i]
                a = amp_list[i] 
                
                for j in range(num_samples):
                    
                    if sigma_eps is None:
                        noise = self.config['sinusoid_noise_var']
                    else:
                        noise = sigma_eps
                        
                    x,y = sin_gen(f,p,a,noise=noise)

                    #store it
                    y_matrix[i,j,:] = y
                    x_matrix[i,j,:] = x

            if return_lists: 
                return y_matrix, x_matrix, phase_list,freq_list,amp_list
            return y_matrix, x_matrix
        
        if self.env_name == 'Multistep':
            if lists is None:
                step_min = self.config['step_min']
                step_max = self.config['step_max']
                num_steps = self.config['num_steps']

                step_mat = np.zeros((num_tasks,num_steps))
            else:
                step_mat = lists
                
            y_matrix = np.zeros((num_tasks,num_samples,1))
            x_matrix = np.zeros((num_tasks,num_samples,1))
            
            for i in range(num_tasks):
                if lists is None:
                    step_list = sorted(build_list(step_min,step_max,num_steps))
                    step_mat[i,:] = step_list
                    
                else:
                    step_list = step_mat[i,:]
                
                for j in range(num_samples):
                    if sigma_eps is None:
                        noise = self.config['sigma_eps']
                    else:
                        noise = sigma_eps
                    
                    x,y = multistep_gen(step_list,noise=noise)

                    #store it
                    y_matrix[i,j,:] = y
                    x_matrix[i,j,:] = x

            if return_lists: 
                return y_matrix, x_matrix, step_mat
            return y_matrix, x_matrix
        
        if self.env_name == 'Pendulum-v0':
            
            mass_min = self.config['pendulum_mass_min']
            mass_max = self.config['pendulum_mass_max']
            mass_list = build_list(mass_min,mass_max,num_tasks)
            
            len_min = self.config['pendulum_len_min']
            len_max = self.config['pendulum_len_max']
            len_list = build_list(len_min,len_max,num_tasks)            
            
            for i in range(num_tasks):
                m = mass_list[i]
                l = len_list[i]
                
                self.env.unwrapped.m = m
                self.env.unwrapped.l = l
                self.env.reset()
                
                for j in range(num_samples):
                    s = self.env.unwrapped.state
                    a = policy(s)
                    self.env.step(a)
                    sp = self.env.unwrapped.state
                    
                    #store it
                    y_matrix[i,j,:] = sp - s
                    x_matrix[i,j,:] = np.concatenate((s, a))
                    
            if return_lists: 
                return y_matrix, x_matrix, mass_list   
            return y_matrix, x_matrix
        
        
        if self.env_name == 'Hopper-v2':
            
            torso_min = self.config['torso_min']
            torso_max = self.config['torso_max']
            torso_list = build_list(torso_min,torso_max,num_tasks)
            
            friction_min = self.config['friction_min']
            friction_max = self.config['friction_max']
            friction_list = build_list(friction_min,friction_max,num_tasks)
            
            for i in tqdm(range(num_tasks)):
                f = friction_list[i]
                ts = torso_list[i]
                
                self.env.unwrapped.friction = f
                self.env.unwrapped.torso_size = ts
                self.env.unwrapped.apply_env_modifications()
                ob = self.env.reset()
                
                for j in range(num_samples):
                    s = self.env.unwrapped.state_vector()
                    a = policy(ob)
                    ob,_,done,_ = self.env.step(a)
                    sp = self.env.unwrapped.state_vector()
                    
                    if done:
                        self.env.reset()
                    
                    #store it
                    y_matrix[i,j,:] = sp - s
                    x_matrix[i,j,:] = np.concatenate((s, a))
                    
            if return_lists: 
                return y_matrix, x_matrix, mass_list   
            return y_matrix, x_matrix
                        
def get_v_dir_from_filename(filename):
    tokens= filename.split("_")
    v = float( tokens[4].split("=")[1] )
    dir = float( tokens[-1].split("=")[1][:-3])
    return v,dir