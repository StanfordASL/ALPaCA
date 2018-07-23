"""
Nonlinear planar quad model with laser sensors implemented by 
James Harrison and Apoorva Sharma
Implements a 6D state space + 14D observation space 
THIS VERSION IS STATE ONLY (NO OBSV) and has floor obstacle only
"""

import logging
import math
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import scipy
from scipy.integrate import odeint
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

class PlanarQuadFloorEnv(gym.Env):
    """This implements the car model used in:
    "Kinodynamic RRT*: Optimal Motion Planning for Systems with Linear Differential Constraints"
    by Dustin Webb and Jur van den Berg
    https://arxiv.org/abs/1205.5088
    """

    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : 30
    }

    def __init__(self):
        self.m = 1.25
        self.Cd_v = 0.25
        self.Cd_phi = 0.02255
        self.Iyy = 0.03
        self.g = 9.81
        self.l = 0.5
        self.Tmax = 1.50*self.m*self.g
        self.Tmin = 0

        self.num_obst = 3
        self.num_sensors = 8

        self.control_cost = 0.01
        self.goal_bonus = 1000
        self.collision_cost = -2*200*self.control_cost*self.Tmax**2

        # What qualifies as a "success" such that we select it when expanding states?
        # This is a normalized value, akin to dividing the reward by the absolute of the 
        # min_cost and then shifting it so that all values are positive between 0 and 1.
        # This does NOT affect PPO, only our selection algorithm after.
        self.R_min = 0.5
        self.R_max = 1.0

        self.quad_rad = self.l

        #bounding box
        self.x_upper = 5.
        self.x_lower = -5.
        self.y_upper = 15.
        self.y_lower = -1.

        #other state bounds
        self.v_limit = 2.5
        self.phi_limit = 5.
        self.omega_limit = np.pi/6.

        #goal region
        # Have no fear, goal_state isn't used anywhere, 
        # it's just for compatibility.
        # x, vx, y, vy, phi, omega
        self.goal_state = np.array([4.5,0,4.5,0,0,0])
        self.goal_w = 0.
        self.goal_vx = 0.
        self.goal_vy = 0.
        self.goal_phi = 0.

        self.xg_lower = 4.
        self.yg_lower = 4.
        self.xg_upper = 5.
        self.yg_upper = 5.
        self.g_vel_limit = 0.25
        self.g_phi_limit = np.pi/6.
        # This isn't actually used for goal pos calculations, 
        # but for backreachability
        self.g_pos_radius = 0.1

        # After defining the goal, create the obstacles.
        # self._generate_obstacles()

        self.dt = 0.1

        self.start_state = np.zeros(6)
        self.start_state[0] = 4.0
        self.start_state[2] = 0.75

        self.min_cost = self.collision_cost - 2*200*self.control_cost*self.Tmax**2 

        high_ob = [self.x_upper,
                  self.v_limit,
                  self.y_upper,
                  self.v_limit,
                  1.,
                  1.,
                  self.omega_limit]

        low_ob = [self.x_lower,
                -self.v_limit,
                self.y_lower,
                -self.v_limit,
                -1.,
                -1.,
                -self.omega_limit]

        # high_ob += [self.x_upper*2]*self.num_sensors
        # low_ob += [self.x_lower*2]*self.num_sensors

        high_state = [self.x_upper,
                      self.v_limit,
                      self.y_upper,
                      self.v_limit,
                      self.phi_limit,
                      self.omega_limit]

        low_state = [self.x_lower,
                    -self.v_limit,
                    self.y_lower,
                    -self.v_limit,
                    -self.phi_limit,
                    -self.omega_limit]

        high_state = np.array(high_state)
        low_state = np.array(low_state)
        high_obsv = np.array(high_ob)
        low_obsv = np.array(low_ob)

        # high_actions = np.array([self.Tmax, self.Tmax])
        # low_actions = np.array([self.Tmin, self.Tmin])
        high_actions = np.array([3., 3.])
        low_actions = np.array([-3., -3.])

        self.action_space = spaces.Box(low=low_actions,high=high_actions)
        self.state_space = spaces.Box(low=low_state, high=high_state)
        self.observation_space = spaces.Box(low=low_obsv, high=high_obsv)

        self.seed(2015)
        self.viewer = None
    

    def set_hovering_goal(self, hover_at_end):
        print('Set hover_end to', hover_at_end, flush=True)
        self.hover_end = hover_at_end


    def map_action(self, action):
        return [ self.Tmin + (0.5 + a/6.0)*(self.Tmax - self.Tmin) for a in action ]

    def set_disturbance(self, disturbance_str):
        #TODO
        pass

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def x_dot(self,z,u):
        x,vx,y,vy,phi,omega = z
        T1,T2 = u
        x_d = [
        vx,
        (-1/self.m)*self.Cd_v*vx - (T1/self.m)*np.sin(phi) - (T2/self.m)*np.sin(phi),
        vy,
        (-1/self.m)*(self.m*self.g + self.Cd_v*vy) + (T1/self.m)*np.cos(phi) + (T2/self.m)*np.cos(phi),
        omega,
        (-1/self.Iyy)*self.Cd_phi*omega - (self.l/self.Iyy)*T1 + (self.l/self.Iyy)*T2
        ]

        return x_d


    # def _generate_obstacles(self):
    #     #currently, making the obstacle placement deterministic so that we guarantee feasibility
    #     # Temporarily removing obstacles.
    #     self.obst_R = np.array([0.5,1.0,0.5])
    #     self.obst_X = np.array([4.0,1.0,1.0])
    #     self.obst_Y = np.array([2.5,1.0,4.0])

    def _in_goal(self, state):
        xq = state[0]
        yq = state[2]

        if (xq < self.xg_upper) and (xq > self.xg_lower) and (yq < self.yg_upper) and (yq > self.yg_lower):
            vx = state[1]
            vy = state[3]
            phi = state[4]
            omega = state[5]
            if self.hover_end:
                if (abs(vx) < self.g_vel_limit) and (abs(vy) < self.g_vel_limit) and (abs(omega) < self.g_vel_limit) and (abs(phi) < self.g_phi_limit):
                    return True
                else:
                    return False

            else:
                return True
        
        else:
            return False
    
    # input: list of obstacle x,y,r
    #        state space bounds (walls) xlow, xhigh, ylow, yhigh  
    #        ray x,y origin, angle th w.r.t global frame x axis
    # output: distance to nearest obstacle
    def ray_dist(self,x,y,th):
        # first compute distances to all obstacles (vectorized)
        th_obs = np.arctan2( self.obst_Y - y, self.obst_X - x) 
        dth = np.mod(th - th_obs + np.pi, 2*np.pi) - np.pi

        R = np.sqrt( (self.obst_X - x)**2 + (self.obst_Y - y)**2 )

        sinalpha = R*np.sin(dth)/self.obst_R
        sinalpha[abs(sinalpha)>1] = np.nan
        alpha = np.pi - np.arcsin(sinalpha)
        beta = np.pi - dth - alpha

        d = np.sqrt(R**2 + self.obst_R**2 - 2*R*self.obst_R*np.cos(beta))
        d[dth>np.pi/2] = np.inf

        beta[np.isnan(beta)] = np.inf
        d[beta>np.pi/2] = np.inf

        # append distances to all walls
        d_xhigh = np.inf
        d_xlow = np.inf
        d_yhigh = np.inf
        d_ylow = np.inf

        if abs(np.cos(th)) > 1e-5:
            delx_high = self.x_upper - x
            d_xhigh = delx_high/np.cos(th)

            delx_low = self.x_lower - x
            d_xlow = delx_low/np.cos(th)

        if np.abs(np.sin(th)) > 1e-5:
            dely_high = self.y_upper - y
            d_yhigh = dely_high/np.sin(th)

            dely_low = self.y_lower - y
            d_ylow = dely_low/np.sin(th)

        d = np.concatenate([d, [d_xhigh, d_xlow, d_yhigh, d_ylow]])
        d[d<0] = np.inf

        return np.min(d)

    def get_ray_angles(self):
        th = self.state[4]
        del_th = 2*np.pi/self.num_sensors
        # Must force it to the first self.num_sensors because of numerical
        # issues (it happens!)
        return np.arange(th, th+2*np.pi, del_th)[:self.num_sensors]

    def sensor_measurements(self):
        x = self.state[0]
        y = self.state[2]
        ray_angles = self.get_ray_angles()
        ray_measurements = [self.ray_dist(x,y,th_r) for th_r in ray_angles]
        return np.array(ray_measurements)

    def plot_quad_in_map(self):
        x = self.state[0]
        y = self.state[2]
        th = self.state[4]
        r_quad = self.quad_rad
        
        # ray_angles = self.get_ray_angles()
        # ray_measurements = [self.ray_dist(x,y,th_r) for th_r in ray_angles]

        # x_points = x + ray_measurements*np.cos(ray_angles)
        # y_points = y + ray_measurements*np.sin(ray_angles)

        # for xi, yi in zip(x_points, y_points):
        #     plt.plot([x,xi], [y, yi], color='r', linestyle=':', alpha=0.5)
        # plt.plot(x_points, y_points, marker='+', color='r', linestyle='none')
        ax = plt.gca()
        # for xo,yo,ro in zip(self.obst_X, self.obst_Y, self.obst_R):
        #     c = plt.Circle((xo,yo),ro, color='black', alpha=1.0)
        #     ax.add_artist(c)

       
        # r = plt.Rectangle((self.xg_lower, self.yg_lower), self.xg_upper-self.xg_lower, self.yg_upper - self.yg_lower, color='g', alpha=0.3, hatch='/')
        # ax.add_artist(r)

        #floor plotting
        plt.fill_between([-10,10],[-2,-2],color='grey')

        plt.plot([x - r_quad*np.cos(th), x + r_quad*np.cos(th)], [y - r_quad*np.sin(th), y + r_quad*np.sin(th)], marker='o', linewidth=2, color='b', markersize=5)

        plt.xlim([self.x_lower, self.x_upper])
        plt.ylim([self.y_lower, self.y_upper])

    def _in_obst(self, state):
        #currently checking if y is negative

        if state[2] < 0.0:
            return True

        return False

    def _get_obs(self, state):
        return state #currently not returning sin/cos
        # x,vx,y,vy,phi,omega = state
        # return np.array([x, vx, y, vy, np.cos(phi), np.sin(phi), omega])

    def _gen_state_rew(self,state):
        x,vx,y,vy,phi,omega = state

        linear_thresh = 5.0
        cost_slope = 10.0

        r = 0
        #linear, for ground
        if y < linear_thresh:
            r += (linear_thresh - y)*cost_slope

        #quadratic state cost
        x_cost = 0.0
        y_cost = 0.0
        vx_cost = 1.0
        vy_cost = 1.0
        th_cost = 10.0
        om_cost = 5.0

        r += x_cost*x**2
        r += y_cost*y**2
        r += vx_cost*vx**2
        r += vy_cost*vy**2
        r += th_cost*phi**2
        r += om_cost*omega**2

        return -r


    def step(self, action):
        #map action
        action = self.map_action(action)

        if sum(np.isnan(action)) > 0:
            raise ValueError("Passed in nan to step! Action: " + str(action));
        
        #clip actions
        action = np.clip(action,self.Tmin,self.Tmax)

        old_state = np.array(self.state)

        t = np.arange(0, self.dt, self.dt*0.01)

        integrand = lambda x,t: self.x_dot(x, action)

        x_tp1 = odeint(integrand, old_state, t)
        self.state = x_tp1[-1,:] 

        # Be close to the goal and have the desired final velocity.
        reward = - self.control_cost*(action[0]**2 + action[1]**2)
        done = False

        #TODO add stochasticity in state transition

        #TODO generate state rewards
        reward += self._gen_state_rew(self.state)

        # if self._in_goal(self.state):
        #     reward += self.goal_bonus
        #     done = True
        # not currently checking along the trajectory for collision violation
        if self._in_obst(self.state):
            reward += self.collision_cost
            done = True

        return self._get_obs(self.state), reward, done, {}

    def reset(self):
        # self._generate_obstacles()
        # self._generate_goal() 

        #arbitrary choices of init state distribution right now
        init_height_offset = 10.0
        x = np.random.randn()
        y = np.random.randn() + init_height_offset
        vx = np.random.randn()*0.2
        vy = np.random.randn()*0.02
        th = np.random.rand()*2*np.pi
        omega = np.random.randn()*0.2
        self.state = np.array([x, vx, y, vy, th, omega])

        # self.state = self.start_state.copy()
        
        return self._get_obs(self.state)

    def render(self, mode='human', close=False):
        pass