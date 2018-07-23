import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env
from xml.etree import ElementTree as ET
import os

class HopperEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        self.friction = 2.3
        self.torso_size = 0.02
        self.apply_env_modifications()
        #mujoco_env.MujocoEnv.__init__(self, 'hopper.xml', 4)
        utils.EzPickle.__init__(self)

    def apply_env_modifications(self):
        path = os.path.join(os.path.dirname(__file__), "assets", 'hopper.xml')
        xmldoc = ET.parse(path)
        root = xmldoc.getroot()
        for geom in root.iter('geom'):
            if geom.get('name') == 'torso_geom':
                #print('torso size =', self.torso_size)
                geom.set('size', str(self.torso_size))
            if geom.get('name') == 'foot_geom':
                #print('foot friction =', self.friction)
                geom.set('friction', str(self.friction))
        
        tmppath = '/tmp/modified_hopper.xml'
        xmldoc.write(tmppath)
        mujoco_env.MujocoEnv.__init__(self, tmppath, 4)            
        
    def step(self, a):
        posbefore = self.sim.data.qpos[0]
        self.do_simulation(a, self.frame_skip)
        posafter, height, ang = self.sim.data.qpos[0:3]
        alive_bonus = 1.0
        reward = (posafter - posbefore) / self.dt
        reward += alive_bonus
        reward -= 1e-3 * np.square(a).sum()
        s = self.state_vector()
        done = not (np.isfinite(s).all() and (np.abs(s[2:]) < 100).all() and
                    (height > .7) and (abs(ang) < .2))
        ob = self._get_obs()
        return ob, reward, done, {}

    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat[1:],
            np.clip(self.sim.data.qvel.flat, -10, 10)
        ])

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(low=-.005, high=.005, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 2
        self.viewer.cam.distance = self.model.stat.extent * 0.75
        self.viewer.cam.lookat[2] += .8
        self.viewer.cam.elevation = -20
