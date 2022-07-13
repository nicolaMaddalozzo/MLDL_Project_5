"""Implementation of the Hopper environment supporting
domain randomization optimization."""
import csv
import pdb
from copy import deepcopy

import numpy as np
import gym
from gym import utils
from .mujoco_env import MujocoEnv
from scipy.stats import truncnorm

class CustomHopper(MujocoEnv, utils.EzPickle):
    def __init__(self, domain=None):
        MujocoEnv.__init__(self, 4)
        utils.EzPickle.__init__(self)

        self.original_masses = np.copy(self.sim.model.body_mass[1:])    # Default link masses

        if domain == 'source':  # Source environment has an imprecise torso mass (1kg shift)
            self.sim.model.body_mass[1] -= 1.0


    def set_random_parameters(self):
        """Set random masses
        TODO
        """
        #print("set")                 #3 ho tolto un "*"
        masses1 = self.set_parameters(self.sample_parameters())
        return masses1
        
        
    def sample_parameters(*self):
        """Sample masses according to a domain randomization distribution
        TODO
        
    
        
        lambda = 15
        +----------------+-----------------+--------------+----------------+
        | Parameter name | Mean calculated | Target value | Std calculated |
        +----------------+-----------------+--------------+----------------+
        |   torsomass    |     3.534261    |   3.534292   |    0.000133    |
        |   thighmass    |     3.926943    |   3.926991   |    0.000097    |           9.7*10^{-5} 
        |    legmass     |     2.714138    |   2.714336   |    0.000138    |           1.38*10^{-4}
        |    footmass    |     5.089602    |   5.089380   |    0.000077    |		  7.7*10^{-5}
        +----------------+-----------------+--------------+----------------+
        
        Da un source? Perché no
        +----------------+-----------------+--------------+----------------+
        | Parameter name | Mean calculated | Target value | Std calculated |
        +----------------+-----------------+--------------+----------------+
        |   torsomass    |     2.534328    |   2.534292   |    0.000079    |
        |   thighmass    |     3.926794    |   3.926991   |    0.000351    |       3.51*10^{-4}
        |    legmass     |     2.714291    |   2.714336   |    0.000044    |       4.4*10^{-5}
        |    footmass    |     5.089435    |   5.089380   |    0.000208    |       2.08*10^{-4}
        +----------------+-----------------+--------------+----------------+


        """
        #print("sample")
        #1 new
        mean = np.array([3.534292,3.926943,2.714138,5.089602])
        cov = np.zeros((4, 4), float)
        np.fill_diagonal(cov, [0,0.000097,0.000138,0.000077])
        new_masses = np.random.multivariate_normal(mean, cov)
        #  = np.random.uniform([2.53429174,10,10,10],[2.53429174,20,20,20]) #[2.92,4.92]	[1.7,3.7]	[4,6]
                                                                                        #[2,6]	[0.5,5.5]	[2,8]
										     #[0.5,7.5]	[2,9]	[1,9]
										     #[2,4]	4,5]	[0,10]
										     #[10,20]	[10,20]	[10,20]




        #print(new_masses)
               #2 new  
        return new_masses

    def get_parameters(self):
        """Get value of mass for each link"""
        #print("get")
        self.set_random_parameters()
        
        masses = np.array( self.sim.model.body_mass[1:] )
        #masses = np.array([1,1,1,1,1,1,1,1,1,1,1,1,1,1])
        #masses = np.random.uniform([1,1,1,1],[3,3,3,3])
        #masses = self.set_random_parameters(self)
        
        print(masses)
        return masses

    def set_parameters(self, task):
        """Set each hopper link's mass to a new value"""
        
        self.sim.model.body_mass[1:] = task

    def step(self, a):
        """Step the simulation to the next timestep

        Parameters
        ----------
        a : ndarray,
            action to be taken at the current timestep
        """
        posbefore = self.sim.data.qpos[0]
        self.do_simulation(a, self.frame_skip)
        posafter, height, ang = self.sim.data.qpos[0:3]
        alive_bonus = 1.0
        reward = (posafter - posbefore) / self.dt
        reward += alive_bonus
        reward -= 1e-3 * np.square(a).sum()
        s = self.state_vector()
        done = not (np.isfinite(s).all() and (np.abs(s[2:]) < 100).all() and (height > .7) and (abs(ang) < .2))
        ob = self._get_obs()

        return ob, reward, done, {}

    def _get_obs(self):
        """Get current state"""
        return np.concatenate([
            self.sim.data.qpos.flat[1:],
            self.sim.data.qvel.flat
        ])

    def reset_model(self):
        """Reset the environment to a random initial state"""
        qpos = self.init_qpos + self.np_random.uniform(low=-.005, high=.005, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 2
        self.viewer.cam.distance = self.model.stat.extent * 0.75
        self.viewer.cam.lookat[2] = 1.15
        self.viewer.cam.elevation = -20


    def set_mujoco_state(self, state):
        """Set the simulator to a specific state

        Parameters:
        ----------
        state: ndarray,
               desired state
        """
        mjstate = deepcopy(self.get_mujoco_state())

        mjstate.qpos[0] = 0.
        mjstate.qpos[1:] = state[:5]
        mjstate.qvel[:] = state[5:]

        self.set_sim_state(mjstate)


    def set_sim_state(self, mjstate):
        """Set internal mujoco state"""
        return self.sim.set_state(mjstate)

    def get_mujoco_state(self):
        """Returns current mjstate"""
        return self.sim.get_state()



"""
    Registered environments
"""
gym.envs.register(
        id="CustomHopper-v0",
        entry_point="%s:CustomHopper" % __name__,
        max_episode_steps=500,
)

gym.envs.register(
        id="CustomHopper-source-v0",
        entry_point="%s:CustomHopper" % __name__,
        max_episode_steps=500,
        kwargs={"domain": "source"}
)

gym.envs.register(
        id="CustomHopper-target-v0",
        entry_point="%s:CustomHopper" % __name__,
        max_episode_steps=500,
        kwargs={"domain": "target"}
)


