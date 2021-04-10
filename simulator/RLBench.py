from simulator.base import BaseSimulator

from rlbench.environment import Environment
from rlbench.action_modes import ArmActionMode, ActionMode
from rlbench.observation_config import ObservationConfig
from rlbench.tasks import ReachTarget
import numpy as np
import matplotlib.pyplot as plt


class RLBench(BaseSimulator):
    def __init__(self):
        
        obs_config = ObservationConfig()
        obs_config.set_all(True)
        
        action_mode = ActionMode(ArmActionMode.ABS_JOINT_VELOCITY)
        self.env = Environment(
            action_mode, obs_config=obs_config, headless=False)
        self.env.launch()
        
        self.task = self.env.get_task(ReachTarget)
        


    def reset(self):
        d, o = self.task.reset()
        return o.front_rgb

    def step(self, action):
        
        action_onehot = np.zeros(8)
        
        if(action%2==0):
            a = int(action/2)
            action_onehot[a] = 1
            
        else:
            a = int((action-1)/2)
            action_onehot[a] = -1
            
        s, r, t = self.task.step(action_onehot)
        
        
        return s.front_rgb, r, t

    @staticmethod
    def n_actions():
        return 16
    
    def shutdown(self):
        print("Shutdown")
        self.env.shutdown()
        

    def __del__(self):
        print("Shutdown")
        self.env.shutdown()
    
