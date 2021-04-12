from simulator.base import BaseSimulator

from rlbench.environment import Environment
from rlbench.action_modes import ArmActionMode, ActionMode
from rlbench.observation_config import ObservationConfig
from rlbench.tasks import ReachTarget
import numpy as np
import matplotlib.pyplot as plt



class RLBench(BaseSimulator):
    def __init__(self,h):
        
        obs_config = ObservationConfig()
        obs_config.set_all(True)
        
        action_mode = ActionMode(ArmActionMode.ABS_JOINT_VELOCITY)
        self.env = Environment(
            action_mode, obs_config=obs_config, headless=h)
        self.env.launch()
        
        self.task = self.env.get_task(ReachTarget)
        


    def reset(self):
        d, o = self.task.reset()
        
        state = np.concatenate((o.front_rgb, o.left_shoulder_rgb,o.right_shoulder_rgb,o.wrist_rgb),axis=2)
        
        return state

    def step(self, action):
        
        action_onehot = np.zeros(8)
        
        if(action%2==0):
            a = int(action/2)
            action_onehot[a] = 1
            
        else:
            a = int((action-1)/2)
            action_onehot[a] = -1
            
        s, r, t = self.task.step(action_onehot)
        
       
        
        state = np.concatenate((s.front_rgb, s.left_shoulder_rgb,s.right_shoulder_rgb,s.wrist_rgb),axis=2)
        
        
        return state, r, t

    @staticmethod
    def n_actions():
        return 12
    
    def shutdown(self):
        print("Shutdown")
        self.env.shutdown()
        

    def __del__(self):
        print("Shutdown")
        self.env.shutdown()
    
