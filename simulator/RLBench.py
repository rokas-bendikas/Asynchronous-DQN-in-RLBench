from simulator.base import BaseSimulator

from rlbench.environment import Environment
from rlbench.action_modes import ArmActionMode, ActionMode
from rlbench.observation_config import ObservationConfig, CameraConfig
from rlbench.tasks import ReachTarget
import numpy as np
from pyquaternion import Quaternion



class RLBench(BaseSimulator):
    def __init__(self,h):
        
        cam = CameraConfig(image_size=(64, 64))
        
        obs_config = ObservationConfig(left_shoulder_camera=cam,right_shoulder_camera=cam,wrist_camera=cam,front_camera=cam)
        obs_config.set_all(True)
        
        action_mode = ActionMode(ArmActionMode.DELTA_EE_POSE_WORLD_FRAME)
        self.env = Environment(
            action_mode, obs_config=obs_config, headless=h)
        self.env.launch()
        
        self.task = self.env.get_task(ReachTarget)
        


    def reset(self):
        d, o = self.task.reset()
        
        state = np.concatenate((o.front_rgb, o.left_shoulder_rgb,o.right_shoulder_rgb,o.wrist_rgb),axis=2)
        
        return state

    def step(self, a):
        
        
        action = np.zeros(8)
        
        qt = Quaternion(1,0,0,0)
        action[3] = qt[0]
        action[4] = qt[1]
        action[5] = qt[2]
        action[6] = qt[3]
        
        
        # For positive values
        if(a%2==0):
            a = int(a/2)
            print(a)
            if ((a==0) or (a==1) or (a== 2)):
                action[a] = 0.01
            else:
                axis = [0,0,0]
                axis[a-3] = 1
                #qt = Quaternion(axis=axis,angle=0.00000000175)
                action[3] = qt[0]
                action[4] = qt[1]
                action[5] = qt[2]
                action[6] = qt[3]
            
        
        # For negative values
        else:
            a = int((a-1)/2)
            print(a)
            if ((a==0) or (a==1) or (a== 2)):
                action[a] = -0.01
            else:
                axis = [0,0,0]
                axis[a-3] = 1
                #qt = Quaternion(axis=axis,angle=0.00000000175)
                action[3] = qt[0]
                action[4] = qt[1]
                action[5] = qt[2]
                action[6] = qt[3]
               
        
        
        """
        action = np.zeros(8)
        
        if(a%2==0):
            a = int(a/2)
            action[a] = 1
            
        else:
            a = int((a-1)/2)
            action[a] = -1
            
        """
        
            
        s, r, t = self.task.step(action)
        
       
        
        state = np.concatenate((s.front_rgb, s.left_shoulder_rgb,s.right_shoulder_rgb,s.wrist_rgb),axis=2)
        
        
        return state, r, t

    @staticmethod
    def n_actions():
        return 14
    
    def shutdown(self):
        print("Shutdown")
        self.env.shutdown()
        

    def __del__(self):
        print("Shutdown")
        self.env.shutdown()
    
