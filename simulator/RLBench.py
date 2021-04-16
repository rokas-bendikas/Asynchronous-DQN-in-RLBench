from simulator.base import BaseSimulator

from rlbench.environment import Environment
from rlbench.action_modes import ArmActionMode, ActionMode
from rlbench.observation_config import ObservationConfig, CameraConfig
from rlbench.tasks import ReachTarget
from rlbench.task_environment import InvalidActionError

import numpy as np


def normalize_action(action: np.ndarray,task):
       
    
        [ax, ay, az] = action[:3]
        x, y, z, qx, qy, qz, qw = task._robot.arm.get_tip().get_pose()
        

        # position
        d_pos = np.array([ax, ay, az])
        #d_pos /= (np.linalg.norm(d_pos) * 100.0)

        # orientation
        d_quat = np.array([0, 0, 0, 1.0])

        # gripper_open = action[-1]
        gripper_open = 1.0

    
        action = np.concatenate([d_pos, d_quat, [gripper_open]])

            
        return action



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

    def step(self, a, prev_state):
        
        
        # orientation
        d_quat = np.array([0, 0, 0, 1])
        
        # gripper_open = action[-1]
        gripper_open = 1.0
        
        d_pos = np.zeros(3)
        
        # For positive values
        if(a%2==0):
            a = int(a/2)
            d_pos[a] = 0.0225
            
        # For negative values
        else:
            a = int((a-1)/2)
            d_pos[a] = -0.0225
            
        action = np.concatenate([d_pos, d_quat, [gripper_open]])
    
        try:
            
            s, r, t = self.task.step(action)
            state = np.concatenate((s.front_rgb, s.left_shoulder_rgb,s.right_shoulder_rgb,s.wrist_rgb),axis=2)
            
        except InvalidActionError:
            state = prev_state
            r = -0.001
            t = False
        

        
        return state, r, t

    @staticmethod
    def n_actions():
        return 6
    
    def shutdown(self):
        print("Shutdown")
        self.env.shutdown()
        

    def __del__(self):
        print("Shutdown")
        self.env.shutdown()
    
