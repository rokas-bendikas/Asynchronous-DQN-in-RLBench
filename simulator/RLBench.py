from simulator.base import BaseSimulator

from rlbench.environment import Environment
from rlbench.action_modes import ArmActionMode, ActionMode
from rlbench.observation_config import ObservationConfig, CameraConfig
from rlbench.tasks import ReachTarget
from rlbench.task_environment import InvalidActionError
from pyrep.errors import ConfigurationPathError

import numpy as np

class RLBench(BaseSimulator):
    def __init__(self,h):
        
        #64x64 camera outputs
        cam = CameraConfig(image_size=(64, 64))
        obs_config = ObservationConfig(left_shoulder_camera=cam,right_shoulder_camera=cam,wrist_camera=cam,front_camera=cam)
        obs_config.set_all(True)
        
        # delta EE control with motion planning
        action_mode = ActionMode(ArmActionMode.DELTA_EE_POSE_PLAN_WORLD_FRAME)
        
        #Inits
        self.env = Environment(action_mode, obs_config=obs_config, headless=h)
        self.env.launch()
        self.task = self.env.get_task(ReachTarget)
        


    def reset(self):
        
        d, o = self.task.reset()
        
        return o

    def step(self, a, prev_state):
        
        
        # delta orientation
        d_quat = np.array([0, 0, 0, 1])
        
        # gripper state
        gripper_open = 1.0
        
        # delta position
        d_pos = np.zeros(3)
        
        # For positive magnitude
        if(a%2==0):
            a = int(a/2)
            d_pos[a] = 0.03
            
        # For negative magnitude
        else:
            a = int((a-1)/2)
            d_pos[a] = -0.03
        
        # Forming action as expected by the environment
        action = np.concatenate([d_pos, d_quat, [gripper_open]])
    
        try:
            s, r, t = self.task.step(action)
            r*=1000
        
        # Handling failure in planning
        except ConfigurationPathError:
            s = prev_state
            r = -0.1
            t = False
        
        # Handling wrong action for inverse Jacobian
        except InvalidActionError:
            s = prev_state
            r = -0.01
            t = False
            
        
        
        # Get bounding box centroids
        x, y, z = self.task._scene._workspace.get_position()
        
        # Set bounding box limits
        minx = x - 0.25
        maxx = x + 0.25
        miny = y - 0.35
        maxy = y + 0.35
        minz = z
        maxz = z + 0.5  
        
        bounding_box = [minx,maxx,miny,maxy,minz,maxz]
        
        # Get gripper position
        gripper_pose = s.gripper_pose
        
        
        # Reward for being in the bounding box
        if (self.bb_check(bounding_box,gripper_pose)):
            r += 0.1
        

        return s, r, t
    
    
    # Check if gripper in the bounding box
    def bb_check(self,bounding_box,gripper_pose):
        
        out = True
        
        if(gripper_pose[0] < bounding_box[0] or gripper_pose[0] > bounding_box[1]):
            out = False
            
        if(gripper_pose[1] < bounding_box[2] or gripper_pose[1] > bounding_box[3]):
            out = False
            
        if(gripper_pose[2] < bounding_box[4] or gripper_pose[2] > bounding_box[5]):
            out = False
        
        
        return out
    
    
    
    @staticmethod
    def n_actions():
        return 6
    
    def shutdown(self):
        print("Shutdown")
        self.env.shutdown()
        

    def __del__(self):
        print("Shutdown")
        self.env.shutdown()
    
