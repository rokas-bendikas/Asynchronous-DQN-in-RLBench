#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 16 12:31:22 2021

@author: rokas
"""


def step(self, a):
        
        # a is an integer [0,5]
        
        # orientation
        d_quat = np.array([0, 0, 0, 1.0])
        
        # gripper_open = action[-1]
        gripper_open = 1.0
        
        d_pos = np.zeros(3)
        
        # For positive values
        if(a%2==0):
            a = int(a/2)
            d_pos[a] = 0.001
            
        # For negative values
        else:
            a = int((a-1)/2)
            d_pos[a] = -0.001
            
        # For action
        action = np.concatenate([d_pos, d_quat, [gripper_open]])
        
         
        s, r, t = self.task.step(action)
        
        state = np.concatenate((s.front_rgb, s.left_shoulder_rgb,s.right_shoulder_rgb,s.wrist_rgb),axis=2)
        
        
        return state, r, t