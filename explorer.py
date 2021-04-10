#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  8 17:03:07 2021

@author: rokas
"""


from collections import deque
from itertools import count

import numpy as np
from tqdm import tqdm

from utils import prepare_batch, as_tensor, data_to_buffer
import torch.multiprocessing as mp

import torch as t

from device import Device
            
def explore(idx,SIMULATOR,model,buffer,args):
        
        try:
           
            model.eval()
            
            simulator = SIMULATOR()
            
            # allocate a device
            n_gpu = t.cuda.device_count()
            if n_gpu > 0:
                #Device.set_device(idx % n_gpu)
                Device.set_device('cpu')
            
            
            for itr in tqdm(count(), position=idx, desc='explorer:{:02}'.format(idx)):
        
                
                state = simulator.reset()
                episode_reward = 0
                for e in count():
                    
                    
                        
                        
                    if np.random.RandomState().rand() < max(args.eps ** itr, args.min_eps):
                        action = np.random.RandomState().randint(simulator.n_actions()-1)
                    else:
                            
                        action = model(as_tensor(state)).argmax().item()
                                       
                          
                    next_state, reward, terminal = simulator.step(action)
                    
                    if(buffer.full()):
                        buffer.get()
                        
                    data = data_to_buffer(state, action, reward, next_state, terminal)
                    
                                    
                    buffer.put(data)
                    
                    episode_reward += reward
                    state = next_state
                    
                    
                    if terminal or (e>args.iter_length):
                        break
                    
                    
        
        except KeyboardInterrupt:
            print('exiting explorer:{:02}'.format(idx))
            simulator.shutdown()
            
            
    
            
        
            
           
