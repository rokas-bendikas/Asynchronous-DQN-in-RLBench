#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  8 17:03:07 2021

@author: rokas
"""


from itertools import count
from tqdm import tqdm

import logging
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

import numpy as np

from device import Device

from utils import as_tensor, data_to_queue

import math
            


def explore(idx,SIMULATOR,model,queue,args):
        
        try:
            
            writer = SummaryWriter('runs/{}/explorer:{:02}'.format(datetime.now().strftime("%d|%m_%H|%M"), idx))
            logging.basicConfig(filename='logs/explorer:{:02}.log'.format(idx),
                                filemode='w',
                                format='%(message)s',
                                level=logging.DEBUG)
           
            
            simulator = SIMULATOR(args.headless)
            
            # allocate a device
            Device.set_device('cpu')
                
               
            
            for itr in tqdm(count(), position=idx, desc='explorer:{:02}'.format(idx)):
        
                
                state = simulator.reset()
                episode_reward = 0
                
                
                for e in count():
                    
                    
                    if (idx==0):
                        val = max(args.eps ** itr, args.min_eps)
                        eps = math.ceil(val / 0.1) * 0.1
                    else:
                        eps = max(args.eps ** itr, args.min_eps)
                    
                    
                    
                    if np.random.RandomState().rand() < eps:
                        action = np.random.RandomState().randint(simulator.n_actions())
                        
                    else:
                            
                        action = model(as_tensor(state)).argmax().item()
                        
                                                              
                        
                    next_state, reward, terminal = simulator.step(action,state)
                        
                    reward *= 200
                        
                 
                    data = data_to_queue(state, action, reward, next_state, terminal)

                    queue.put(data)
        
                    episode_reward += reward
                    state = next_state
                    
                    
                    if (terminal or (e>args.episode_length)):
                        
                        break
                    
                    
                logging.debug('Episode reward: {:.2f}, Epsilon: {:.2f}, Queue length: {}'.format(episode_reward,eps,queue.qsize()))
                writer.add_scalar('episode_reward', episode_reward, itr)
                writer.close()
                    
        
        except KeyboardInterrupt:
            print('exiting explorer:{:02}'.format(idx))
            simulator.shutdown()
            
            
    
            
        
            
           
