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
import numpy as np
from device import Device
from utils import as_tensor, data_to_queue
import torch as t
import math
            


def explore(idx,SIMULATOR,model,queue,args,lock):
        
        try:
            
            writer = SummaryWriter('runs/e{}'.format(idx))
            
            logging.basicConfig(filename='logs/explorer:{:02}.log'.format(idx),
                                filemode='w',
                                format='%(message)s',
                                level=logging.DEBUG)
            
            simulator = SIMULATOR(args.headless)
            
            # allocate a device
            Device.set_device('cpu')
            
            total_reward = 0
                
            for itr in tqdm(count(), position=idx, desc='explorer:{:02}'.format(idx)):
        
                
                state = simulator.reset()
                state_processed = np.concatenate((state.front_rgb,state.wrist_rgb),axis=2)
                
                episode_reward = 0
                
                
                for e in count():
                    
                    
                    # During warmup epsilon is one
                    if (itr < args.warmup):
                        eps = 1
                    
                    # Calculating epsilon
                    else:
                        
                        if (idx==0):
                            val = max(args.eps ** (itr+ args.advance_iteration - args.warmup), args.min_eps)
                            eps = math.ceil(val / 0.1) * 0.1
                            
                            
                        else:
                            eps = max(args.eps ** (itr+ args.advance_iteration - args.warmup), args.min_eps)
                    
                    
                    # Epsilon-greedy policy
                    if np.random.RandomState().rand() < eps:
                        action = np.random.RandomState().randint(simulator.n_actions())
                        
                    else:
                        
                        with t.no_grad():
                            action = model(as_tensor(state_processed)).argmax().item()
                        
                                                              
                    # Agent step   
                    next_state, reward, terminal = simulator.step(action,state)
                    
                    # Concainating diffrent cameras
                    next_state_processed = np.concatenate((next_state.front_rgb,next_state.wrist_rgb),axis=2)
                    state_processed = np.concatenate((state.front_rgb,state.wrist_rgb),axis=2)
    
                    # Processing for the queue format
                    data = data_to_queue(state_processed, action, reward, next_state_processed, terminal)
                    
                    # Pushing to the queue
                    lock.acquire()
                    queue.put(data)
                    lock.release()
        
                    # Updating running metrics
                    episode_reward += reward
                    total_reward += reward
                    state = next_state
                    
                    # Early termination conditions
                    if (terminal or (e>args.episode_length)):
                        break
                    
                # Logging
                logging.debug('Episode reward: {:.2f}, Epsilon: {:.2f}, Queue length: {}'.format(episode_reward,eps,queue.qsize()))
                
                writer.add_scalar('Episode reward', episode_reward, itr)
                writer.add_scalar('Total reward',total_reward,itr)
                writer.add_scalar('Epsilon value',eps,itr)
                writer.close()
                    
        
        except KeyboardInterrupt:
            print('exiting explorer:{:02}'.format(idx))
            simulator.shutdown()
            
            
    
            
        
            
           
