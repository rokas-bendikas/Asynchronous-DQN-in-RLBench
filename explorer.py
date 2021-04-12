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

from utils import prepare_batch, as_tensor, data_to_buffer, init_states,reset_states,action_onehot
import torch.multiprocessing as mp

import torch as t
from torch.autograd import Variable

from device import Device

import logging
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
            
def explore(idx,SIMULATOR,model,buffer,args):
        
        try:
            
            writer = SummaryWriter('runs/{}/explorer:{:02}'.format(datetime.now().strftime("%d|%m_%H|%M"), idx))
            logging.basicConfig(filename='logs/explorer:{:02}.log'.format(idx),
                                filemode='w',
                                format='%(message)s',
                                level=logging.DEBUG)
           
            
            simulator = SIMULATOR(args.headless)
            
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
                        action = np.random.RandomState().randint(simulator.n_actions())
                        
                    else:
                            
                        action = model(as_tensor(state)).argmax().item()
                        
                                                              
                          
                    next_state, reward, terminal = simulator.step(action)
                    
                    
                    reward *= 1000
                    if(buffer.full()):
                        buffer.get()
                        
                    data = data_to_buffer(state, action, reward, next_state, terminal)

                                 
                    buffer.put(data)
                    
        
                    episode_reward += reward
                    state = next_state
                    
                    
                    if terminal:
                        #print("\nDummy number: {} reached the goal!\n".format(idx))
                        break
                    '''
                    if (e>args.episode_length):
                        break
                    '''
                    
                logging.debug('Episode reward: {:.2f}, epsilon: {:.2f}'.format(episode_reward,max(args.eps ** itr, args.min_eps)))
                writer.add_scalar('episode_reward', episode_reward, itr)
                writer.close()
                    
        
        except KeyboardInterrupt:
            print('exiting explorer:{:02}'.format(idx))
            simulator.shutdown()
            
            
    
            
        
            
           
