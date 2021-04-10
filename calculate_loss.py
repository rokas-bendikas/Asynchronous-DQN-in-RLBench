import torch as t
import torch.nn.functional as f
import numpy as np
import time


def calculate_loss(q_network, target_network, batch, hyperparameters,device):
    
    state, action, reward, next_state, terminal = batch
    
    state = state.to(device)
    reward = reward.to(device)
    next_state = next_state.to(device)
    terminal = terminal.to(device)
    action = action.type(t.int64)
    
    
    with t.no_grad():
        target = reward + terminal * hyperparameters.gamma * target_network(next_state).max()

      
    predicted = q_network(state)
    print(predicted.shape)
    
    print(action.shape)
    
    
    #.gather(1,action)
   
    
    #print(predicted)
    #print(target)
    
    print(action)
    
    predicted = predicted.gather(1,action)
    
    time.sleep(10)
    
    return f.smooth_l1_loss(predicted, target[0])
