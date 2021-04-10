import random

import torch as t

from device import Device

from collections import deque


def copy_gradients(target, source):
    for shared_param, param in zip(target.parameters(), source.parameters()):
        if param.grad is not None:
            shared_param._grad = param.grad.clone().cpu()

"""
def prepare_batch(buffer, batch_size):
    batch_size = min(len(buffer), batch_size)
    batch = random.sample(buffer, batch_size)

    states, actions, rewards, next_states, terminal = zip(*batch)

    return t.stack(states), t.stack(actions), t.stack(rewards), t.stack(next_states), t.stack(terminal)
"""

def prepare_batch(buffer,args,lock):
    
    data = []
    
    # The critical section begins
    lock.acquire()
    
    buffer_size = buffer.qsize()
    
    for i in range(buffer_size):
        data_from_buffer = buffer.get()
        data_processed = buffer_to_data(data_from_buffer)
        data.append(data_processed)
        buffer.put(data_from_buffer)
    
    batch_size = min(buffer_size, args.batch_size)
    lock.release()
    
    # The critical section ends
    

    batch = random.sample(data, batch_size)
    
   
    states, actions, rewards, next_states, terminal = zip(*batch)
    
    return t.stack(states), t.stack(actions), t.stack(rewards), t.stack(next_states), t.stack(terminal)

def as_tensor(x, dtype=t.float32):
    return t.tensor(x, dtype=dtype, device=Device.get_device())

"""
def transition_to_tensor(state, action, reward, next_state, terminal):
    
    
    return (as_tensor(state),
            as_tensor([action], t.long),
            as_tensor([reward]),
            as_tensor(next_state),
            as_tensor([not terminal]))

"""

def data_to_buffer(state, action, reward, next_state, terminal):
    
    state = as_tensor(state).unsqueeze(3)
    action = as_tensor([action], t.long).unsqueeze(1).unsqueeze(2).unsqueeze(3).expand(128, 128,3,1)
    reward = as_tensor([reward]).unsqueeze(1).unsqueeze(2).unsqueeze(3).expand(128, 128,3,1)
    next_state = as_tensor(next_state).unsqueeze(3)
    terminal = as_tensor([not terminal]).unsqueeze(1).unsqueeze(2).unsqueeze(3).expand(128, 128,3,1)
    

    
    data = t.cat((state,action,reward,next_state,terminal),dim=3)
    
    
    return data



def buffer_to_data(data):
    
 
    
    state = data[:,:,:,0]
    action = data[0,0,0,1]
    reward = data[0,0,0,2]
    next_state = data[:,:,:,3]
    terminal = data[0,0,0,4]

    
    
    return (state,action,reward,next_state,terminal)