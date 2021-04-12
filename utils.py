import random


import torch as t
from torch.autograd import Variable

from device import Device



def copy_gradients(target, source):
    for shared_param, param in zip(target.parameters(), source.parameters()):
        if param.grad is not None:
            shared_param._grad = param.grad.clone().cpu()



def prepare_batch(buffer,args,lock):
    
    data = []
    
    batch_size = args.batch_size
    
    for i in range(batch_size):
        data_from_buffer = buffer.get()
        data_processed = buffer_to_data(data_from_buffer)
        data.append(data_processed)
        
    
    random.shuffle(data)
    
   
    states, actions, rewards, next_states, terminal = zip(*data)
    
    return t.stack(states), t.stack(actions), t.stack(rewards), t.stack(next_states), t.stack(terminal)

def as_tensor(x, dtype=t.float32):
    return t.tensor(x, dtype=dtype, device=Device.get_device())



def data_to_buffer(state, action, reward, next_state, terminal):
    
    state = as_tensor(state).unsqueeze(3)
    action = as_tensor([action], t.long).unsqueeze(1).unsqueeze(2).unsqueeze(3).expand(128, 128,12,1)
    reward = as_tensor([reward]).unsqueeze(1).unsqueeze(2).unsqueeze(3).expand(128, 128,12,1)
    next_state = as_tensor(next_state).unsqueeze(3)
    terminal = as_tensor([terminal],t.bool).unsqueeze(1).unsqueeze(2).unsqueeze(3).expand(128, 128,12,1)
    
    
    
    data = t.cat((state,action,reward,next_state,terminal),dim=3)
    
    return data



def buffer_to_data(data):
    
 
    
    state = data[:,:,:,0]
    action = data[0,0,0,1]
    reward = data[0,0,0,2]
    next_state = data[:,:,:,3]
    terminal = data[0,0,0,4]
    
    
    return (state,action,reward,next_state,terminal)




def reset_states(hx, cx):
		hx[:, :] = 0
		cx[:, :] = 0
		return hx.detach(), cx.detach()

def init_states(batch_dim):
		hx = Variable(t.zeros(batch_dim, 12))
		cx = Variable(t.zeros(batch_dim, 12))
		return hx, cx
    
def action_onehot(action):
    a_onehot = t.zeros(12)  
    a_onehot[action] = 1
        
    return a_onehot
            
    