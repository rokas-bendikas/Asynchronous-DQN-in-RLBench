import torch as t
from device import Device
from collections import deque
import numpy as np
from calculate_loss import calculate_loss



def copy_gradients(target, source):
    for shared_param, param in zip(target.parameters(), source.parameters()):
        if param.grad is not None:
            shared_param._grad = param.grad.clone().cpu()



def as_tensor(x, dtype=t.float32):
    return t.tensor(x, dtype=dtype, device=Device.get_device())


   
def data_to_queue(state, action, reward, next_state, terminal):
    
    state = as_tensor(state).unsqueeze(3)
    action = as_tensor([action], t.int64).unsqueeze(1).unsqueeze(2).unsqueeze(3).expand(64, 64,12,1)
    reward = as_tensor([reward]).unsqueeze(1).unsqueeze(2).unsqueeze(3).expand(64, 64,12,1)
    next_state = as_tensor(next_state).unsqueeze(3)
    terminal = as_tensor([terminal],t.bool).unsqueeze(1).unsqueeze(2).unsqueeze(3).expand(64, 64,12,1)
    
    #print(state.shape)
    #print(action.shape)
    #print(reward.shape)
    #print(next_state.shape)
    #print(terminal.shape)
    
    data = t.cat((state,action,reward,next_state,terminal),dim=3)
    
    return data


def queue_to_data(data):
    
    state = data[:,:,:,0]
    action = data[0,0,0,1]
    reward = data[0,0,0,2]
    next_state = data[:,:,:,3]
    terminal = data[0,0,0,4]
    
    return (state,action,reward,next_state,terminal)
    


class ReplayBuffer():
    def __init__(self,args):
        
        self.memory = deque(maxlen=args.buffer_size)
        self.priority = deque(maxlen=args.buffer_size)
        self.args = args
        
    def load_queues(self,queues,q_network,target_network):
        for q in queues:
            
            for i in range(int(q.qsize())):
                
                # Read from the queue
                data = queue_to_data(q.get())
                
                # Push to the buffer
                self.memory.append(data)
                
                state, action, reward, next_state, terminal = data
                
                state = state.unsqueeze(0).permute(0,3,1,2)
                action = action.unsqueeze(0).type(t.int64)
                reward = reward.unsqueeze(0)
                next_state = next_state.unsqueeze(0).permute(0,3,1,2)
                terminal = terminal.unsqueeze(0)
                
                batch = (state,action,reward,next_state,terminal)
        
                utility = calculate_loss(q_network, target_network, batch, self.args, Device.get_device()).item()
                
                self.priority.append(abs(utility) + 1)
                
                
                
    def prepare_batch(self):
        
        utils = list(self.priority)
        
        
        probs = utils/np.sum(utils)
        
        batch_size = min(len(self.memory), self.args.batch_size)
       
        batch_idx = np.random.choice(len(self.memory), size=batch_size,p=probs)
        
        batch = [self.memory[i] for i in batch_idx]
        
        states, actions, rewards, next_states, terminal = zip(*batch)
        
        
        states = t.stack(states).permute(0,3,1,2)
        actions = t.stack(actions).type(t.int64)
        rewards = t.stack(rewards)
        next_states = t.stack(next_states).permute(0,3,1,2)
        terminal = t.stack(terminal)
        
        return states,actions,rewards,next_states,terminal
    
    def __len__(self):
        
        return len(self.memory)
        
                
            
            
    