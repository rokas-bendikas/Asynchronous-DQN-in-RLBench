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



def data_to_queue(state, action, reward, next_state, terminal, queue):
    
    queue.put(as_tensor(state))
    queue.put(as_tensor([action],t.int64))
    queue.put(as_tensor([reward]))
    queue.put(as_tensor(next_state))
    queue.put(as_tensor([terminal],t.bool))
    



def queue_to_data(queue):
    
    state = queue.get()
    action = queue.get()
    reward = queue.get()
    next_state = queue.get()
    terminal = queue.get()
    
    return (state,action,reward,next_state,terminal)


class ReplayBuffer():
    def __init__(self,args):
        
        self.memory = deque(maxlen=args.buffer_size)
        self.priority = deque(maxlen=args.buffer_size)
        self.args = args
        
    def load_queues(self,queues,q_network,target_network):
        for q in queues:
            
            for i in range(int(q.qsize()/5)):
                
                # Read from the queue
                data = queue_to_data(q)
                
                # Push to the buffer
                self.memory.append(data)
                
                state, action, reward, next_state, terminal = data
                
                batch = (state.unsqueeze(0),action.unsqueeze(1),reward.unsqueeze(1),next_state.unsqueeze(0),terminal.unsqueeze(1))
        
                utility = calculate_loss(q_network, target_network, batch, self.args, Device.get_device()).item()
                
                self.priority.append(abs(utility) + 1)
                
                
                
    def prepare_batch(self):
        
        utils = list(self.priority)
        
        
        probs = utils/np.sum(utils)
        
        batch_size = min(len(self.memory), self.args.batch_size)
       
        batch_idx = np.random.choice(len(self.memory), size=batch_size,p=probs)
        
        batch = [self.memory[i] for i in batch_idx]
        
        states, actions, rewards, next_states, terminal = zip(*batch)
        
        return t.stack(states), t.stack(actions), t.stack(rewards), t.stack(next_states), t.stack(terminal)
    
    def __len__(self):
        
        return len(self.memory)
        
                
            
            
    