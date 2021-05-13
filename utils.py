import torch as t
from device import Device
import torch.nn.functional as f
from cpprb import PrioritizedReplayBuffer




def copy_gradients(target, source):
    for shared_param, param in zip(target.parameters(), source.parameters()):
        if param.grad is not None:
            shared_param._grad = param.grad.clone().cpu()



def as_tensor(x, dtype=t.float32):
    return t.tensor(x, dtype=dtype, device=Device.get_device(),requires_grad=False)


   
def data_to_queue(state, action, reward, next_state, terminal):
    
    state = as_tensor(state).unsqueeze(3)
    action = as_tensor([action], t.int64).unsqueeze(1).unsqueeze(2).unsqueeze(3).expand(64, 64,6,1)
    reward = as_tensor([reward]).unsqueeze(1).unsqueeze(2).unsqueeze(3).expand(64, 64,6,1)
    next_state = as_tensor(next_state).unsqueeze(3)
    terminal = as_tensor([terminal],t.bool).unsqueeze(1).unsqueeze(2).unsqueeze(3).expand(64, 64,6,1)
    
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
        
        #self.memory = deque(maxlen=args.buffer_size)
        self.memory = PrioritizedReplayBuffer(args.buffer_size,
                              {"obs": {"shape": (64,64,6)},
                               "act": {},
                               "rew": {},
                               "next_obs": {"shape": (64,64,6)},
                               "terminal": {}})
        #self.priority = deque(maxlen=args.buffer_size)
        self.length = 0
        self.args = args
        
    def load_queues(self,queues,q_network,target_network,lock,args):
        for q in queues:
            
            for i in range(int(q.qsize())):
                
                
                # Read from the queue
                # The critical section begins
                lock.acquire()
                data = queue_to_data(q.get())
                lock.release()
                
                # Convert to numpy for storage
                state = data[0].numpy()
                action = data[1].numpy()
                reward = data[2].numpy()
                next_state = data[3].numpy()
                terminal = data[4].numpy()
                
                #data_np = (state,action,reward,next_state,terminal)
                
                # Push to the buffer
                #self.memory.append(data_np)
                
                
                self.memory.add(obs=state,
                                act=action,
                                rew=reward,
                                next_obs=next_state,
                                terminal=terminal)
                
                self.length = min(self.args.buffer_size,self.length+1)
                
                
               
                
    def prepare_batch(self,target_network,q_network):
        
        
        batch_size = min(self.length, self.args.batch_size)
        
        sample = self.memory.sample(batch_size)
        
        s = t.tensor(sample['obs'])
        a = t.tensor(sample['act'])
        r = t.tensor(sample['rew'])
        ns = t.tensor(sample['next_obs'])
        term = t.tensor(sample['terminal'])
    
        states = s.permute(0,3,1,2).to(Device.get_device())
        actions = a.type(t.int64).to(Device.get_device())
        rewards = r.to(Device.get_device())
        next_states = ns.permute(0,3,1,2).to(Device.get_device())
        terminals = term.to(Device.get_device())
        
        indexes = sample["indexes"]
            
        with t.no_grad():
              
            target = rewards + terminals * self.args.gamma * target_network(next_states).max()
            predicted = q_network(states).gather(1,actions)
                  
            
        new_priorities = f.smooth_l1_loss(predicted, target,reduction='none').cpu().numpy()
        new_priorities[new_priorities<1] = 1
            
        self.memory.update_priorities(indexes,new_priorities)
        
        
        return states,actions,rewards,next_states,terminals
    
    
    def __len__(self):
        
        return self.length
        
               
            
            
    
