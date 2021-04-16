import torch as t
import torch.nn.functional as f




def calculate_loss(q_network, target_network, batch, hyperparameters,device):
    
    state, action, reward, next_state, terminal = batch
    

    state = state.to(device)
    reward = reward.to(device)
    next_state = next_state.to(device)
    terminal = terminal.to(device)
    action = action.to(device).unsqueeze(1)
    
    
    with t.no_grad():
        target = reward + terminal * hyperparameters.gamma * target_network(next_state).max()
        
        
    predicted = q_network(state).gather(1,action).squeeze(1)
    

    return f.smooth_l1_loss(predicted, target)
