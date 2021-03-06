import logging
from copy import deepcopy
from itertools import count
import torch as t
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from calculate_loss import calculate_loss
from device import Device
from optimise_model import optimise_model
from utils import ReplayBuffer

t.multiprocessing.set_sharing_strategy('file_system')


def optimise(idx, shared_model, queues, args, lock):
    try:
        
        writer = SummaryWriter('runs/o{}'.format(idx))
        
        logging.basicConfig(filename='logs/optimiser:{:02}.log'.format(idx),
                                filemode='w',
                                format='%(message)s',
                                level=logging.DEBUG)
        
        sgd = t.optim.Adam(params=shared_model.parameters(), lr=args.lr)
    
        # allocate a device
        n_gpu = t.cuda.device_count()
        if n_gpu > 0:
            Device.set_device(0)
            
    
        q_network = deepcopy(shared_model)
        q_network.to(Device.get_device())
        q_network.train()
    
        target_network = deepcopy(q_network)
        target_network.to(Device.get_device())
        target_network.eval()
        
        buffer = ReplayBuffer(args)
    
    
        for itr in tqdm(count(), position=idx, desc='optimiser:{:02}'.format(idx)):
    
            
                
            buffer.load_queues(queues,q_network,target_network,lock,args)
            
            while(len(buffer) < min(args.n_workers*args.episode_length*args.warmup,args.buffer_size/2)):
                buffer.load_queues(queues,q_network,target_network,lock,args)
                continue
                
                
            # Sample a data point from dataset
            batch = buffer.prepare_batch(target_network,q_network)
                  
            # Sync local model with shared model
            q_network.load_state_dict(shared_model.state_dict())
                
            # Calculate loss for the batch
            loss = calculate_loss(q_network, target_network, batch, args, Device.get_device())
                
            # Optimise for the batch
            loss = optimise_model(shared_model, q_network, loss, sgd, args, lock)
                
            # Log the results
            logging.debug('Batch loss: {:.2f}, Buffer size: {}'.format(loss,len(buffer)))
            writer.add_scalar('Batch loss', loss,itr)
                
            if itr % args.target_update_frequency == 0:
                target_network.load_state_dict(q_network.state_dict())
                      
                      
        writer.close()
        
            
            
    except KeyboardInterrupt:
        print('exiting optimiser:{:02}'.format(idx))