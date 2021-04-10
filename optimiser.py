import logging
from collections import deque
from copy import deepcopy
from datetime import datetime
from itertools import count

import numpy as np
import torch as t
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from calculate_loss import calculate_loss
from device import Device
from optimise_model import optimise_model
from utils import prepare_batch, as_tensor



def optimise(idx, shared_model, buffer, args, lock):
    try:
        writer = SummaryWriter('runs/{}/optimiser:{:02}'.format(datetime.now().strftime("%d|%m_%H|%M"), idx))
        logging.basicConfig(filename='logs/optimiser:{:02}.log'.format(idx),
                                filemode='w',
                                format='%(message)s',
                                level=logging.DEBUG)
    
        sgd = t.optim.SGD(params=shared_model.parameters(), lr=args.lr)
    
        # allocate a device
        n_gpu = t.cuda.device_count()
        if n_gpu > 0:
            #Device.set_device(idx % n_gpu)
            Device.set_device(0)
    
        q_network = deepcopy(shared_model)
        q_network.to(Device.get_device())
        q_network.train()
    
        target_network = deepcopy(q_network)
        target_network.to(Device.get_device())
        target_network.eval()
    

    
        for itr in tqdm(count(), position=idx, desc='optimiser:{:02}'.format(idx)):
    
            
            for e in count():
                if(not buffer.empty()):
                
                
                    # Sample a data point from dataset
                    batch = prepare_batch(buffer, args,lock)
                    
                  
                    # Sync local model with shared model
                    q_network.load_state_dict(shared_model.state_dict())
                
                    # Calculate loss for the batch
                    loss = calculate_loss(q_network, target_network, batch, args,Device.get_device())
                
                    # Optimise for the batch
                    loss = optimise_model(shared_model, q_network, loss, sgd, args, lock)
                
                    # Log the results
                    logging.debug('Batch loss: {:.2f}'.format(loss))
                    writer.add_scalar('batch/loss', loss, e)
                
                     
              
            
            writer.close()
        
            if itr % args.target_update_frequency == 0:
                target_network.load_state_dict(q_network.state_dict())
            
    except KeyboardInterrupt:
        print('exiting optimiser:{:02}'.format(idx))