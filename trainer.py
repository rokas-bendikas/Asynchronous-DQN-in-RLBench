import argparse
import os
import shutil

import torch.multiprocessing as mp
from torch.multiprocessing import set_start_method

from checkpoint import checkpoint
from environments import environments
from optimiser import optimise
from explorer import explore




    
    
def main():
    os.environ['PYTHONWARNINGS'] = 'ignore:semaphore_tracker:UserWarning'
    mp.set_start_method('spawn', True)

    shutil.rmtree('runs', ignore_errors=True)
    if not os.path.exists('logs'):
        os.makedirs('logs')
    if not os.path.exists('trained'):
        os.makedirs('trained')

    parser = argparse.ArgumentParser()

    parser.add_argument('--environment', default='RLBench', help='Environment to use for training [default = RLBench]')
    parser.add_argument('--save_model', default='', help='Path to save the model [default = '']')
    parser.add_argument('--load_model', default='', help='Path to load the model [default = '']')
    parser.add_argument('--n_workers', default=1, type=int, help='Number of workers [default = 1]')
    parser.add_argument('--target_update_frequency', default=10, type=int, help='Frequency for syncing target network [default = 10]')
    parser.add_argument('--checkpoint_frequency', default=10, type=int, help='Frequency for creating checkpoints [default = 10]')

    parser.add_argument('--lr', default=0.0005, type=float, help='Learning rate for the training [default = 0.0005]')
    parser.add_argument('--batch_size', default=32, type=int, help='Batch size for the training [default = 32]')
    parser.add_argument('--gamma', default=0.99, type=float, help='Discount factor for the training [default = 0.99]')
    parser.add_argument('--eps', default=0.999, type=float, help='Greedy constant for the training [default = 0.999]')
    parser.add_argument('--min_eps', default=0.1, type=float, help='Minimum value for greedy constant [default = 0.1]')
    parser.add_argument('--buffer_size', default=100000, type=int, help='Buffer size [default = 100000]')
    parser.add_argument('--max_grad_norm', default=10, type=float, help='Maximum value of L2 norm for gradients [default = 10]')
    parser.add_argument('--iter_length', default=100, type=int, help='Maximum number of iterations [default=100]')

    args = parser.parse_args()
    

    SIMULATOR, NETWORK = environments[args.environment]
    model = NETWORK()
    model.load(args.load_model)
    model.share_memory()

    lock = mp.Lock()
    
    buffer = mp.Queue(args.buffer_size)
    
    
    
    workers_explore = [mp.Process(target=explore,args=(idx,SIMULATOR,model,buffer,args)) for idx in range(args.n_workers)]
    workers_explore.append(mp.Process(target=checkpoint, args=(model, args)))
    
    workers_optimize = mp.Process(target=optimise,args=(0, model, buffer, args, lock))
   
    [p.start() for p in workers_explore]
    print("Succesfully started explorers!")
    
    workers_optimize.start()
    print("Succesfully started optimisers!")


    try:
        [p.join() for p in workers_explore]
        print("Explorers joint!")
        
        workers_optimize.join()
        print("Optimiser joint!")
        
    except Exception as e:
        print(e)
    except KeyboardInterrupt:
        print('<< EXITING >>')
    finally:
        [p.kill() for p in workers_explore]
        workers_optimize.kill()
        buffer.close()
        

        os.system('clear')
        if input('Save model? (y/n): ') in ['y', 'Y', 'yes']:
            print('<< SAVING MODEL >>')
            model.save(args.save_model)
            
            
if __name__ == '__main__':
    main()