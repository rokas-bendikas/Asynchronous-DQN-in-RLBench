from models.cart_pole_model import CartPoleModel
from models.RLBench_model import RLBenchModel
from simulator.cart_pole import CartPole
from simulator.RLBench import RLBench

environments = {
    'cartpole': (CartPole, CartPoleModel),
    'RLBench': (RLBench, RLBenchModel)
}
