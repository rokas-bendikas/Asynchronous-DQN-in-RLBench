import torch.nn as nn

from models.base import BaseModel
from models.commons import ResidualLinear
from simulator.cart_pole import CartPole
from simulator.RLBench import RLBench


class RLBenchModel(BaseModel):
    def __init__(self):
        super().__init__()

        self.fc = nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=16,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=16,out_channels=32,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=128,out_channels=128,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten(),
            nn.Linear(2048, out_features=128), 
            nn.ReLU(),
            ResidualLinear(128),
            nn.Linear(in_features=128, out_features=16)
        )

    def forward(self, x):
      
        if (len(x.size())==3):
            x = x.unsqueeze(0)
            
        x = x.permute(0,3,1,2)
        return self.fc(x)
