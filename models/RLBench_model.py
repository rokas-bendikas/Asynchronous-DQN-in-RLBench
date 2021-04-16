import torch.nn as nn

from models.base import BaseModel



class RLBenchModel(BaseModel):
    def __init__(self):
        super().__init__()
        

        self.fc = nn.Sequential(
            nn.Conv2d(in_channels=12,out_channels=32,kernel_size=5,padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=32,out_channels=64,kernel_size=5,padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=64,out_channels=64,kernel_size=5,padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten(),
            nn.Linear(4096, out_features=256), 
            nn.ReLU(),
            nn.Linear(256, out_features=14), 
        )
        
        
        

    def forward(self,x):
      
        if (len(x.size())==3):
            x = x.unsqueeze(0)
            
        x = x.permute(0,3,1,2)
        
        y = self.fc(x)
        
        
        
        return y
