import torch
import torch.nn as nn
import torch.nn.functional as F


class SCNN(nn.Module):
    def __init__(self, num_class) -> None:
        super(SCNN, self).__init__()     
        
        self.conv_layers1 = nn.Sequential(
            # Layer 1
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=9, padding=4),
            nn.ReLU()
        )
        self.conv_layers2 = nn.Sequential(
            # Layer  2,3,4,5
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=1, padding=0, stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, stride=1),
            nn.ReLU()
        )
        self.conv_layers3 = nn.Sequential(
            # Layer 6 ~ 13
            nn.Conv2d(in_channels= 96, out_channels=64, kernel_size=1, padding=0, stride=1),
            nn.ReLU(), # 6
            nn.MaxPool2d(kernel_size=3, stride=2), # 7
            nn.BatchNorm2d(num_features=64),
            nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3,stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=128),
            nn.Conv2d(in_channels=128,out_channels=128,kernel_size=3,stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2), # 10
            nn.BatchNorm2d(num_features=128),
            nn.Conv2d(in_channels=128,out_channels=256,kernel_size=3,stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=256),
            nn.Conv2d(in_channels=256,out_channels=256,kernel_size=3,stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=num_class, kernel_size=1, stride=1, padding=1),
            nn.ReLU()            
        )        
         
        
    def forward(self, x):
        x1 = self.conv_layers1(x)
        x2 = self.conv_layers2(x1)
        x = torch.cat([x1,x2], 1)
        x = self.conv_layers3(x)
        x = torch.mean(x.view(x.size(0), x.size(1), -1), dim=2)
        x = F.softmax(x)
        
        return x

if  __name__ == "__main__":
    import numpy as np
    sample_mat = np.random.randint(0, 255, (1,3,128,128))/255
    sample_tensor = torch.from_numpy(sample_mat).float()
    print(sample_tensor.shape)
    print(sample_tensor.dtype)
    
    net = SCNN(3)
    print(net)
    a= net.forward(sample_tensor)
    print(a)
    