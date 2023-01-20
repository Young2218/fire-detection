import torch
import torch.nn as nn
import torch.nn.functional as F


def get_size_after_max_pool(start_size, n):
    if n == 0:
        return start_size
    else:
        size = (start_size-3)//2 + 1
        return get_size_after_max_pool(size, n-1)

class SBNN(nn.Module):
    def __init__(self, img_size, num_class) -> None:
        super(SBNN, self).__init__()
        fc_layer_input_size = get_size_after_max_pool(img_size, 3)
        
        self.conv_layers = nn.Sequential(
            # Layer 1,2,3
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2), # 48 -> 23
            
            # Layer 4,5,6
            nn.BatchNorm2d(num_features=32),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=64),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2), # 23 -> 11
            
            # Layer 7,8,9
            nn.BatchNorm2d(num_features=64),
            nn.Conv2d(in_channels=64, out_channels=384, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=384),
            nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2), # 11 -> 5
        )
        
        self.fc_layers = nn.Sequential(
            nn.Linear(fc_layer_input_size**2 * 384, 2048),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(2048, num_class)
        )
        
         
        
    def forward(self, x):
        x = self.conv_layers(x)
        x = torch.flatten(x,1)
        x = self.fc_layers(x)
        x = F.softmax(x)
        
        return x

if  __name__ == "__main__":
    import numpy as np
    sample_mat = np.random.randint(0, 255, (1,3,128,128))/255
    sample_tensor = torch.from_numpy(sample_mat).float()
    print(sample_tensor.shape)
    print(sample_tensor.dtype)
    
    net = SBNN(128,3)
    print(net)
    a = net.forward(sample_tensor)
    print(a.shape, a)
    