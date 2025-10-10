import torch
import torch.nn as nn
import torch.nn.functional as F

class ActionClassifier(nn.Module):
   
    def __init__(self, num_classes=2):
        super(ActionClassifier, self).__init__()
        
      
        
        # Layer 1: Input (2 channels: Horizontal Flow, Vertical Flow)
        self.conv1 = nn.Conv3d(2, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        
        # Layer 2
        self.conv2 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        
        # Layer 3
        self.conv3 = nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

       
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc1 = nn.Linear(256, 512)
        self.fc2 = nn.Linear(512, num_classes) 

    def forward(self, x):
        # x shape: (B, 2, T, H, W)
        
        # Conv Layers
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        
        x = F.relu(self.conv3(x))
        x = self.pool3(x)

        x = self.avgpool(x) 
        x = x.view(x.size(0), -1)
        
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x

if __name__ == '__main__':
  
    
  
    dummy_input = torch.randn(4, 2, 16, 120, 160)
    
    model = ActionClassifier(num_classes=2)
    output = model(dummy_input)
   
    print(f"Model Output Shape (Expected: 4, 2): {output.shape}")