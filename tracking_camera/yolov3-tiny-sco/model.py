import torch
import torch.nn as nn


# Defining CNN Block
class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_batch_norm=True, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=not use_batch_norm, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.LeakyReLU(0.1)
        self.use_batch_norm = use_batch_norm

    def forward(self, x):
        x = self.conv(x)
        if self.use_batch_norm:
            x = self.bn(x)
            return self.activation(x)
        else:
            return x
        
class YOLOv2(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        self.in_channels = in_channels

        # Based on darknet-19
        self.features = nn.Sequential(
            CNNBlock(in_channels, 32, kernel_size=3, stride=1, padding=1), 
            nn.MaxPool2d(2,2), 

            CNNBlock(32, 64, kernel_size=3, stride=1, padding=1), 
            nn.MaxPool2d(2,2), 

            CNNBlock(64, 128, kernel_size=3, stride=1, padding=1), 
            CNNBlock(128, 64, kernel_size=1, stride=1, padding=0), 
            CNNBlock(64, 128, kernel_size=3, stride=1, padding=1), 
            nn.MaxPool2d(2,2), 

            CNNBlock(128, 256, kernel_size=3, stride=1, padding=1), 
            CNNBlock(256, 128, kernel_size=1, stride=1, padding=0), 
            CNNBlock(128, 256, kernel_size=3, stride=1, padding=1), 
            nn.MaxPool2d(2,2), 

            CNNBlock(256, 512, kernel_size=3, stride=1, padding=1), 
            CNNBlock(512, 256, kernel_size=1, stride=1, padding=0), 
            CNNBlock(256, 512, kernel_size=3, stride=1, padding=1), 
            nn.MaxPool2d(2,2), 

            CNNBlock(512, 1024, kernel_size=3, stride=1, padding=1), 
            CNNBlock(1024, 512, kernel_size=1, stride=1, padding=0), 
            CNNBlock(512, 1024, kernel_size=3, stride=1, padding=1), 
            CNNBlock(1024, 512, kernel_size=1, stride=1, padding=0), 
            CNNBlock(512, 1024, kernel_size=3, stride=1, padding=1),

            CNNBlock(1024, 5, kernel_size=1, stride=1, padding=0) # (1, 5, S, S) : S = grid size, 5 -> (confidence, center x, center y, width, height)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.sigmoid(x) # Convert to normailzed value (0,1)
        return x
    
if __name__ == "__main__":
    IMAGE_SIZE = 640
    model = YOLOv2()
    x = torch.randn((1, 3, IMAGE_SIZE, IMAGE_SIZE))
    out = model(x)
    print(out.shape)
