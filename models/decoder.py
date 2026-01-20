import torch
import torch.nn as nn

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1), # 32x32
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), # 16x16
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
            
            nn.AdaptiveAvgPool2d(1), # Reduz para 1x1
            nn.Flatten(),
            nn.Linear(128, 16) # Saída de 16 bits
        )

    def forward(self, x):
        return self.model(x)