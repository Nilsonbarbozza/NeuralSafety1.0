import torch
import torch.nn as nn
import torch.nn.functional as F

class ProfessionalNoiseLayer(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        # 1. Simulação de Redimensionamento (Resizing)
        original_shape = x.shape[2:]
        if torch.rand(1) > 0.5:
            scale = torch.rand(1).item() * 0.5 + 0.5 # Escala entre 0.5 e 1.0
            x = F.interpolate(x, scale_factor=scale, mode='bilinear', align_corners=False)
            x = F.interpolate(x, size=original_shape, mode='bilinear', align_corners=False)
        
        # 2. Ruído Gaussiano
        if torch.rand(1) > 0.5:
            noise = torch.randn_like(x) * 0.02
            x = x + noise
            
        # 3. Simulação de Compressão (Suavização)
        if torch.rand(1) > 0.5:
            x = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
            
        return torch.clamp(x, 0, 1)