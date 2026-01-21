import torch
import torch.nn as nn
import torch.nn.functional as F

class RobustNoiseLayer(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        # x: Imagem Marcada vinda do Encoder [B, 3, H, W]
        
        # 1. Simulação Diferenciável de JPEG (Suavização Seletiva)
        # Em 2026, usamos desfoque gaussiano aleatório para simular perda de detalhes
        if torch.rand(1) > 0.7:
            kernel_size = 3
            sigma = torch.rand(1).item() * 1.5 + 0.1
            x = self.gaussian_blur(x, kernel_size, sigma)

        # 2. Redimensionamento Dinâmico (Resizing)
        # Essencial para resistir a fotos tiradas de telas ou compressão de redes sociais
        if torch.rand(1) > 0.5:
            h, w = x.shape[2:]
            scale = torch.rand(1).item() * 0.4 + 0.6 # Escala entre 0.6 e 1.0
            new_h, new_w = int(h * scale), int(w * scale)
            x = F.interpolate(x, size=(new_h, new_w), mode='bilinear', align_corners=False)
            x = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=False)
        
        # 3. Ruído de Brilho e Contraste (Color Jittering)
        # Ajuda a marca a sobreviver a edições de filtros (Instagram/TikTok)
        if torch.rand(1) > 0.3:
            shift = (torch.rand(1).item() - 0.3) * 0.2
            x = torch.clamp(x + shift, 0, 1)
            
        # 4. Adição de Ruído Branco (Gaussian Noise)
        if torch.rand(1) > 0.3:
            std = torch.rand(1).item() * 0.03
            x = x + torch.randn_like(x) * std
            
        return torch.clamp(x, 0, 1)

    def gaussian_blur(self, x, kernel_size, sigma):
        # Função auxiliar para manter o desfoque dentro do fluxo de gradiente
        coords = torch.arange(kernel_size).to(x.device) - (kernel_size - 1) / 2.
        g = torch.exp(-(coords**2) / (2. * sigma**2))
        g = g / g.sum()
        kernel = g.view(1, 1, -1, 1) * g.view(1, 1, 1, -1)
        kernel = kernel.repeat(3, 1, 1, 1)
        return F.conv2d(x, kernel, padding=kernel_size//2, groups=3)
