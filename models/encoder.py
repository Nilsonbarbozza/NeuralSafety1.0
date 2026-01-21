import torch
import torch.nn as nn
import torch.nn.functional as F

class RobustEncoder(nn.Module):
    def __init__(self, msg_length=16, hidden_size=64):
        super(RobustEncoder, self).__init__()
        self.hidden_size = hidden_size
        
        # 1. Processamento Inicial da Imagem
        # Usamos padding=1 para manter a dimensão espacial constante
        self.conv1 = nn.Conv2d(3, hidden_size, kernel_size=3, padding=1)
        
        # 2. Processamento da Mensagem
        # Em vez de Linear fixo, projetamos a mensagem em 'hidden_size' canais
        self.msg_processor = nn.Linear(msg_length, hidden_size)
        
        # 3. Camadas de Fusão Profunda (ResNet-style)
        # Concatenaremos a imagem (hidden_size) + mensagem (hidden_size) = 128 canais
        self.fusion_layers = nn.Sequential(
            nn.Conv2d(hidden_size * 2, hidden_size, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(hidden_size), # Estabilidade no treinamento
            nn.Conv2d(hidden_size, hidden_size, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(hidden_size, 3, kernel_size=1) # Saída final: 3 canais (RGB)
        )

    def forward(self, image, msg):
        # image: [batch, 3, H, W]
        # msg: [batch, msg_length]
        
        # Passo 1: Extrair features da imagem
        x_img = F.relu(self.conv1(image)) # [batch, 64, H, W]
        
        # Passo 2: Orquestração Dinâmica da Mensagem
        # Expandimos a mensagem para ter a mesma dimensão espacial da imagem
        # (batch, 16) -> (batch, 64) -> (batch, 64, 1, 1)
        x_msg = self.msg_processor(msg).view(msg.size(0), self.hidden_size, 1, 1)
        # (batch, 64, 1, 1) -> (batch, 64, H, W)
        x_msg = x_msg.repeat(1, 1, image.shape[2], image.shape[3])
        
        # Passo 3: Concatenação de Tensores
        # Unimos as features da imagem com as da mensagem no eixo dos canais (dim 1)
        combined = torch.cat([x_img, x_msg], dim=1) # [batch, 128, H, W]
        
        # Passo 4: Residual Learning (Segredo da Robustez)
        # O modelo aprende apenas o "ruído" da marca d'água, não a imagem toda
        residual = self.fusion_layers(combined)
        
        # Clamping profissional: garante que a imagem saia entre 0 e 1 (range de cores)
        return torch.clamp(image + residual, 0, 1)
