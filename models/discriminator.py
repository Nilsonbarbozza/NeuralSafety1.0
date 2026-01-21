import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, hidden_size=64): # Adicionado o parâmetro aqui
        super(Discriminator, self).__init__()
        
        # Usamos Spectral Normalization para estabilidade padrão 2026
        def conv_block(in_channels, out_channels, stride=2):
            return nn.Sequential(
                nn.utils.spectral_norm(nn.Conv2d(in_channels, out_channels, 3, stride, 1)),
                nn.LeakyReLU(0.2, inplace=True),
                nn.BatchNorm2d(out_channels)
            )

        self.model = nn.Sequential(
            conv_block(3, hidden_size),           # Saída: 32x32
            conv_block(hidden_size, hidden_size*2),   # Saída: 16x16
            conv_block(hidden_size*2, hidden_size*4), # Saída: 8x8
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(hidden_size * 4, 1)
            # Sem Sigmoid aqui, pois usamos BCEWithLogitsLoss no treino
        )

    def forward(self, x):
        return self.model(x)
