import torch
import torch.nn as nn

class Decoder(nn.Module):
    def __init__(self, msg_length=16, hidden_size=128):
        super(Decoder, self).__init__()
        
        # Bloco de extração inicial
        self.features = nn.Sequential(
            # Camada 1: Captura variações sutis de cor/textura
            nn.Conv2d(3, hidden_size, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_size),
            nn.ReLU(inplace=True),
            
            # Camadas de Downsampling (Reduzem resolução, aumentam profundidade)
            self._make_layer(hidden_size, hidden_size),   # H x W
            self._make_layer(hidden_size, hidden_size*2), # H/2 x W/2
            self._make_layer(hidden_size*2, hidden_size*4), # H/4 x W/4
        )
        
        # Cabeça de Classificação (Recuperação dos Bits)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), # Resume o mapa de features em um vetor
            nn.Flatten(),
            nn.Linear(hidden_size * 4, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, msg_length) # Saída: Logits dos 16 bits
        )

    def _make_layer(self, in_channels, out_channels):
        """Cria um bloco de convolução com stride para reduzir a dimensão"""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # x: Imagem que pode ter sofrido ruído/ataque
        features = self.features(x)
        logits = self.classifier(features)
        
        # Em Deep Learning, retornamos os LOGITS (valores sem ativação final)
        # Para converter em 0 e 1 no teste, usaríamos torch.sigmoid(logits) > 0.5
        return logits
