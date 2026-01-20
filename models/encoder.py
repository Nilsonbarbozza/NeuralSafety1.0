import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        # Processamento da Imagem
        self.image_pre = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        # Processamento da Mensagem (16 bits -> expandir para o tamanho da imagem)
        self.msg_expand = nn.Linear(16, 64 * 64) # Exemplo para imagem 64x64
        
        # Fusão de Imagem + Mensagem
        self.fusion = nn.Sequential(
            nn.Conv2d(64 + 1, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 3, kernel_size=3, padding=1)
        )

    def forward(self, image, msg):
        x_img = self.image_pre(image)
        
        # Transforma a mensagem num "mapa" de 1 canal para concatenar com a imagem
        batch_size = image.shape[0]
        x_msg = self.msg_expand(msg).view(batch_size, 1, image.shape[2], image.shape[3])
        
        combined = torch.cat([x_img, x_msg], dim=1)
        residual = self.fusion(combined)
        
        return torch.clamp(image + residual, 0, 1)