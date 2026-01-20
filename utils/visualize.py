import torch
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import numpy as np

def visualize_watermark_and_residuals(original_image, marked_image, save_path=None, filename="watermark_analysis.png"):
    """
    Gera uma imagem comparativa mostrando: Imagem Original | Imagem com Marca | Resíduos (amplificados).

    Args:
        original_image (torch.Tensor): Tensor da imagem original (CxHxW).
        marked_image (torch.Tensor): Tensor da imagem com marca d'água (CxHxW).
        save_path (str, optional): Caminho da pasta para salvar a imagem. Se None, apenas mostra.
        filename (str): Nome do arquivo a ser salvo.
    """
    if not isinstance(original_image, torch.Tensor) or not isinstance(marked_image, torch.Tensor):
        raise TypeError("As imagens devem ser tensores PyTorch.")
    if original_image.dim() != 3 or marked_image.dim() != 3:
        raise ValueError("As imagens devem ter 3 dimensões (C, H, W).")

    # Garante que as imagens estejam na CPU e sem gradientes
    original_image = original_image.cpu().detach()
    marked_image = marked_image.cpu().detach()

    # Calcular o resíduo (diferença)
    residual = torch.abs(marked_image - original_image) # Usamos abs para ver magnitude da diferença
    
    # Normalizar e amplificar o resíduo para visualização
    # Multiplicamos por um fator para tornar as pequenas diferenças visíveis
    residual_amplified = residual * 10.0 # Ajuste este fator conforme necessário
    residual_amplified = torch.clamp(residual_amplified, 0, 1) # Clampa entre 0 e 1

    # Empilhar as três imagens horizontalmente
    # Adicionamos uma dimensão de batch para vutils.make_grid
    display_grid = vutils.make_grid([original_image, marked_image, residual_amplified], nrow=3, padding=5, normalize=True)

    # Converter para formato que o matplotlib entende (HxWxC)
    plt_image = display_grid.permute(1, 2, 0).numpy()

    # Plotar a imagem
    plt.figure(figsize=(15, 5))
    plt.imshow(plt_image)
    plt.title("Original Image | Watermarked Image | Residuals (Amplified)")
    plt.axis('off')
    
    if save_path:
        import os
        os.makedirs(save_path, exist_ok=True)
        full_path = os.path.join(save_path, filename)
        plt.savefig(full_path, bbox_inches='tight')
        print(f"Análise de marca d'água salva em: {full_path}")
    else:
        plt.show()
    
    plt.close() # Fecha a figura para liberar memória