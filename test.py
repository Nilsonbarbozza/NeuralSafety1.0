import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import os
import numpy as np

# Importar seus modelos e utilitários
from models.encoder import Encoder
from models.decoder import Decoder
from models.noise_layer import ProfessionalNoiseLayer
from utils.visualize import visualize_watermark_and_residuals
from utils.metrics import evaluate_performance # Para calcular PSNR e Acc

def test_watermarking_system(image_path, model_path_encoder, model_path_decoder, message_to_hide=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- 1. Carregar Modelos Treinados ---
    encoder = Encoder().to(device)
    decoder = Decoder().to(device)

    # Carregar os pesos (state_dict) dos seus modelos salvos
    # Certifique-se de que os arquivos .pth existem na pasta 'checkpoints/'
    try:
        encoder.load_state_dict(torch.load(model_path_encoder, map_location=device))
        decoder.load_state_dict(torch.load(model_path_decoder, map_location=device))
        print(f"✅ Modelos carregados com sucesso de:\n- {model_path_encoder}\n- {model_path_decoder}")
    except FileNotFoundError:
        print(f"❌ Erro: Arquivos de modelos não encontrados nos caminhos especificados.")
        print("Por favor, certifique-se de que você salvou os modelos durante o treinamento.")
        return

    encoder.eval()
    decoder.eval()

    # --- 2. Preparar Imagem de Teste ---
    transform = transforms.Compose([
        transforms.Resize((64, 64)), # Mesma dimensão usada no treino
        transforms.ToTensor(),
    ])

    try:
        image = Image.open(image_path).convert("RGB")
        original_image_tensor = transform(image).unsqueeze(0).to(device) # Adiciona dimensão de batch
        print(f"✅ Imagem de teste '{image_path}' carregada.")
    except FileNotFoundError:
        print(f"❌ Erro: Imagem de teste não encontrada em '{image_path}'.")
        return
    except Exception as e:
        print(f"❌ Erro ao processar imagem '{image_path}': {e}")
        return

    # --- 3. Preparar Mensagem ---
    if message_to_hide is None:
        # Se nenhuma mensagem for fornecida, gera uma aleatória de 16 bits
        message = torch.randint(0, 2, (1, 16)).float().to(device)
        print(f"Mensagem aleatória gerada: {message.int().tolist()}")
    else:
        # Converte a lista de int para tensor de float
        message = torch.tensor([float(b) for b in message_to_hide]).unsqueeze(0).to(device)
        if message.shape[1] != 16:
            raise ValueError("A mensagem deve ter 16 bits.")
        print(f"Mensagem fornecida: {message.int().tolist()}")

    # --- 4. Esconder a Mensagem (Encoder) ---
    with torch.no_grad():
        marked_image_tensor = encoder(original_image_tensor, message)

    # --- 5. Simular Ruído (Opcional, para testar robustez) ---
    # Usamos a mesma NoiseLayer do treino para simular ataques
    noise_layer = ProfessionalNoiseLayer().to(device)
    noise_layer.eval() # Em eval mode
    
    with torch.no_grad():
        noisy_marked_image_tensor = noise_layer(marked_image_tensor)
    
    print("✅ Imagem processada pelo Encoder e com ruído simulado.")

    # --- 6. Recuperar a Mensagem (Decoder) ---
    with torch.no_grad():
        recovered_message_logits = decoder(noisy_marked_image_tensor)
        recovered_message = (torch.sigmoid(recovered_message_logits) > 0.5).float()
    
    print(f"Mensagem recuperada (com ruído): {recovered_message.int().tolist()}")

    # --- 7. Avaliar Desempenho ---
    # Cálculo do Bit Accuracy
    acc = (recovered_message == message).float().mean().item()
    print(f"✨ Acurácia de Recuperação da Mensagem (com ruído): {acc*100:.2f}%")

    # Cálculo do PSNR (entre original e marcada, e entre original e ruidosa)
    # Obs: evaluate_performance espera um loader, vamos adaptar para uma única imagem aqui
    mse_marked = F.mse_loss(marked_image_tensor, original_image_tensor)
    psnr_marked = 10 * torch.log10(1 / mse_marked).item()
    print(f"PSNR (Original vs Marcada): {psnr_marked:.2f} dB")
    
    mse_noisy = F.mse_loss(noisy_marked_image_tensor, original_image_tensor)
    psnr_noisy = 10 * torch.log10(1 / mse_noisy).item()
    print(f"PSNR (Original vs Marcada com Ruído): {psnr_noisy:.2f} dB")


    # --- 8. Visualizar Resultados ---
    output_dir = "results/test_visualizations"
    os.makedirs(output_dir, exist_ok=True)
    
    # Criar uma imagem com 4 painéis (Original | Marcada | Marcada+Ruído | Resíduos)
    # visualize_watermark_and_residuals já faz Original | Marcada | Resíduos
    # Precisamos de uma adaptação para 4 painéis ou rodar visualize 2x
    
    # Visualização Original, Marcada, Resíduo
    visualize_watermark_and_residuals(
        original_image=original_image_tensor[0],
        marked_image=marked_image_tensor[0],
        save_path=output_dir,
        filename="watermark_original_vs_marked.png"
    )

    # Visualização Original, Marcada com Ruído, Resíduo (da imagem ruidosa)
    visualize_watermark_and_residuals(
        original_image=original_image_tensor[0],
        marked_image=noisy_marked_image_tensor[0],
        save_path=output_dir,
        filename="watermark_original_vs_noisy_marked.png"
    )

if __name__ == "__main__":
    # --- CONFIGURAÇÕES DE TESTE ---
    # ⚠️ 1. Caminho para uma imagem de teste no seu computador
    TEST_IMAGE_PATH = "data/treino/base_fotos/foto.jpg" # EX: "data/test_images/cat.jpg"
    
    # ⚠️ 2. Caminhos para os modelos salvos (a serem criados no train.py)
    # Lembre-se de salvar seus modelos no train.py com torch.save(model.state_dict(), 'checkpoints/encoder.pth')
    MODEL_PATH_ENCODER = "checkpoints/encoder_final.pth"
    MODEL_PATH_DECODER = "checkpoints/decoder_final.pth"
    
    # ⚠️ 3. Mensagem para esconder (opcional, se None, será aleatória)
    # Exemplo: [0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 1]
    MESSAGE = None 

    print("Iniciando teste do sistema de marca d'água...\n")
    test_watermarking_system(
        image_path=TEST_IMAGE_PATH,
        model_path_encoder=MODEL_PATH_ENCODER,
        model_path_decoder=MODEL_PATH_DECODER,
        message_to_hide=MESSAGE
    )
    print("\nTeste concluído. Verifique a pasta 'results/test_visualizations'.")