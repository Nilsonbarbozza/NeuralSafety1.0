import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# --- IMPORTAÇÕES DE ARQUIVOS EXTERNOS ---
# Agora que os arquivos existem em /models e /utils, apenas os chamamos
from models.encoder import Encoder
from models.decoder import Decoder
from models.discriminator import Discriminator
from models.noise_layer import ProfessionalNoiseLayer
from utils.metrics import evaluate_performance
from utils.visualize import visualize_watermark_and_residuals

from torchvision import transforms, datasets

import os


def train_professional_system(encoder, decoder, discriminator, train_loader, epochs=50):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder.to(device)
    decoder.to(device)
    discriminator.to(device)

    opt_enc_dec = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=1e-4)
    opt_disc = torch.optim.Adam(discriminator.parameters(), lr=5e-5)
    noise_layer = ProfessionalNoiseLayer().to(device)

    best_acc = 0.0  # Para rastrear a melhor acurácia

    for epoch in range(epochs):
        encoder.train()
        decoder.train()
        
        total_acc = 0.0
        num_batches = len(train_loader)
        w_msg = min(1.0 + epoch * 0.5, 10.0) # Aumentei o peso da mensagem para subir a acurácia

        for images, _ in train_loader:
            images = images.to(device)
            batch_size = images.size(0)
            msgs = torch.randint(0, 2, (batch_size, 16)).float().to(device)

            # --- Treino do Discriminador ---
            opt_disc.zero_grad()
            with torch.no_grad():
                imgs_wmk = encoder(images, msgs)
            d_real = discriminator(images)
            d_fake = discriminator(imgs_wmk)
            loss_d = F.binary_cross_entropy(d_real, torch.ones_like(d_real)) + \
                     F.binary_cross_entropy(d_fake, torch.zeros_like(d_fake))
            loss_d.backward()
            opt_disc.step()

            # --- Treino Encoder + Decoder ---
            opt_enc_dec.zero_grad()
            imgs_wmk = encoder(images, msgs)
            imgs_noisy = noise_layer(imgs_wmk)
            bits_pred = decoder(imgs_noisy)
            
            # Cálculo de Acurácia para monitoramento
            preds = (torch.sigmoid(bits_pred) > 0.5).float()
            current_acc = (preds == msgs).float().mean()
            total_acc += current_acc.item()

            l_img = F.mse_loss(imgs_wmk, images)
            l_msg = F.binary_cross_entropy_with_logits(bits_pred, msgs)
            l_adv = F.binary_cross_entropy(discriminator(imgs_wmk), torch.ones_like(d_real))
            
            loss_g = (l_img * 15.0) + (l_msg * w_msg) + (l_adv * 0.5)
            loss_g.backward()
            opt_enc_dec.step()

        avg_acc = (total_acc / num_batches) * 100
        print(f"Epoch [{epoch+1}/{epochs}] | Acc: {avg_acc:.2f}% | Peso Msg: {w_msg:.1f}")

        # SALVAR O MELHOR MODELO (Best Model)
        if avg_acc > best_acc:
            best_acc = avg_acc
            os.makedirs("checkpoints", exist_ok=True)
            torch.save(encoder.state_dict(), "checkpoints/encoder_best.pth")
            torch.save(decoder.state_dict(), "checkpoints/decoder_best.pth")
            print(f"⭐ Novo recorde de acurácia! Modelos salvos.")

        # Visualização periódica
        if (epoch + 1) % 5 == 0:
            visualize_watermark_and_residuals(images[0], imgs_wmk[0], save_path="results/visualizations", filename=f"epoch_{epoch+1}.png")
        # --- VISUALIZAÇÃO DE RESÍDUOS ---
        if (epoch + 1) % 5 == 0 or epoch == 0:
            encoder.eval()
            with torch.no_grad():
                sample_img, _ = next(iter(train_loader))
                sample_img = sample_img[:1].to(device)
                sample_msg = torch.randint(0, 2, (1, 16)).float().to(device)
                marked_sample = encoder(sample_img, sample_msg)

                visualize_watermark_and_residuals(
                    original_image=sample_img[0],
                    marked_image=marked_sample[0],
                    save_path="results/visualizations",
                    filename=f"epoch_{epoch + 1}.png",
                )


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Transformações: Redimensiona para 64x64 (tamanho que definimos no Encoder)
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
    ])

    # 2. Carregamento Inteligente
    if os.path.exists("data/treino") and len(os.listdir("data/treino")) > 0:
        try:
            dataset = datasets.ImageFolder(root="data/treino", transform=transform)
            train_loader = DataLoader(dataset, batch_size=16, shuffle=True)
            print(f"✅ Dataset real carregado com {len(dataset)} imagens.")
        except Exception as e:
            print(f"❌ Erro ao carregar pasta: {e}")
            # Fallback para dummy se a pasta estiver vazia ou mal estruturada
            dummy_images = torch.rand(100, 3, 64, 64)
            train_loader = DataLoader(TensorDataset(dummy_images, torch.zeros(100)), batch_size=16)
    else:
        print("⚠️ Pasta 'data/treino' não encontrada ou vazia. Usando dados simulados.")
        dummy_images = torch.rand(100, 3, 64, 64)
        train_loader = DataLoader(TensorDataset(dummy_images, torch.zeros(100)), batch_size=16)

    # --- INICIALIZAÇÃO DOS MODELOS ---
    encoder = Encoder()
    decoder = Decoder()
    discriminator = Discriminator()

    print(f"🚀 Iniciando treinamento em: {device}")
    train_professional_system(encoder, decoder, discriminator, train_loader, epochs=30)
    # No final do seu train.py, na função main(), após o loop de treino:
    # Salvar modelos após o treinamento
    print("\nSalvando modelos treinados...")
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(encoder.state_dict(), "checkpoints/encoder_final.pth")
    torch.save(decoder.state_dict(), "checkpoints/decoder_final.pth")
    print("Modelos salvos em 'checkpoints/'.")

if __name__ == "__main__":
    main()
