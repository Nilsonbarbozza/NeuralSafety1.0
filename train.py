import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms, datasets
import os
import matplotlib.pyplot as plt
from datetime import datetime  # Novo: Para data e hora no nome do arquivo

# --- IMPORTAÇÕES DE ARQUIVOS EXTERNOS ---
from models.encoder import Encoder
from models.decoder import Decoder
from models.discriminator import Discriminator
from models.noise_layer import ProfessionalNoiseLayer
from utils.visualize import visualize_watermark_and_residuals

def train_professional_system(encoder, decoder, discriminator, train_loader, epochs=50):
    device = torch.device("xpu" if torch.xpu.is_available() else "cpu")
    
    encoder.to(device)
    decoder.to(device)
    discriminator.to(device)

    opt_enc_dec = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=1e-4)
    opt_disc = optim.Adam(discriminator.parameters(), lr=5e-5)
    noise_layer = ProfessionalNoiseLayer().to(device)

    scheduler_enc_dec = optim.lr_scheduler.ReduceLROnPlateau(opt_enc_dec, mode='max', factor=0.5, patience=5)

    best_acc = 0.0
    
    # AJUSTE 1: Adicionado "w_msg" ao histórico
    history = {
            "acc": [], 
            "loss_img": [], 
            "loss_adv": [], 
            "w_msg": []
        }
    for epoch in range(epochs):
        encoder.train()
        decoder.train()
        discriminator.train()

        total_acc = 0.0
        total_loss_img = 0.0
        total_loss_adv = 0.0
        num_batches = len(train_loader)
        w_msg = min(10.0 + epoch * 0.2, 10.0)

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

            # Acurácia
            preds = (torch.sigmoid(bits_pred) > 0.5).float()
            current_acc = (preds == msgs).float().mean()
            total_acc += current_acc.item()

            l_img = F.mse_loss(imgs_wmk, images)
            total_loss_img += l_img.item()
            
            l_msg = F.binary_cross_entropy_with_logits(bits_pred, msgs)
            l_adv = F.binary_cross_entropy(discriminator(imgs_wmk), torch.ones_like(d_real))
            total_loss_adv += l_adv.item()

            loss_g = (l_img * 15.0) + (l_msg * w_msg) + (l_adv * 0.5)
            loss_g.backward()
            opt_enc_dec.step()

        # Armazenar métricas da época
        avg_acc = (total_acc / num_batches) * 100
        avg_loss_img = total_loss_img / num_batches
        avg_loss_adv = total_loss_adv / num_batches

        history["loss_adv"].append(avg_loss_adv)
        history["acc"].append(avg_acc)
        history["loss_img"].append(avg_loss_img)
        history["w_msg"].append(w_msg) # Salva o peso da mensagem
        
        scheduler_enc_dec.step(avg_acc)

        print(f"Epoch [{epoch + 1}/{epochs}] | Acc: {avg_acc:.2f}% | L_img: {avg_loss_img:.4f} | Peso Msg: {w_msg:.1f}")

        if avg_acc > best_acc:
            best_acc = avg_acc
            os.makedirs("checkpoints", exist_ok=True)
            torch.save(encoder.state_dict(), "checkpoints/encoder_best.pth")
            torch.save(decoder.state_dict(), "checkpoints/decoder_best.pth")
            print(f"⭐ Novo recorde de acurácia! Modelos salvos.")

        if (epoch + 1) % 5 == 0:
            encoder.eval()
            with torch.no_grad():
                visualize_watermark_and_residuals(
                    images[0], imgs_wmk[0], 
                    save_path="results/visualizations", 
                    filename=f"epoch_{epoch + 1}.png"
                )
            encoder.train()
            
    return history

def save_training_curves(history):
    epochs = range(1, len(history["acc"]) + 1)
    fig, ax1 = plt.subplots(figsize=(12, 7))

    # Eixo Esquerdo: Acurácia (0 a 100%)
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Acurácia (%)", color="blue", fontsize=12)
    ax1.plot(epochs, history["acc"], color="blue", label="Accuracy", linewidth=2)
    ax1.set_ylim([0, 105]) # Fixa o topo em 100% para melhor perspectiva
    ax1.tick_params(axis="y", labelcolor="blue")
    ax1.grid(True, linestyle='--', alpha=0.5)

    # Eixo Direito: Losses e Pesos
    ax2 = ax1.twinx()
    ax2.set_ylabel("Valor das Perdas / Pesos", color="black", fontsize=12)
    
    # Plotando L_img, L_adv e Peso Msg
    ax2.plot(epochs, history["loss_img"], color="red", label="L_img (MSE)", linestyle="-")
    ax2.plot(epochs, history.get("loss_adv", []), color="orange", label="L_adv (Adversarial)", linestyle="--")
    ax2.plot(epochs, history["w_msg"], color="green", label="Peso Msg (w_msg)", linestyle=":")
    
    # Adicionando legenda unificada
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc="upper left", frameon=True)

    plt.title("Progresso Detalhado do Treinamento")
    
    # AJUSTE: Salvando com data e hora exata
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs("results", exist_ok=True)
    save_path = f"results/learning_curves_{timestamp}.png"
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"📈 Novo gráfico detalhado salvo em: {save_path}")

def main():
    device = torch.device("xpu" if torch.xpu.is_available() else "cpu")
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    path_treino = "data/treino"

    if os.path.exists(path_treino):
        try:
            dataset = datasets.ImageFolder(root=path_treino, transform=transform)
            train_loader = DataLoader(dataset, batch_size=16, shuffle=True)
            print(f"✅ Dataset real carregado com {len(dataset)} imagens.")
        except Exception as e:
            print(f"❌ Erro ao carregar: {e}")
            print("💡 Dica: Verifique se as fotos estão dentro de uma SUBPASTA em data/treino")
            dummy_images = torch.rand(100, 3, 64, 64)
            train_loader = DataLoader(TensorDataset(dummy_images, torch.zeros(100)), batch_size=16)
    else:
        print("⚠️ Pasta não encontrada. Usando dados simulados.")
        dummy_images = torch.rand(100, 3, 64, 64)
        train_loader = DataLoader(TensorDataset(dummy_images, torch.zeros(100)), batch_size=16)
    
    encoder = Encoder()
    decoder = Decoder()
    discriminator = Discriminator()

    print(f"🚀 Iniciando treinamento em: {device}")
    history = train_professional_system(encoder, decoder, discriminator, train_loader, epochs=100)
    
    save_training_curves(history)

    print("\nSalvando modelos finais...")
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(encoder.state_dict(), "checkpoints/encoder_final.pth")
    torch.save(decoder.state_dict(), "checkpoints/decoder_final.pth")
    print("✅ Processo concluído!")

if __name__ == "__main__":
    main()