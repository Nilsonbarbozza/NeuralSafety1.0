import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import os
import matplotlib.pyplot as plt
from datetime import datetime

# --- IMPORTAÇÕES ---
from models.encoder import RobustEncoder as Encoder
from models.decoder import Decoder
from models.discriminator import Discriminator
from models.noise_layer import RobustNoiseLayer as ProfessionalNoiseLayer


def train_professional_system(
    encoder, decoder, discriminator, train_loader, epochs=100
):
    # Detecta GPU (NVIDIA), XPU (Intel) ou CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print(f"🔥 Treinando na GPU: {torch.cuda.get_device_name(0)}")

    encoder.to(device)
    decoder.to(device)
    discriminator.to(device)
    noise_layer = ProfessionalNoiseLayer().to(device)

    # Otimizadores com Learning Rates diferenciados (padrão GAN)
    opt_enc_dec = optim.Adam(
        list(encoder.parameters()) + list(decoder.parameters()), lr=2e-4
    )
    opt_disc = optim.Adam(discriminator.parameters(), lr=1e-4)

    # Scheduler monitorando a acurácia para reduzir LR se estagnar
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        opt_enc_dec, mode="max", factor=0.5, patience=8
    )

    history = {"acc": [], "loss_img": [], "loss_adv": [], "w_msg": []}
    best_acc = 0.0

    for epoch in range(epochs):
        encoder.train()
        decoder.train()
        discriminator.train()
        total_acc, total_loss_img, total_loss_adv = 0.0, 0.0, 0.0

        # O peso da mensagem começa baixo para focar na invisibilidade e sobe
        w_msg = min(5.0 + epoch * 0.5, 40.0)

        for images, _ in train_loader:
            images = images.to(device)
            batch_size = images.size(0)
            msgs = torch.randint(0, 2, (batch_size, 16)).float().to(device)

            # --- 1. TREINO DO DISCRIMINADOR ---
            opt_disc.zero_grad()
            with torch.no_grad():
                imgs_wmk = encoder(images, msgs)

            d_real = discriminator(images)
            d_fake = discriminator(imgs_wmk)

            # Usamos WithLogits porque removemos o Sigmoid da rede
            loss_d = F.binary_cross_entropy_with_logits(
                d_real, torch.ones_like(d_real)
            ) + F.binary_cross_entropy_with_logits(d_fake, torch.zeros_like(d_fake))
            loss_d.backward()
            opt_disc.step()

            # --- 2. TREINO ENCODER + DECODER (O Gerador) ---
            opt_enc_dec.zero_grad()
            imgs_wmk = encoder(images, msgs)
            imgs_noisy = noise_layer(imgs_wmk)
            bits_pred = decoder(imgs_noisy)

            # Perda de Qualidade de Imagem (Invisibilidade)
            l_img = F.mse_loss(imgs_wmk, images)

            # Perda de Mensagem (Robustez)
            l_msg = F.binary_cross_entropy_with_logits(bits_pred, msgs)

            # Perda Adversária (Enganar o Discriminador)
            l_adv = F.binary_cross_entropy_with_logits(
                discriminator(imgs_wmk), torch.ones_like(d_real)
            )

            # EQUAÇÃO FINAL (Ajuste de Pesos Profissional)
            # l_img tem peso alto para garantir que a imagem não mude
            loss_g = (l_img * 10.0) + (l_msg * w_msg) + (l_adv * 0.1)

            loss_g.backward()
            opt_enc_dec.step()

            # Métricas
            preds = (torch.sigmoid(bits_pred) > 0.5).float()
            total_acc += (preds == msgs).float().mean().item()
            total_loss_img += l_img.item()
            total_loss_adv += l_adv.item()

        avg_acc = (total_acc / len(train_loader)) * 100
        history["acc"].append(avg_acc)
        history["loss_img"].append(total_loss_img / len(train_loader))
        history["w_msg"].append(w_msg)

        scheduler.step(avg_acc)

        print(
            f"Epoch [{epoch + 1}/{epochs}] Acc: {avg_acc:.2f}% | L_Img: {history['loss_img'][-1]:.4f}"
        )

        if avg_acc > best_acc:
            best_acc = avg_acc
            torch.save(encoder.state_dict(), "checkpoints/encoder_best.pth")
            torch.save(decoder.state_dict(), "checkpoints/decoder_best.pth")

    return history


def save_training_curves(history):
    epochs = range(1, len(history["acc"]) + 1)
    fig, ax1 = plt.subplots(figsize=(12, 7))

    # Eixo Esquerdo: Acurácia (0 a 100%)
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Acurácia (%)", color="blue", fontsize=12)
    ax1.plot(epochs, history["acc"], color="blue", label="Accuracy", linewidth=2)
    ax1.set_ylim([0, 105])  # Fixa o topo em 100% para melhor perspectiva
    ax1.tick_params(axis="y", labelcolor="blue")
    ax1.grid(True, linestyle="--", alpha=0.5)

    # Eixo Direito: Losses e Pesos
    ax2 = ax1.twinx()
    ax2.set_ylabel("Valor das Perdas / Pesos", color="black", fontsize=12)

    # Plotando L_img, L_adv e Peso Msg
    ax2.plot(
        epochs, history["loss_img"], color="red", label="L_img (MSE)", linestyle="-"
    )
    ax2.plot(
        epochs,
        history.get("loss_adv", []),
        color="orange",
        label="L_adv (Adversarial)",
        linestyle="--",
    )
    ax2.plot(
        epochs, history["w_msg"], color="green", label="Peso Msg (w_msg)", linestyle=":"
    )

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
    # Em 2026, priorizamos 'cuda', 'xpu' (Intel) ou 'mps' (Apple)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch, "xpu") and torch.xpu.is_available():
        device = torch.device("xpu")
    else:
        device = torch.device("cpu")

    transform = transforms.Compose(
        [
            transforms.Resize((64, 64)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
    )

    path_treino = "data/treino"
    train_loader = None

    # Tenta carregar o dataset real
    if os.path.exists(path_treino) and len(os.listdir(path_treino)) > 0:
        try:
            dataset = datasets.ImageFolder(root=path_treino, transform=transform)
            # Garantir que o dataset não está vazio
            if len(dataset) > 0:
                train_loader = DataLoader(
                    dataset, batch_size=16, shuffle=True, drop_last=True
                )
                print(f"✅ Dataset real carregado com {len(dataset)} imagens.")
        except Exception as e:
            print(f"⚠️ Erro ao ler subpastas: {e}")

    # Fallback: Se o dataset real falhar ou não existir
    if train_loader is None:
        print("⚠️ Usando dados simulados (TensorDataset).")
        # Criamos imagens (float32) e labels (long) para simular o ImageFolder
        dummy_images = torch.randn(100, 3, 64, 64)
        dummy_labels = torch.zeros(100).long()
        dataset = TensorDataset(dummy_images, dummy_labels)
        train_loader = DataLoader(dataset, batch_size=16, shuffle=True)

    # Inicialização dos Modelos Atualizados (parâmetros de 2026)
    # Certifique-se de que os argumentos batem com as classes novas
    encoder = Encoder(msg_length=16, hidden_size=64)
    decoder = Decoder(msg_length=16, hidden_size=128)
    discriminator = Discriminator(hidden_size=64)

    print(f"🚀 Iniciando treinamento em: {device}")

    # Executa o treino
    history = train_professional_system(
        encoder, decoder, discriminator, train_loader, epochs=100
    )

    # Salva os resultados
    save_training_curves(history)

    print("\n💾 Salvando modelos finais...")
    os.makedirs("checkpoints", exist_ok=True)
    # Salva também os pesos do Discriminador (útil para retomar treino)
    torch.save(encoder.state_dict(), "checkpoints/encoder_final.pth")
    torch.save(decoder.state_dict(), "checkpoints/decoder_final.pth")
    torch.save(discriminator.state_dict(), "checkpoints/discriminator_final.pth")
    print("✅ Processo concluído com sucesso!")


if __name__ == "__main__":
    main()
