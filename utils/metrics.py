import torch
import torch.nn.functional as F

def evaluate_performance(encoder, decoder, test_loader, device):
    psnr_total = 0
    accuracy_total = 0
    
    with torch.no_grad():
        for images, _ in test_loader:
            images = images.to(device)
            # Para avaliação, use uma mensagem fixa ou randômica para consistência
            msgs = torch.randint(0_000, 2, (images.size(0), 16)).float().to(device) # Mensagem de 16 bits
            
            marked = encoder(images, msgs)
            recovered = decoder(marked)
            
            # Cálculo de PSNR
            mse = F.mse_loss(marked, images)
            psnr = 10 * torch.log10(1 / mse)
            
            # Cálculo de Bit Accuracy
            pred_bits = (torch.sigmoid(recovered) > 0.5).float()
            acc = (pred_bits == msgs).float().mean()
            
            psnr_total += psnr.item()
            accuracy_total += acc.item()
            
    print(f"Resultado Final: PSNR médio: {psnr_total/len(test_loader):.2f}dB | Acc Bits: {accuracy_total/len(test_loader)*100:.2f}%")