import torch
import torchvision.models as models
from src.attacks import AdversarialAttackEngine


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Executing on device: {device}")

    # Load a lightweight pre-trained vision model
    weights = models.MobileNet_V2_Weights.DEFAULT
    model = models.mobilenet_v2(weights=weights).to(device)
    model.eval()

    # Synthetic image input batch: (batch_size=1, channels=3, height=224, width=224)
    inputs = torch.rand(1, 3, 224, 224, device=device)

    with torch.no_grad():
        logits = model(inputs)
        clean_pred = logits.argmax(dim=1).item()
        clean_conf = torch.softmax(logits, dim=1)[0, clean_pred].item()

    print(f"[+] Clean Prediction:   Class {clean_pred} (Confidence: {clean_conf:.4f})")

    # Use clean prediction as ground truth label to craft adversarial perturbation
    labels = torch.tensor([clean_pred], device=device)
    engine = AdversarialAttackEngine(model=model, device=device)

    # 1. Evaluate single-step FGSM
    adv_fgsm = engine.fgsm(inputs, labels, epsilon=16 / 255)
    with torch.no_grad():
        fgsm_logits = model(adv_fgsm)
        fgsm_pred = fgsm_logits.argmax(dim=1).item()
        fgsm_conf = torch.softmax(fgsm_logits, dim=1)[0, fgsm_pred].item()
    print(f"[!] FGSM Attack Result: Class {fgsm_pred} (Confidence: {fgsm_conf:.4f})")

    # 2. Evaluate iterative PGD (10 steps)
    adv_pgd = engine.pgd(inputs, labels, epsilon=16 / 255, alpha=2 / 255, num_steps=10)
    with torch.no_grad():
        pgd_logits = model(adv_pgd)
        pgd_pred = pgd_logits.argmax(dim=1).item()
        pgd_conf = torch.softmax(pgd_logits, dim=1)[0, pgd_pred].item()
    print(f"[!] PGD-10 Attack Result: Class {pgd_pred} (Confidence: {pgd_conf:.4f})")


if __name__ == "__main__":
    main()
