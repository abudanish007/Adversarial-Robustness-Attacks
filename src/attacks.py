import torch
import torch.nn as nn
from typing import Optional


class AdversarialAttackEngine:
    """
    Gradient-based adversarial attack engine for PyTorch models.
    Implements Fast Gradient Sign Method (FGSM) and Projected Gradient Descent (PGD).
    """

    def __init__(self, model: nn.Module, device: Optional[torch.device] = None) -> None:
        self.model = model
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.loss_fn = nn.CrossEntropyLoss()

    def fgsm(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
        epsilon: float = 8 / 255,
        clamp_min: float = 0.0,
        clamp_max: float = 1.0,
    ) -> torch.Tensor:
        """
        Generates single-step adversarial examples:
        x_adv = clamp(x + eps * sign(grad_x(L(theta, x, y))))
        """
        x_adv = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)
        x_adv.requires_grad = True

        self.model.eval()
        outputs = self.model(x_adv)
        loss = self.loss_fn(outputs, labels)

        self.model.zero_grad()
        loss.backward()

        if x_adv.grad is None:
            raise RuntimeError("Gradient computation failed during FGSM backward pass.")

        perturbed = x_adv + epsilon * x_adv.grad.sign()
        return torch.clamp(perturbed, clamp_min, clamp_max).detach()

    def pgd(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
        epsilon: float = 8 / 255,
        alpha: float = 2 / 255,
        num_steps: int = 10,
        random_start: bool = True,
        clamp_min: float = 0.0,
        clamp_max: float = 1.0,
    ) -> torch.Tensor:
        """
        Generates multi-step Projected Gradient Descent adversarial examples (L_inf norm).
        """
        x_orig = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)
        x_adv = x_orig.clone().detach()

        if random_start:
            noise = torch.FloatTensor(*x_orig.shape).uniform_(-epsilon, epsilon).to(self.device)
            x_adv = torch.clamp(x_adv + noise, clamp_min, clamp_max)

        self.model.eval()

        for _ in range(num_steps):
            x_adv.requires_grad = True
            outputs = self.model(x_adv)
            loss = self.loss_fn(outputs, labels)

            self.model.zero_grad()
            loss.backward()

            if x_adv.grad is None:
                raise RuntimeError("Gradient computation failed during PGD step.")

            with torch.no_grad():
                x_adv = x_adv + alpha * x_adv.grad.sign()
                eta = torch.clamp(x_adv - x_orig, min=-epsilon, max=epsilon)
                x_adv = torch.clamp(x_orig + eta, min=clamp_min, max=clamp_max)

        return x_adv.detach()
