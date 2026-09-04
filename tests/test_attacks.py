import torch
import torch.nn as nn
from src.attacks import AdversarialAttackEngine


class SimpleConvNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(16, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def test_fgsm_perturbation_bound() -> None:
    model = SimpleConvNet()
    engine = AdversarialAttackEngine(model=model, device=torch.device("cpu"))

    batch_size = 4
    inputs = torch.rand(batch_size, 3, 32, 32)
    labels = torch.randint(0, 10, (batch_size,))
    eps = 0.03

    adv_inputs = engine.fgsm(inputs, labels, epsilon=eps)

    assert adv_inputs.shape == inputs.shape
    linf_diff = (adv_inputs - inputs).abs().max().item()
    assert linf_diff <= eps + 1e-6
    assert adv_inputs.min() >= 0.0 and adv_inputs.max() <= 1.0


def test_pgd_perturbation_bound() -> None:
    model = SimpleConvNet()
    engine = AdversarialAttackEngine(model=model, device=torch.device("cpu"))

    batch_size = 4
    inputs = torch.rand(batch_size, 3, 32, 32)
    labels = torch.randint(0, 10, (batch_size,))
    eps = 0.03

    adv_inputs = engine.pgd(inputs, labels, epsilon=eps, alpha=0.01, num_steps=5)

    assert adv_inputs.shape == inputs.shape
    linf_diff = (adv_inputs - inputs).abs().max().item()
    assert linf_diff <= eps + 1e-6
    assert adv_inputs.min() >= 0.0 and adv_inputs.max() <= 1.0


if __name__ == "__main__":
    test_fgsm_perturbation_bound()
    test_pgd_perturbation_bound()
    print("All adversarial attack tests passed.")
