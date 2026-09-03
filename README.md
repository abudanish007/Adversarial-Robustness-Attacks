# Adversarial Robustness Engine

A modular PyTorch implementation of gradient-based adversarial attack methods for deep neural network vulnerability assessment.

## Features
- **Fast Gradient Sign Method (FGSM)**: Single-step $L_\infty$-bounded perturbation generation.
- **Projected Gradient Descent (PGD)**: Iterative $L_\infty$ first-order adversary with uniform random initialization.
- **Vectorized & Device Agnostic**: Native support for CPU, CUDA, and Apple Silicon (MPS).

## Mathematical Formulation
The optimization problem maximizes cross-entropy loss within an $L_\infty$ ball of radius $\epsilon$:

$$\max_{\|\delta\|_\infty \le \epsilon} \mathcal{L}(\theta, x + \delta, y)$$
