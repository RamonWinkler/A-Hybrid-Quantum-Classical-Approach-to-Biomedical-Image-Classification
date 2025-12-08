import argparse
import random
import sys
from pathlib import Path
from typing import Tuple, Dict, List, Optional

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.optim.lr_scheduler import CosineAnnealingLR

import torchquantum as tq
import medmnist
from medmnist import BreastMNIST
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# --- Configuration ---
SEED = 42
BATCH_SIZE = 16
LEARNING_RATE = 5e-3
WEIGHT_DECAY = 1e-4
# BreastMNIST: 28x28 grayscale images
IMG_SIZE = 28
KERNEL_SIZE = 3
STRIDE = 2


def set_reproducibility(seed: int = SEED):
    """Sets seeds for reproducibility across numpy, random, and torch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class QuanvolutionFilter(tq.QuantumModule):
    """
    Quantum convolution layer.

    Mechanism:
    1. Extracts 3x3 patches from the input image.
    2. Encodes pixel data into quantum states using rotation gates.
    3. Applies a random quantum circuit.
    4. Measures expectation values to generate feature maps.
    """

    def __init__(self, n_wires: int = 9):
        super().__init__()
        self.n_wires = n_wires

        # Encoder: Maps classical pixel values to quantum state parameters.
        # We use Ry gates. Given pixel value x_i, the state becomes:
        # |psi> = Ry(x_i)|0>
        self.encoder = tq.GeneralEncoder(
            [{"input_idx": [i], "func": "ry", "wires": [i]} for i in range(self.n_wires)]
        )

        # Ansatz: Random quantum layer for feature transformation
        self.q_layer = tq.RandomLayer(n_ops=16, wires=list(range(self.n_wires)))

        # Measurement: Expectation value of Pauli-Z on all qubits
        # Result \in [-1, 1]
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x: torch.Tensor, use_qiskit: bool = False) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (bsz, 1, 28, 28)
            use_qiskit: Boolean to toggle Qiskit backend simulation.
        """
        bsz = x.shape[0]
        # Initialize quantum device
        qdev = tq.QuantumDevice(self.n_wires, bsz=bsz, device=x.device)

        # Remove channel dim for processing: (bsz, 28, 28)
        x = x.view(bsz, IMG_SIZE, IMG_SIZE)

        data_list = []

        # Sliding window operation (naive implementation)
        # Note: Ideally, use torch.nn.Unfold for vectorization, but
        # tq.QuantumDevice batching requires care with unfolded dimensions.
        for c in range(0, IMG_SIZE - KERNEL_SIZE + 1, STRIDE):
            for r in range(0, IMG_SIZE - KERNEL_SIZE + 1, STRIDE):
                # Extract 3x3 patch
                patch = x[:, c: c + KERNEL_SIZE, r: r + KERNEL_SIZE]

                # Flatten patch to (bsz, 9) for quantum encoding
                data = patch.reshape(bsz, self.n_wires)

                if use_qiskit:
                    # External backend processing
                    data = self.qiskit_processor.process_parameterized(
                        qdev, self.encoder, self.q_layer, self.measure, data
                    )
                else:
                    # TorchQuantum internal simulation
                    self.encoder(qdev, data)
                    self.q_layer(qdev)
                    data = self.measure(qdev)

                data_list.append(data.view(bsz, self.n_wires))

        # Recombine features.
        # Output spatial dim: 13x13 (for 28x28 in, 3x3 kernel, stride 2)
        # Result shape: (bsz, 13*13*9)
        result = torch.cat(data_list, dim=1).float()

        return result


class HybridModel(nn.Module):
    """Hybrid Quantum-Classical Neural Network."""

    def __init__(self):
        super().__init__()
        self.qf = QuanvolutionFilter()
        # Input features: 9 channels * 13 * 13 spatial dims
        self.linear = nn.Linear(9 * 13 * 13, 2)

    def forward(self, x: torch.Tensor, use_qiskit: bool = False) -> torch.Tensor:
        # Quantum filter layer (frozen or trainable depending on config)
        # Generally in Quanvolution, the quantum layer is fixed or trainable.
        # Here we do not detach, allowing gradients if q_layer parameters allow.
        # However, the user script had `with torch.no_grad():` inside forward
        # suggesting the quantum layer is used purely as a fixed feature extractor.
        with torch.no_grad():
            x = self.qf(x, use_qiskit)

        x = self.linear(x)
        return F.log_softmax(x, -1)


class ClassicalBenchmark(nn.Module):
    """Standard classical MLP for baseline comparison."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(IMG_SIZE * IMG_SIZE, 2)

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        x = x.view(-1, IMG_SIZE * IMG_SIZE)
        x = self.linear(x)
        return F.log_softmax(x, -1)


def get_dataloaders(batch_size: int) -> Dict[str, DataLoader]:
    """Downloads BreastMNIST and returns dataloaders."""
    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    loaders = {}
    for split in ['train', 'val', 'test']:
        dataset = BreastMNIST(split=split, transform=data_transform, download=True)
        loaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == 'train'),
            num_workers=2
        )
    return loaders


def train_one_epoch(
        dataloader: DataLoader,
        model: nn.Module,
        device: torch.device,
        optimizer: optim.Optimizer
) -> None:
    """Performs one epoch of training."""
    model.train()
    for inputs, targets in dataloader:
        inputs = inputs.to(device)
        targets = targets.squeeze().long().to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = F.nll_loss(outputs, targets)
        loss.backward()
        optimizer.step()
        print(f"Batch Loss: {loss.item():.4f}", end="\r")


def evaluate(
        dataloader: DataLoader,
        split: str,
        model: nn.Module,
        device: torch.device,
        qiskit: bool = False,
        plot_cm: bool = False,
        output_dir: Path = Path("./")
) -> Tuple[float, float]:
    """
    Evaluates the model and optionally plots a confusion matrix.
    Returns (accuracy, average_loss).
    """
    model.eval()
    target_all = []
    output_all = []

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.squeeze().long().to(device)

            # Pass use_qiskit flag only if model supports it (HybridModel)
            if isinstance(model, HybridModel):
                outputs = model(inputs, use_qiskit=qiskit)
            else:
                outputs = model(inputs)

            target_all.append(targets)
            output_all.append(outputs)

        target_all = torch.cat(target_all, dim=0)
        output_all = torch.cat(output_all, dim=0)

    # Metrics
    loss = F.nll_loss(output_all, target_all).item()
    _, indices = output_all.topk(1, dim=1)
    corrects = indices.eq(target_all.view(-1, 1).expand_as(indices)).sum().item()
    accuracy = corrects / target_all.size(0)

    print(f"\n{split.capitalize()} set - Accuracy: {accuracy:.4f} | Loss: {loss:.4f}")

    if plot_cm:
        _plot_confusion_matrix(target_all, indices, model, qiskit, output_dir)

    return accuracy, loss


def _plot_confusion_matrix(targets, preds, model, qiskit, output_dir):
    """Helper function to plot and save confusion matrix."""
    preds_np = preds.view(-1).cpu().numpy()
    targets_np = targets.view(-1).cpu().numpy()
    class_names = ["malignant", "benign/normal"]

    cm = confusion_matrix(targets_np, preds_np)

    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap='Blues', ax=ax, colorbar=False)

    title = "Confusion Matrix - "
    if isinstance(model, HybridModel):
        title += "9 Qubits Hybrid"
        if qiskit:
            title += " (Qiskit)"
    else:
        title += "Classical Benchmark"

    plt.title(title)
    filename = output_dir / f"cm_{title.lower().replace(' ', '_').replace('(', '').replace(')', '')}.png"
    plt.savefig(filename)
    print(f"Saved plot to {filename}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Quantum vs Classical BreastMNIST Classification")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--qiskit", action="store_true", help="Run validation on Qiskit backend")
    args = parser.parse_args()

    set_reproducibility(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Data
    loaders = get_dataloaders(BATCH_SIZE)

    # Models to train: (Model Instance, Model Name)
    experiments = [
        (HybridModel().to(device), "Hybrid Quantum"),
        (ClassicalBenchmark().to(device), "Classical Benchmark")
    ]

    for model, name in experiments:
        print(f"\n{'=' * 20}\nTraining {name}\n{'=' * 20}")

        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

        for epoch in range(1, args.epochs + 1):
            print(f"Epoch {epoch}/{args.epochs} [LR: {optimizer.param_groups[0]['lr']:.6f}]")
            train_one_epoch(loaders['train'], model, device, optimizer)

            # Run validation
            is_last = (epoch == args.epochs)
            evaluate(loaders['test'], "test", model, device, plot_cm=is_last)
            scheduler.step()

    # Optional Qiskit Simulation for the Hybrid Model
    if args.qiskit:
        print("\n--- Running Qiskit Simulation ---")
        try:
            from torchquantum.plugin import QiskitProcessor
            # Retrieve the hybrid model (index 0 in experiments list)
            hybrid_model = experiments[0][0]

            processor = QiskitProcessor(use_real_qc=False)
            hybrid_model.qf.set_qiskit_processor(processor)

            evaluate(loaders['test'], "test", hybrid_model, device, qiskit=True, plot_cm=True)
        except ImportError:
            print("Error: Qiskit or QiskitProcessor not available.")
        except Exception as e:
            print(f"Qiskit simulation failed: {e}")


if __name__ == "__main__":
    main()