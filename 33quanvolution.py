import torchquantum as tq
import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
import argparse
import medmnist
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from medmnist import BreastMNIST
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR


class QuanvolutionFilter(tq.QuantumModule):
    def __init__(self):
        super().__init__()
        # Changed to 9 wires for 3x3 patch
        self.n_wires = 9

        # Encoder for 3x3 patches (9 pixels -> 9 wires)
        # Using Ry gates to encode 9 input pixel values (normalized to [0, pi] implicitly)
        self.encoder = tq.GeneralEncoder(
            [{"input_idx": [i], "func": "ry", "wires": [i]} for i in range(self.n_wires)]
        )

        # Trainable random quantum layer for feature extraction
        self.q_layer = tq.RandomLayer(n_ops=16, wires=list(range(self.n_wires)))
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x, use_qiskit=False):
        # Input x shape: (bsz, 1, 28, 28)
        bsz = x.shape[0]
        # Initialize a quantum device for the current batch
        qdev = tq.QuantumDevice(self.n_wires, bsz=bsz, device=x.device)
        size = 28
        x = x.view(bsz, size, size)

        data_list = []

        # Quanvolution operation: 3x3 window, stride 2
        for c in range(0, size - 2, 2):
            for r in range(0, size - 2, 2):
                # Extract 3x3 patch (bsz, 3, 3)
                patch = x[:, c : c + 3, r : r + 3]

                # Flatten patch to (bsz, 9) for encoding
                data = patch.reshape(bsz, 9)

                if use_qiskit:
                    # Qiskit processing path (if specified)
                    data = self.qiskit_processor.process_parameterized(
                        qdev, self.encoder, self.q_layer, self.measure, data
                    )
                else:
                    # TorchQuantum simulation path
                    self.encoder(qdev, data)
                    self.q_layer(qdev)
                    # Measure the expectation values of Pauli Z on all 9 qubits
                    data = self.measure(qdev)

                data_list.append(data.view(bsz, 9))

        # Recombine processed patches (output spatial dim is 13x13)
        # Result shape: (bsz, 13*13*9)
        result = torch.cat(data_list, dim=1).float()

        return result


class HybridModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.qf = QuanvolutionFilter()
        self.linear = torch.nn.Linear(9 * 13 * 13, 2)

    def forward(self, x, use_qiskit=False):
        with torch.no_grad():
            x = self.qf(x, use_qiskit)
        x = self.linear(x)
        return F.log_softmax(x, -1)


class HybridModel_without_qf(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Classical benchmark: 28*28 input pixels -> 2 classes
        self.linear = torch.nn.Linear(28 * 28, 2)

    def forward(self, x, use_qiskit=False):
        x = x.view(-1, 28 * 28)
        x = self.linear(x)
        return F.log_softmax(x, -1)


def train(dataloader, model, device, optimizer):
    model.train()
    for inputs, targets in dataloader:
        inputs = inputs.to(device)
        # BreastMNIST targets are [batch_size, 1], need squeeze for nll_loss
        targets = targets.squeeze().long().to(device)

        outputs = model(inputs)
        loss = F.nll_loss(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"loss: {loss.item():.4f}", end="\r")


def valid_test(dataloader, split, model, device, qiskit=False, plot_cm=False, output_dir=None):
    model.eval()
    target_all = []
    output_all = []
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.squeeze().long().to(device)

            outputs = model(inputs, use_qiskit=qiskit)

            target_all.append(targets)
            output_all.append(outputs)

        target_all = torch.cat(target_all, dim=0)
        output_all = torch.cat(output_all, dim=0)

    _, indices = output_all.topk(1, dim=1)
    masks = indices.eq(target_all.view(-1, 1).expand_as(indices))
    size = target_all.shape[0]
    corrects = masks.sum().item()
    accuracy = corrects / size
    loss = F.nll_loss(output_all, target_all).item()

    print(f"{split} set accuracy: {accuracy:.4f}")
    print(f"{split} set loss: {loss:.4f}")

    if plot_cm and output_dir:
        preds = indices.view(-1).cpu().numpy()
        targets_np = target_all.view(-1).cpu().numpy()

        # BreastMNIST Labels: 0 = Malignant, 1 = Benign/Normal
        class_names = ["malignant", "benign/normal"]

        cm = confusion_matrix(targets_np, preds)

        # Plot using sklearn's built-in display which uses matplotlib
        fig, ax = plt.subplots(figsize=(6, 5))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)

        # cmap='Blues' matches the provided reference image
        disp.plot(cmap='Blues', ax=ax, colorbar=False)

        title = f"Confusion Matrix - 9 Qubits Hybrid Model"
        if not isinstance(model, HybridModel):
            title = f"Confusion Matrix - Classical Benchmark"
        elif qiskit:
            title += " (Qiskit)"

        plt.title(title)

        # Use the provided output_dir for safe saving, regardless of CWD
        filename = output_dir / f"confusion_matrix_3x3quanvolution.png"

        plt.savefig(filename)
        print(f"Confusion matrix plot saved to {filename.name}")
        plt.close(fig)

    return accuracy, loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--epochs", type=int, default=15, help="number of training epochs"
    )
    parser.add_argument(
        "--qiskit-simulation", action="store_true", help="run on real quantum computer/simulator"
    )
    args = parser.parse_args()

    # SCRIPT_DIR is already defined globally by the path fix block
    SCRIPT_DIR = PROJECT_ROOT

    train_model_without_qf = True
    n_epochs = args.epochs

    # Reproducibility
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    # Data Transforms
    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    # Load BreastMNIST
    data_flag = 'breastmnist'
    info = medmnist.INFO[data_flag]
    n_channels = info['n_channels']
    n_classes = len(info['label']) # Should be 2

    print(f"Loading {data_flag}...")
    print(f"Channels: {n_channels}, Classes: {n_classes}")

    # Note on medmnist: download=True will typically save data to ~/.medmnist,
    # which is independent of the script's execution directory.
    train_dataset = BreastMNIST(split='train', transform=data_transform, download=True)
    val_dataset = BreastMNIST(split='val', transform=data_transform, download=True)
    test_dataset = BreastMNIST(split='test', transform=data_transform, download=True)

    # DataLoaders
    batch_size = 16

    loaders = {
        'train': DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2),
        'val': DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2),
        'test': DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    }

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"Using device: {device}")

    # Initialize Models
    model = HybridModel().to(device)
    model_without_qf = HybridModel_without_qf().to(device)

    # --- Hybrid Model Training ---
    optimizer = optim.Adam(model.parameters(), lr=5e-3, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=n_epochs)

    print("--- Training Hybrid Quantum-Classical Model (9 Qubits) ---")
    accu_list1 = []
    loss_list1 = []

    for epoch in range(1, n_epochs + 1):
        print(f"Epoch {epoch}:")
        train(loaders['train'], model, device, optimizer)
        print(f"LR: {optimizer.param_groups[0]['lr']:.6f}")

        is_last = (epoch == n_epochs)
        # Pass SCRIPT_DIR to ensure output files are saved correctly in the root
        accu, loss = valid_test(loaders['test'], "test", model, device, plot_cm=is_last, output_dir=SCRIPT_DIR)
        accu_list1.append(accu)
        loss_list1.append(loss)
        scheduler.step()

    # --- Classical Benchmark Training ---
    if train_model_without_qf:
        print("\n--- Training Classical Model (Benchmark) ---")
        optimizer = optim.Adam(
            model_without_qf.parameters(), lr=5e-3, weight_decay=1e-4
        )
        scheduler = CosineAnnealingLR(optimizer, T_max=n_epochs)

        accu_list2 = []
        loss_list2 = []

        for epoch in range(1, n_epochs + 1):
            print(f"Epoch {epoch}:")
            train(loaders['train'], model_without_qf, device, optimizer)
            print(f"LR: {optimizer.param_groups[0]['lr']:.6f}")

            is_last = (epoch == n_epochs)
            # Pass SCRIPT_DIR to ensure output files are saved correctly in the root
            accu, loss = valid_test(loaders['test'], "test", model_without_qf, device, plot_cm=is_last, output_dir=SCRIPT_DIR)
            accu_list2.append(accu)
            loss_list2.append(loss)
            scheduler.step()

    if args.qiskit_simulation:
        try:
            from qiskit import IBMQ
            from torchquantum.plugin import QiskitProcessor

            print(f"\nTest with Qiskit Simulator")
            processor_simulation = QiskitProcessor(use_real_qc=False)
            model.qf.set_qiskit_processor(processor_simulation)
            valid_test(loaders['test'], "test", model, device, qiskit=True, plot_cm=True, output_dir=SCRIPT_DIR)

        except ImportError:
            print("Qiskit not installed or configured correctly.")
        except Exception as e:
            print(f"An error occurred during Qiskit simulation: {e}")

if __name__ == "__main__":
    main()