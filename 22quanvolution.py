"""
MIT License

Copyright (c) 2020-present TorchQuantum Authors (Original Code)
Modifications for MedMNIST BreastDataset (2025)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

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
        self.n_wires = 4
        # A generic encoder for 2x2 patches (4 pixels -> 4 wires)
        self.encoder = tq.GeneralEncoder(
            [
                {"input_idx": [0], "func": "ry", "wires": [0]},
                {"input_idx": [1], "func": "ry", "wires": [1]},
                {"input_idx": [2], "func": "ry", "wires": [2]},
                {"input_idx": [3], "func": "ry", "wires": [3]},
            ]
        )

        self.q_layer = tq.RandomLayer(n_ops=8, wires=list(range(self.n_wires)))
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x, use_qiskit=False):
        # Input x shape: (bsz, 1, 28, 28)
        bsz = x.shape[0]
        qdev = tq.QuantumDevice(self.n_wires, bsz=bsz, device=x.device)
        size = 28
        x = x.view(bsz, size, size)

        data_list = []

        # Iterate over the image with a stride of 2
        for c in range(0, size, 2):
            for r in range(0, size, 2):
                # Extract 2x2 patch
                data = torch.transpose(
                    torch.cat(
                        (x[:, c, r], x[:, c, r + 1], x[:, c + 1, r], x[:, c + 1, r + 1])
                    ).view(4, bsz),
                    0,
                    1,
                )
                if use_qiskit:
                    data = self.qiskit_processor.process_parameterized(
                        qdev, self.encoder, self.q_layer, self.measure, data
                    )
                else:
                    self.encoder(qdev, data)
                    self.q_layer(qdev)
                    data = self.measure(qdev)

                data_list.append(data.view(bsz, 4))

        # Recombine processed patches
        # Output shape logic: (28/2)*(28/2) patches * 4 channels = 14*14*4 features
        result = torch.cat(data_list, dim=1).float()

        return result


class HybridModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.qf = QuanvolutionFilter()
        # Input features: 4 channels * 14 * 14 spatial dim
        # Output features: 2 (BreastMNIST is binary: Malignant/Benign)
        self.linear = torch.nn.Linear(4 * 14 * 14, 2)

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


def valid_test(dataloader, split, model, device, qiskit=False, plot_cm=False):
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

    if plot_cm:
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
        
        title = f"Confusion Matrix - 4 qubits"
        if qiskit:
            title += " (Qiskit)"
        plt.title(title)
        
        # Save and show
        filename = f"confusion_matrix_2x2quanvolution.png"
        plt.savefig(filename)
        print(f"Confusion matrix plot saved to {filename}")
        plt.show()

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
    # Note: BreastMNIST images are 28x28 grayscale
    data_flag = 'breastmnist'
    info = medmnist.INFO[data_flag]
    n_channels = info['n_channels']
    n_classes = len(info['label']) # Should be 2

    print(f"Loading {data_flag}...")
    print(f"Channels: {n_channels}, Classes: {n_classes}")

    train_dataset = BreastMNIST(split='train', transform=data_transform, download=True)
    val_dataset = BreastMNIST(split='val', transform=data_transform, download=True)
    test_dataset = BreastMNIST(split='test', transform=data_transform, download=True)

    # DataLoaders
    # Reduced batch size slightly as quantum simulation can be memory intensive if parallelized
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
    
    optimizer = optim.Adam(model.parameters(), lr=5e-3, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=n_epochs)

    print("--- Training Hybrid Quantum-Classical Model ---")
    accu_list1 = []
    loss_list1 = []

    for epoch in range(1, n_epochs + 1):
        print(f"Epoch {epoch}:")
        train(loaders['train'], model, device, optimizer)
        print(f"LR: {optimizer.param_groups[0]['lr']:.6f}")

        # Validate on test set (or val set)
        # Plot CM only on last epoch
        is_last = (epoch == n_epochs)
        accu, loss = valid_test(loaders['test'], "test", model, device, plot_cm=is_last)
        accu_list1.append(accu)
        loss_list1.append(loss)
        scheduler.step()

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
            accu, loss = valid_test(loaders['test'], "test", model_without_qf, device, plot_cm=is_last)
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
            valid_test(loaders['test'], "test", model, device, qiskit=True, plot_cm=True)

            # Uncomment to run on REAL QC
            # backend_name = "ibmq_quito"
            # print(f"\nTest on Real Quantum Computer {backend_name}")
            # processor_real_qc = QiskitProcessor(use_real_qc=True, backend_name=backend_name)
            # model.qf.set_qiskit_processor(processor_real_qc)
            # valid_test(loaders['test'], "test", model, device, qiskit=True, plot_cm=True)

        except ImportError:
            print("Qiskit not installed or configured correctly.")
        except Exception as e:
            print(f"An error occurred during Qiskit simulation: {e}")

if __name__ == "__main__":
    main()