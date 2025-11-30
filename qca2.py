import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import medmnist
from medmnist import INFO

from qiskit import QuantumCircuit
from qiskit.circuit.library import ZFeatureMap, RealAmplitudes
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.connectors import TorchConnector
from qiskit.primitives import StatevectorEstimator

# --- Configuration ---
DATA_FLAG = 'breastmnist'
BATCH_SIZE = 16 
NUM_QUBITS = 8  # Expanded to 8 Qubits
EPOCHS = 5      
LEARNING_RATE = 0.001
TRAIN_SUBSET_SIZE = 150 # Reduced slightly to offset higher simulation cost of 8 qubits
TEST_SUBSET_SIZE = 50

def load_data():
    """Load MedMNIST data as PyTorch tensors."""
    print(f"Loading {DATA_FLAG}...")
    info = INFO[DATA_FLAG]
    DataClass = getattr(medmnist, info['python_class'])

    # Transformation: Convert to Tensor and Normalize
    # MedMNIST images are uint8 [0, 255], ToTensor scales to [0.0, 1.0]
    from torchvision import transforms
    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[.5], std=[.5]) # Normalize to [-1, 1]
    ])

    train_dataset = DataClass(split='train', transform=data_transform, download=True)
    test_dataset = DataClass(split='test', transform=data_transform, download=True)

    # Subset for simulation performance
    train_dataset = Subset(train_dataset, range(TRAIN_SUBSET_SIZE))
    test_dataset = Subset(test_dataset, range(TEST_SUBSET_SIZE))

    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    return train_loader, test_loader

def create_qnn(num_qubits):
    """
    Creates a Qiskit QNN to be inserted into the PyTorch model.
    Structure: ZFeatureMap (Data Encoding) -> RealAmplitudes (Trainable Ansatz)
    """
    # 1. Feature Map (encodes the 8 classical features into quantum state)
    feature_map = ZFeatureMap(feature_dimension=num_qubits, reps=1)
    
    # 2. Ansatz (Variational circuit)
    # With 8 qubits, we stick to 1 rep to keep parameter count manageable (16 params)
    ansatz = RealAmplitudes(num_qubits, reps=1, entanglement='linear')
    
    # 3. Compose Circuit
    qc = QuantumCircuit(num_qubits)
    qc.compose(feature_map, inplace=True)
    qc.compose(ansatz, inplace=True)
    
    # 4. Define Observables
    # Measure the Z expectation of EACH of the 8 qubits individually
    observables = [
        SparsePauliOp.from_sparse_list([("Z", [i], 1.0)], num_qubits)
        for i in range(num_qubits)
    ]
    
    # 5. Define QNN
    qnn = EstimatorQNN(
        circuit=qc,
        observables=observables,
        input_params=feature_map.parameters,
        weight_params=ansatz.parameters,
        estimator=StatevectorEstimator()
    )
    return qnn

class HybridModel(nn.Module):
    def __init__(self, qnn):
        super(HybridModel, self).__init__()
        
        # --- Classical Pre-processing (CNN) ---
        # Input: 1 x 28 x 28
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5) # -> 6 x 24 x 24
        self.pool = nn.MaxPool2d(2, 2)              # -> 6 x 12 x 12
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)# -> 16 x 8 x 8
        # Pool again -> 16 x 4 x 4 = 256 elements
        
        # Reduce classical features to match number of qubits
        # 256 features -> 8 features
        self.fc_reduce = nn.Linear(16 * 4 * 4, NUM_QUBITS)
        
        # --- Quantum Layer ---
        # TorchConnector allows Qiskit QNN to act as a PyTorch layer
        self.qnn = TorchConnector(qnn)
        
        # --- Classical Post-processing (FC) ---
        # Takes the 8 quantum measurements and maps to class output
        self.fc_out = nn.Linear(NUM_QUBITS, 1)

    def forward(self, x):
        # Classical CNN Path
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # Flatten for linear layer
        
        # Reduction to "Latent Quantum Space"
        x = self.fc_reduce(x)
        # Using tanh to keep inputs within [-1, 1] range for the Angle Encoding (ZFeatureMap)
        x = torch.tanh(x) * np.pi 
        
        # Quantum Path
        x = self.qnn(x) # Returns expectation values [-1, 1]
        
        # Final Classification
        x = self.fc_out(x)
        return torch.sigmoid(x) # Output probability (0 to 1)

# --- Training Loop ---
if __name__ == "__main__":
    train_loader, test_loader = load_data()
    
    # Initialize Hybrid Model
    qnn = create_qnn(NUM_QUBITS)
    model = HybridModel(qnn)
    
    # Print Architecture
    print("\n--- Hybrid Model Architecture ---")
    print(model)
    print("\n--- Quantum Circuit (8 Qubits) ---")
    print(qnn.circuit.draw(output='text'))
    print("-------------------------------\n")
    
    # Optimizer and Loss
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.BCELoss() # Binary Cross Entropy
    
    loss_list = []
    
    print(f"Starting hybrid training for {EPOCHS} epochs...")
    model.train()
    
    for epoch in range(EPOCHS):
        total_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            
            # Forward pass
            output = model(data)
            
            # Calculate loss (Target needs to be float for BCELoss)
            loss = criterion(output, target.float())
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            print(f"\rEpoch {epoch+1} | Batch {batch_idx+1}/{len(train_loader)} | Loss: {loss.item():.4f}", end="")
            
        avg_loss = total_loss / len(train_loader)
        loss_list.append(avg_loss)
        print(f"\nEpoch {epoch+1} Average Loss: {avg_loss:.4f}")

    # --- Evaluation ---
    print("\nEvaluating...")
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            predicted = (output > 0.5).float()
            total += target.size(0)
            correct += (predicted == target.float()).sum().item()
            
    acc = 100 * correct / total
    print(f"Test Accuracy: {acc:.2f}%")

    # Save Plot
    plt.figure(figsize=(8, 6))
    plt.plot(loss_list, marker='o')
    plt.title("Hybrid CNN-QNN Training Loss (8 Qubits)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid()
    plt.savefig("hybrid_training_loss.png")
    print("Plot saved to 'hybrid_training_loss.png'")