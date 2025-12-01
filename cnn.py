import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from PIL import Image
from tqdm import tqdm
import pennylane as qml


DATA_PATH = "data/train/images/"
MASK_PATH = "data/train/masks/"
IMG_SIZE = 64
BATCH_SIZE = 6
EPOCHS = 12
LR = 0.0005

def load_image(path):
    img = Image.open(path).convert("L")
    img = img.resize((IMG_SIZE, IMG_SIZE))
    img = np.array(img) / 255.0
    return img.astype(np.float32)

class TGSDataset(Dataset):
    def __init__(self, files):
        self.files = files

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        name = self.files[idx]
        image = load_image(os.path.join(DATA_PATH, name))
        mask  = load_image(os.path.join(MASK_PATH, name))
        mask = (mask > 0.5).astype(np.float32)
        return torch.tensor(image).unsqueeze(0), torch.tensor(mask).unsqueeze(0)

files = [f for f in os.listdir(DATA_PATH) if f.endswith(".png")]
train_files, val_files = train_test_split(files, test_size=0.15, random_state=42)
train_loader = DataLoader(TGSDataset(train_files), batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(TGSDataset(val_files),   batch_size=BATCH_SIZE)

n_qubits = 8
dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev, interface="torch")
def q_block(inputs, weights):
    # normalize inputs to [-pi, pi]
    scaled = (inputs - torch.mean(inputs)) / (torch.std(inputs) + 1e-6)
    scaled = scaled * np.pi

    for i in range(n_qubits):
        qml.RY(scaled[i], wires=i)

    qml.templates.layers.BasicEntanglerLayers(weights, wires=range(n_qubits))
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

class QuantumLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(1, n_qubits) * 0.15)

    def forward(self, x):
        outputs = []
        for vec in x:
            out = q_block(vec, self.weights)
            outputs.append(torch.stack(out).float())
        return torch.stack(outputs)


class HybridQuantumCNN(nn.Module):
    def __init__(self):
        super().__init__()

        # stronger encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten()
        )

        self.fc_enc = nn.Linear(8192, n_qubits)

        # Quantum bottleneck
        self.q_layer = QuantumLayer()

        # decode
        self.fc_dec = nn.Linear(n_qubits, 8192)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 2, stride=2), nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 2, stride=2),
            nn.Sigmoid()
        )

    def forward(self, x):
        z = self.encoder(x)
        z = self.fc_enc(z)
        z = self.q_layer(z)
        d = self.fc_dec(z).reshape(-1, 32, 16, 16)
        return self.decoder(d)

def dice_coef(pred, target, eps=1e-8):
    pred = (pred > 0.5).float()
    inter = (pred * target).sum()
    union = pred.sum() + target.sum()
    return (2 * inter + eps) / (union + eps)

def iou(pred, target, eps=1e-8):
    pred = (pred > 0.5).float()
    inter = (pred * target).sum()
    union = pred.sum() + target.sum() - inter
    return (inter + eps) / (union + eps)

def accuracy(pred, target):
    pred = (pred > 0.5).float()
    correct = (pred == target).float().sum()
    total = torch.numel(pred)
    return correct / total


class BCEDiceLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCELoss()

    def forward(self, pred, target):
        bce = self.bce(pred, target)
        dice = 1 - dice_coef(pred, target)
        return bce + dice

model = HybridQuantumCNN()
criterion = BCEDiceLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

print("\nTraining Hybrid Quantum CNN...\n")

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    print(f"Epoch {epoch+1}/{EPOCHS}")
    train_bar = tqdm(train_loader, desc="Training", leave=False)

    for imgs, masks in train_bar:
        optimizer.zero_grad()
        preds = model(imgs)
        loss = criterion(preds, masks)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        train_bar.set_postfix({"loss": f"{loss.item():.4f}"})

    model.eval()
    dices, ious, accs = [], [], []

    with torch.no_grad():
        for imgs, masks in val_loader:
            preds = model(imgs)
            dices.append(dice_coef(preds, masks).item())
            ious.append(iou(preds, masks).item())
            accs.append(accuracy(preds, masks).item())

    print(f"Loss={total_loss:.3f} | Dice={np.mean(dices):.4f} | IoU={np.mean(ious):.4f} | Acc={np.mean(accs):.4f}\n")

print("Training complete!")
