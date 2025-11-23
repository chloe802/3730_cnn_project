import os
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F



def load_dataset(base_path="train", size=(64, 64), limit=400):
    images, masks = [], []
    img_dir = os.path.join(base_path, "images")
    mask_dir = os.path.join(base_path, "masks")
    img_files = os.listdir(img_dir)[:limit]

    for fname in tqdm(img_files, desc="Loading dataset"):
        img = np.array(Image.open(os.path.join(img_dir, fname))
                       .convert("L").resize(size)) / 255.0

        mask = np.array(Image.open(os.path.join(mask_dir, fname))
                        .convert("L").resize(size)) / 255.0

        if mask.sum() > 0:
            images.append(img)
            masks.append(mask)

    print(f"Loaded {len(images)} valid samples of size {images[0].shape}")
    return np.array(images), np.array(masks)



class DoubleConv(nn.Module):
    """Conv → ReLU → Conv → ReLU"""
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_c, out_c, 3, padding=1),
            nn.ReLU()
        )

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.down1 = DoubleConv(1, 16)
        self.down2 = DoubleConv(16, 32)
        self.down3 = DoubleConv(32, 64)

        self.pool = nn.MaxPool2d(2)

        self.mid = DoubleConv(64, 128)

        self.up3 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv3 = DoubleConv(128, 64)

        self.up2 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.conv2 = DoubleConv(64, 32)

        self.up1 = nn.ConvTranspose2d(32, 16, 2, stride=2)
        self.conv1 = DoubleConv(32, 16)

        self.output_layer = nn.Conv2d(16, 1, 1)

    def forward(self, x):
        x1 = self.down1(x)
        x2 = self.down2(self.pool(x1))
        x3 = self.down3(self.pool(x2))

        xm = self.mid(self.pool(x3))

        x = self.up3(xm)
        x = self.conv3(torch.cat([x, x3], dim=1))

        x = self.up2(x)
        x = self.conv2(torch.cat([x, x2], dim=1))

        x = self.up1(x)
        x = self.conv1(torch.cat([x, x1], dim=1))

        return torch.sigmoid(self.output_layer(x))




def dice_loss(pred, target, smooth=1e-5):
    pred = pred.contiguous()
    target = target.contiguous()

    intersection = (pred * target).sum()
    return 1 - (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)


def bce_dice_loss(pred, target):
    return F.binary_cross_entropy(pred, target) + dice_loss(pred, target)


# =====================================================
#                     TRAINING LOOP
# =====================================================

def train_unet(model, images, masks, epochs=10, lr=0.001):
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        losses = []

        for img, mask in zip(images, masks):
            img_t = torch.tensor(img).unsqueeze(0).unsqueeze(0).float()   # [B,C,H,W]
            mask_t = torch.tensor(mask).unsqueeze(0).unsqueeze(0).float()

            optimizer.zero_grad()

            preds = model(img_t)
            loss = bce_dice_loss(preds, mask_t)

            loss.backward()
            optimizer.step()

            losses.append(loss.item())

        print(f"Epoch {epoch+1}/{epochs} | Loss = {np.mean(losses):.4f}")
        visualize_prediction(model, images[0], masks[0], epoch+1)


def visualize_prediction(model, image, mask, epoch):
    img_t = torch.tensor(image).unsqueeze(0).unsqueeze(0).float()
    pred = model(img_t).detach().cpu().numpy()[0,0]

    plt.figure(figsize=(9, 3))

    plt.subplot(1, 3, 1)
    plt.imshow(image, cmap="gray")
    plt.title("Input")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(mask, cmap="gray")
    plt.title("True Mask")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(pred, cmap="inferno")
    plt.title(f"Predicted (Epoch {epoch})")
    plt.axis("off")

    plt.tight_layout()
    plt.savefig(f"unet_pred_epoch_{epoch}.png", dpi=200)
    plt.close()
    print(f"Saved unet_pred_epoch_{epoch}.png")



if __name__ == "__main__":
    images, masks = load_dataset("data/train", size=(64, 64), limit=400)

    model = UNet()
    train_unet(model, images, masks, epochs=10, lr=0.001)
