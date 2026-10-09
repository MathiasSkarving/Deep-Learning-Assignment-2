import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import optim
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
# !pip install torchvision
import torchvision
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms
# !pip install torchmetrics
import torchmetrics
from medmnist import BloodMNIST
from sklearn.metrics import ConfusionMatrixDisplay

from cnn import CNN, CNNLeNet5;

batch_size = 60

transform = transforms.Compose([
    transforms.ToTensor(),
])

trainDataset = BloodMNIST(split="train", transform=transform, download=True, size=28)
valDataset   = BloodMNIST(split="val",   transform=transform, download=True, size=28)
testDataset  = BloodMNIST(split="test",  transform=transform, download=True, size=28)

train_loader = DataLoader(dataset=trainDataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=testDataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(dataset=valDataset, batch_size=batch_size, shuffle=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

model = CNN(in_channels=3, num_classes=8).to(device)

# Define the loss function
loss_function = nn.CrossEntropyLoss()

# Define the optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)
finetimizer = optim.Adam(model.parameters(), lr=0.0001)
train_acc = torchmetrics.Accuracy(task="multiclass", num_classes=8).to(device)

learning_rates = [0.1, 0.01, 0.001, 0.0001, 0.00001]


num_epochs=20
for epoch in range(num_epochs):
    # Iterate over training batches
    print(f"Epoch [{epoch + 1}/{num_epochs}]")
    train_acc.reset()
    running_loss = 0.0
       
    for batch_index, (data, targets) in enumerate(tqdm(train_loader)):
        data = data.to(device)
        targets = targets.to(device)
        scores = model(data)
        targets = targets.squeeze(1).long()
        loss = loss_function(scores, targets)
        if(epoch > 15):
            finetimizer.zero_grad()
            loss.backward()
            finetimizer.step()
        else:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        running_loss += loss.item() * data.size(0)
        train_acc.update(scores.argmax(dim=1), targets)
    
    epoch_loss = running_loss / len(trainDataset)
    print(f"Epoch [{epoch + 1}/{num_epochs}] " f"loss: {epoch_loss:.4f}  train acc: {train_acc.compute():.4f}")

       
acc       = torchmetrics.Accuracy(task="multiclass", num_classes=8).to(device)
precision = torchmetrics.Precision(task="multiclass", num_classes=8, average="macro").to(device)
recall    = torchmetrics.Recall(task="multiclass", num_classes=8, average="macro").to(device)
model.eval()

all_preds, all_labels = [], []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.squeeze(1).long().to(device)
        # Get predicted probabilities for test data batch
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        acc(preds, labels)
        precision(preds, labels)
        recall(preds, labels)
        all_preds.append(preds.cpu())
        all_labels.append(labels.cpu())

print(f"Test accuracy:  {acc.compute():.4f}")
print(f"Test precision: {precision.compute():.4f}")
print(f"Test recall:    {recall.compute():.4f}")

class_names = ["basophil", "eosinophil", "erythroblast", "immature gran.",
               "lymphocyte", "monocyte", "neutrophil", "platelet"]

y_pred = torch.cat(all_preds).numpy()
y_true = torch.cat(all_labels).numpy()

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

ConfusionMatrixDisplay.from_predictions(
    y_true, y_pred, display_labels=class_names,
    cmap="Blues", xticks_rotation=45, ax=axes[0],
)
axes[0].set_title("Counts")

ConfusionMatrixDisplay.from_predictions(
    y_true, y_pred, display_labels=class_names,
    normalize="true", values_format=".2f",
    cmap="Blues", xticks_rotation=45, ax=axes[1],
)
axes[1].set_title("Normalized (per true class)")


plt.tight_layout()
plt.show()

from matplotlib.colors import PowerNorm

fig, ax = plt.subplots(figsize=(9, 8))

disp = ConfusionMatrixDisplay.from_predictions(
    y_true, y_pred, display_labels=class_names,
    cmap="Blues", xticks_rotation=45, ax=ax,
)

cm = disp.confusion_matrix
norm = PowerNorm(gamma=0.3, vmin=0, vmax=cm.max())
disp.im_.set_norm(norm)

# recolor the numbers so they stay readable under the new scale
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        disp.text_[i, j].set_color("white" if norm(cm[i, j]) > 0.6 else "black")

ax.set_title("Confusion matrix (power scale, errors emphasized)")
plt.tight_layout()
plt.show()

import matplotlib.pyplot as plt
import torch

# Assuming 'model' is your trained CNNLeNet5 instance
# 1. Extract weights, clone them to avoid messing with the actual model, and convert to NumPy
filters = model.conv1.weight.data.clone().cpu().numpy()

# 2. Normalize the weights to a [0, 1] range so Matplotlib can render them properly
# Without this, the negative weights might not render or will clip
filters = (filters - filters.min()) / (filters.max() - filters.min())

# 3. Set up a Matplotlib figure to plot all 6 filters side-by-side
fig, axes = plt.subplots(1, 6, figsize=(15, 3))

for i in range(6):
    # Select the i-th output filter, and the 0-th input channel (since BloodMNIST is grayscale/1-channel)
    kernel = filters[i, 0, :, :]
    
    ax = axes[i]
    ax.imshow(kernel)
    ax.axis('off')
    ax.set_title(f'Filter {i+1}')

plt.tight_layout()
plt.show()