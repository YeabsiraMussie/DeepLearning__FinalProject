
import copy
import random
import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import TensorDataset, DataLoader
from medmnist import BloodMNIST, INFO
from sklearn.metrics import accuracy_score, f1_score, classification_report

# 1. Reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# 2. Load official dataset partitions
train = BloodMNIST(split="train", download=True)
val = BloodMNIST(split="val", download=True)

# 3. Normalize and flatten images
X_train = train.imgs.astype(np.float32) / 255.0
X_val = val.imgs.astype(np.float32) / 255.0

X_train = X_train.reshape(-1, 2352)
X_val = X_val.reshape(-1, 2352)

y_train = train.labels.flatten().astype(np.int64)
y_val = val.labels.flatten().astype(np.int64)

# 4. Convert to PyTorch tensors
train_data = TensorDataset(
    torch.from_numpy(X_train),
    torch.from_numpy(y_train)
)

val_data = TensorDataset(
    torch.from_numpy(X_val),
    torch.from_numpy(y_val)
)

train_loader = DataLoader(
    train_data, batch_size=128, shuffle=True
)

val_loader = DataLoader(
    val_data, batch_size=128, shuffle=False
)

# 5. Define the MLP architecture
class MLP(nn.Module):
    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(2352, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 8)
        )

    def forward(self, x):
        return self.network(x)

model = MLP()

# 6. Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(
    model.parameters(), lr=0.001
)

# 7. Training settings
epochs = 15
best_f1 = -1.0
best_weights = None
patience = 5
bad_epochs = 0

# 8. Training loop
for epoch in range(epochs):
    model.train()
    total_loss = 0.0

    for images, labels in train_loader:
        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(labels)

    train_loss = total_loss / len(train_data)

    # Validation
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in val_loader:
            outputs = model(images)
            preds = outputs.argmax(dim=1)

            all_preds.extend(preds.numpy())
            all_labels.extend(labels.numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(
        all_labels, all_preds,
        average="macro", zero_division=0
    )

    print(
        f"Epoch {epoch+1:02d} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Val Accuracy: {accuracy:.4f} | "
        f"Val Macro-F1: {macro_f1:.4f}"
    )

    # Save best validation model
    if macro_f1 > best_f1:
        best_f1 = macro_f1
        best_weights = copy.deepcopy(model.state_dict())
        bad_epochs = 0
    else:
        bad_epochs += 1

    if bad_epochs >= patience:
        print("Early stopping triggered.")
        break

# 9. Restore and save best model
model.load_state_dict(best_weights)
torch.save(model.state_dict(), "best_mlp.pth")

# 10. Final validation report
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in val_loader:
        outputs = model(images)
        preds = outputs.argmax(dim=1)

        all_preds.extend(preds.numpy())
        all_labels.extend(labels.numpy())

class_names = [
    INFO["bloodmnist"]["label"][str(i)]
    for i in range(8)
]

print("\nBEST MLP RESULTS")
print("Validation Accuracy:",
      accuracy_score(all_labels, all_preds))
print("Validation Macro-F1:",
      f1_score(all_labels, all_preds,
               average="macro", zero_division=0))

print("\nClassification Report:")
print(classification_report(
    all_labels, all_preds,
    labels=list(range(8)),
    target_names=class_names,
    zero_division=0
))
