
import numpy as np
import medmnist
from medmnist import INFO

# Load official BloodMNIST partitions
info = INFO["bloodmnist"]
BloodMNIST = getattr(medmnist, info["python_class"])

train = BloodMNIST(split="train", download=True)
val = BloodMNIST(split="val", download=True)

# Access image arrays and labels
X_train = train.imgs
X_val = val.imgs

y_train = train.labels.flatten()
y_val = val.labels.flatten()

# Normalize pixel values to [0, 1]
X_train = X_train.astype(np.float32) / 255.0
X_val = X_val.astype(np.float32) / 255.0

# Flatten images for logistic regression and MLP
X_train_flat = X_train.reshape(len(X_train), -1)
X_val_flat = X_val.reshape(len(X_val), -1)

# Verify shapes and normalized values
print("Training image shape:", X_train.shape)
print("Validation image shape:", X_val.shape)

print("Flattened training shape:", X_train_flat.shape)
print("Flattened validation shape:", X_val_flat.shape)

print("Training labels shape:", y_train.shape)
print("Validation labels shape:", y_val.shape)

print("Minimum pixel:", X_train.min())
print("Maximum pixel:", X_train.max())
