
import time
import numpy as np
import medmnist

from medmnist import INFO
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report
)

# 1. Load the dataset
info = INFO["bloodmnist"]
BloodMNIST = getattr(medmnist, info["python_class"])

train = BloodMNIST(split="train", download=True)
val = BloodMNIST(split="val", download=True)

# 2. Prepare images and labels
X_train = train.imgs.astype(np.float32) / 255.0
X_val = val.imgs.astype(np.float32) / 255.0

y_train = train.labels.flatten()
y_val = val.labels.flatten()

# 3. Flatten the images
X_train = X_train.reshape(len(X_train), -1)
X_val = X_val.reshape(len(X_val), -1)

# 4. Define the model
model = LogisticRegression(
    max_iter=3000,
    solver="lbfgs",
    random_state=42
)

# 5. Train the model
print("Training logistic regression...")

start = time.time()
model.fit(X_train, y_train)
training_time = time.time() - start

# 6. Predict on the validation set
y_pred = model.predict(X_val)

# 7. Calculate performance metrics
accuracy = accuracy_score(y_val, y_pred)
macro_f1 = f1_score(
    y_val, y_pred, average="macro"
)

print("\nRESULTS")
print("-" * 40)
print(f"Training time: {training_time:.2f} seconds")
print(f"Validation accuracy: {accuracy:.4f}")
print(f"Validation macro-F1: {macro_f1:.4f}")

# 8. Display class-wise metrics
class_names = [
    info["label"][str(i)] for i in range(8)
]

print("\nClassification Report:")
print(classification_report(
    y_val,
    y_pred,
    labels=list(range(8)),
    target_names=class_names,
    zero_division=0
))
