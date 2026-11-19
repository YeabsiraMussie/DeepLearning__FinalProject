
import numpy as np
import matplotlib.pyplot as plt
import medmnist
from medmnist import INFO

# Load dataset information
info = INFO["bloodmnist"]
BloodMNIST = getattr(medmnist, info["python_class"])

# Load training data only
train = BloodMNIST(split="train", download=True)

# Get training labels
labels = train.labels.flatten()

# Count images in each class
counts = np.bincount(labels, minlength=8)

# Get class names
class_names = [
    info["label"][str(i)]
    for i in range(8)
]

# Shorten long names for plotting
class_names[3] = "Immature granulocytes"

# Print distribution
print("TRAINING CLASS DISTRIBUTION")
print("-" * 40)

for name, count in zip(class_names, counts):
    percentage = count / len(train) * 100
    print(f"{name}: {count} ({percentage:.2f}%)")

# Plot distribution
plt.figure(figsize=(11, 5))

plt.bar(class_names, counts)

plt.title("BloodMNIST Training Class Distribution")
plt.xlabel("Blood Cell Class")
plt.ylabel("Number of Images")

plt.xticks(rotation=35, ha="right")
plt.tight_layout()
plt.show()
