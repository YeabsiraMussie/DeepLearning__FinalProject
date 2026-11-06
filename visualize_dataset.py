
import matplotlib.pyplot as plt
import medmnist
from medmnist import INFO

# Load information about BloodMNIST
info = INFO["bloodmnist"]
BloodMNIST = getattr(medmnist, info["python_class"])

# Load the training dataset
train = BloodMNIST(split="train", download=True)

# Create a grid for 8 images
fig, axes = plt.subplots(2, 4, figsize=(12, 6))

# Find and show one example from each class
for class_id in range(8):
    for image, label in train:
        if int(label.item()) == class_id:
            ax = axes[class_id // 4, class_id % 4]

            ax.imshow(image)
            ax.set_title(info["label"][str(class_id)],
                         fontsize=9)
            ax.axis("off")
            break

plt.tight_layout()
plt.show()
