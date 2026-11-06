
import medmnist
from medmnist import INFO

# Get information about BloodMNIST
info = INFO["bloodmnist"]

print("Dataset:", info["description"])
print("Classes:", info["label"])
print("Number of classes:", len(info["label"]))

# Load the dataset
BloodMNIST = getattr(medmnist, info["python_class"])

train = BloodMNIST(split="train", download=True)
val = BloodMNIST(split="val", download=True)
test = BloodMNIST(split="test", download=True)

print("\nTraining images:", len(train))
print("Validation images:", len(val))
print("Test images:", len(test))

image, label = train[0]

print("\nFirst image size:", image.size)
print("First image label:", label)
