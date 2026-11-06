
import numpy as np
import sklearn
import torch
import medmnist
import matplotlib

print("NumPy:", np.__version__)
print("Scikit-learn:", sklearn.__version__)
print("PyTorch:", torch.__version__)
print("MedMNIST:", medmnist.__version__)
print("CUDA available:", torch.cuda.is_available())

print("SETUP SUCCESSFUL!")
