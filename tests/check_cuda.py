# Checks if CUDA is available and prints the version of PyTorch and NumPy

import torch
import numpy as np

print("PyTorch version:", torch.__version__)
print("NumPy version:", np.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("CUDA device:", torch.cuda.get_device_name(0))
    print("CUDA version:", torch.version.cuda)