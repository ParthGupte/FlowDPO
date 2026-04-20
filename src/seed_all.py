import torch
import random
import numpy as np

def set_seed(seed=42):
    torch.manual_seed(seed)
    # This is useful when using CUDA
    torch.cuda.manual_seed_all(seed)
    # This is useful when using numpy
    np.random.seed(seed)
    # This is useful for python's random module
    random.seed(seed)

# Call the function at the beginning of your script