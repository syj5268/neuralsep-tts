import random
import torch
import numpy as np


def set_seed(seed: int,
             use_cuda: bool):
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    torch.manual_seed(seed)
    if use_cuda:
        torch.cuda.manual_seed(seed)
