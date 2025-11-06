import random

import numpy
import torch
from pytorch_lightning import seed_everything


def seed_all(seed = 0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    numpy.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    seed_everything(seed)