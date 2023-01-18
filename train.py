from __future__ import absolute_import, division, print_function

from trainer import Trainer
from options import MonodepthOptions
import os

import torch
import numpy as np
import random

#####Seed setting for reproductivility#########
random_seed = 0
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(random_seed)
random.seed(random_seed)
###############################################


options = MonodepthOptions()
opts = options.parse()


if __name__ == "__main__":

    trainer = Trainer(opts)
    trainer.train()
