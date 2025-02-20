import os
import random
import numpy as np
import torch

from .generatorX import Generator as GeneratorX
from .generatorX import Loader as LoaderX
from .generatorU import Generator as GeneratorU
from .generatorU import Loader as LoaderU

def set_seed(seed=2021):
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False






