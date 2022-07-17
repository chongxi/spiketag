from .bram_thres import threshold
from .bram_thres import offset
from .bram_thres import channel_hash 
from .memory_api import *
from .configFPGA import FPGA
from .run import run


import torch

def load_param(filename):
    return torch.load(filename)

