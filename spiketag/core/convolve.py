from scipy import signal
import torch
from torch.nn.functional import conv1d
import numpy as np

def scipy_conv1d(sig, win, scale=1):
    filtered = signal.convolve(sig, win, mode='full') / sum(win) / scale
    return filtered

def torch_conv1d(sig, win, scale=1):
    torch_sig = torch.from_numpy(sig).cuda().float()/scale
    torch_win = torch.from_numpy(win).cuda().float()
    torch_filtered = conv1d(torch_sig.view(1, 1, -1), torch_win.view(1,1,-1), padding=win.shape[0]//2)
    torch_filtered = torch_filtered.view(-1)/sum(win)
    return torch_filtered.cpu().numpy()

def convolve(sig, win, scale=1, device='gpu', mode='same'):
    if device == 'cpu':
        y = scipy_conv1d(sig, win, scale)
    elif device == 'gpu':
        y = torch_conv1d(sig, win, scale)
    if mode == 'full':
        return y
    elif mode == 'same':
        length = sig.shape[0]
        return y[:length]
