import numpy as np
from scipy import signal
from numba import njit

def gkern2d(kernlen=21, std=2):
    """Returns a 2D Gaussian kernel array."""
    gkern1d = signal.gaussian(kernlen, std=std).reshape(kernlen, 1)
    _gkern2d = np.outer(gkern1d, gkern1d)
    _gkern2d /= _gkern2d.sum()
    return _gkern2d

def gkern3d(kernlen=21, std=3):
    gkern1d = signal.gaussian(kernlen, std=std).reshape(kernlen, 1)
    _gkern3d = np.outer(np.outer(gkern1d, gkern1d), gkern1d)
    _gkern3d = _gkern3d.reshape((kernlen, kernlen, kernlen))
    _gkern3d /= _gkern3d.sum()
    return _gkern3d


@njit(cache=True)
def firing_pos_from_scv(scv, pos, neuron_id, valid_bin):
    firing_pos, t_bin = [], 0
    for count in scv[neuron_id]:
        count = int(count)
        if count!=0 and valid_bin[0]<t_bin<valid_bin[1]: # compute when in bin range and has spike count
            for i in range(count):
                firing_pos.append(pos[t_bin])
        t_bin += 1
    _firing_pos_array = np.zeros((len(firing_pos), 2))
    for i in range(len(firing_pos)):
        _firing_pos_array[i] = firing_pos[i]
    return _firing_pos_array
