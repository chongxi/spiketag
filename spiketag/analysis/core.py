import numpy as np
import torch
from scipy import signal
from numba import njit, prange
from numba.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning
import warnings

warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)



def spk_time_to_scv(spk_time, ts, delta_t=250e-3, sublist=None):
    if sublist is None:
        spk_time_list=list(spk_time.values())
    else:
        spk_time_list = [spk_time.get(key) for key in sublist]
    suv = scv_from_spk_time_list(spk_time_list, ts, delta_t)
    return suv

@njit(cache=True, parallel=True, fastmath=True)
def scv_from_spk_time_list(spk_time_list, ts, delta_t=250e-3):
    N = len(spk_time_list)
    T = ts.shape[0]
    suv = np.zeros((N,T))
    for j in prange(T):    
        for i in prange(N):
            suv[i, j] = np.sum(np.logical_and(spk_time_list[i] >  ts[j]-delta_t, 
                                              spk_time_list[i] <= ts[j]))
    return suv


@njit(cache=True)
def licomb_Matrix(w, X):
    '''
    input:
    1. `w` a weight vector (could be n single units vector(suv) )
    2. `X` which is a 3D matrix (could be n firing rate map)
    
    output:
    the linear combination w[0]*X[0] + w[1]*X[1] + ... + w[n]*X[N]
    X[0], X[1] ... X[N] are all 2D matrix

    ### another way of doing it is einsum (einstein summation)
    # x.shape: torch.Size([100, 800, 800])
    # y.shape: torch.Size([100, 1])
    # einsum works better than licomb_Matrix
    # z = torch.einsum('ijk, il -> jk', (x, y))
    # z = x (100,800,800) weighted by y (100,1) 
    '''
    Y = np.dot(w, X.reshape(w.shape[0], -1)).reshape(X.shape[1], X.shape[2])
    return Y


@njit(cache=True, parallel=True, fastmath=True)
def bayesian_decoding(Fr, suv, delta_t=100e-3):
    '''
    fast version of below labmda expression
    log_posterior = lambda x,y,i: np.nansum(np.log(Fr[:,x,y]) * suv[:,i]) - 100e-3*Fr[:,x,y].sum(axis=0)

    return both true (x,y) and log_posterior map

    Usage:
    1. for all suv
    true_xy, post_2d = bayesian_decoding(Fr=Fr, suv=suv[:, :], delta_t=100e-3)

    2. for specific suv
    true_xy, post_2d = bayesian_decoding(Fr=Fr, suv=suv[:, 100:101], delta_t=100e-3)
    '''

    true_xy = np.zeros((suv.shape[1],2))
    post_xy = np.zeros((suv.shape[1],2))
    post_2d = np.zeros((suv.shape[1], Fr.shape[1], Fr.shape[2]))
    possion_matrix = delta_t*Fr.sum(axis=0)
    log_fr = np.log(Fr) # make sure Fr[Fr==0] = 1e-12
    for i in prange(suv.shape[1]): # i is time point
        suv_weighted_log_fr = licomb_Matrix(suv[:,i].ravel(), log_fr)
        post_2d[i] = np.exp(suv_weighted_log_fr - possion_matrix)
    return post_2d


def argmax_2d_tensor(X):
    if X.ndim<3:
        X = X[np.newaxis, :]
    values, indices = torch.max(torch.from_numpy(X.reshape(X.shape[0],-1)), 1)
    post_xy = np.vstack((indices.numpy()%X.shape[2], indices.numpy()//X.shape[2])).T
    return np.squeeze(post_xy)

def smooth(x, window_len=60):
    '''
    moving weighted average
    '''
    tau = 0.0005
    y = np.empty_like(x)
    # box = np.exp(tau*np.arange(window_len))
    box = np.ones((window_len,))
    box = box/float(box.sum())
    for i in range(y.shape[1]):
        y[:,i] = np.convolve(x[:,i], box, mode='same')
    return y



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


def interp_nan(data):
    '''
    use this to fix any bad position segment (use `np.apply_along_axis` trick)
    np.apply_along_axis(interp_nan, 0, pos_seg)
    '''
    bad_indexes = np.isnan(data)
    good_indexes = np.logical_not(bad_indexes)
    good_data = data[good_indexes]
    interpolated = np.interp(bad_indexes.nonzero()[0], good_indexes.nonzero()[0], good_data)
    data[bad_indexes] = interpolated
    return data