import numpy as np
import torch
from scipy import signal
from numba import njit, prange
from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)


def softmax(X):
    X = torch.tensor(X).double()
    X_exp = torch.exp(X-X.max())
    partition = torch.sum(X_exp, dim=1, keepdim=True)
    _softmax = X_exp / partition # The broadcast mechanism is applied here
    return _softmax.numpy()


def spike_time_from_fet(fet, fs=25000.):
    spike_timing = [ fet[fet[:,-1]==i][:,0]/fs for i in np.unique(fet[:,-1]) ]
    return spike_timing

def firing_rate_from_fet(fet, fs=25000., binsize=50e-3):
    win = signal.blackman(250)  # 10ms smoothing window
    win /= np.sum(win)
    spike_timing   = spike_time_from_fet(fet, fs)   # a spike time list
    N_neuron       = len(spike_timing)
    t_start,t_end  = fet[0][0]/fs, fet[-1][0]/fs
    bins           = np.arange(t_start, t_end, binsize)
    B_bins         = len(bins) - 1
    spike_count    = np.zeros((B_bins, N_neuron))
    spike_rate     = np.zeros((B_bins, N_neuron))
    for i in range(N_neuron):
        spike_count[:, i], _ = np.histogram(spike_timing[i], bins)
        spike_rate[:, i] = np.convolve(spike_count[:, i]/binsize, win, 'same')
        
    return bins[:-1], spike_rate


@njit(cache=True, parallel=True, fastmath=True)
def _spike_binning(spike_time, event_time, spike_id, windows=np.array([-0.5, 0.5])):
    '''
    bin(count) the `spike_time` in an array of `window` that around `event_time`.
    related issue: https://github.com/chongxi/spiketag/issues/57
    --------
    Parameters
    ----------
    spike_time: a numpy array (T,) contains time stamps of spikes
    event_time: a numpy array (B,) contains time stamps of binning position
    spike_id:   a numpy array (T,) contains spike ids with (N,) unique labels
    window:    a numpy array (2,) contains the binning window 
    
    Returns
    -------
    B is #events(#bins), N is #cells, W is #window
    count: a numpy array representing the binning result (W, B, N)
    '''
    
    B = event_time.shape[0]
    cell_ids = np.unique(spike_id)
    N = np.unique(cell_ids).shape[0]
    spike_id -= cell_ids.min()
    W = windows.shape[0]
    count = np.zeros((W, B, N))  # (#Win, #Bin, #Cell)
    
    for k in prange(N):
        _spike_time = spike_time[spike_id==k]
#         print(_spike_time.shape)
        for w in range(W):
            window = windows[w]
            idx_start = np.searchsorted(_spike_time, event_time+window[0], side='right')
            for i in range(event_time.shape[0]):
                nspk = 0
                while True:
                    idx = int(idx_start[i] + nspk)
                    if idx<_spike_time.shape[0] and event_time[i]+window[0] <= _spike_time[idx] and _spike_time[idx] < event_time[i]+window[1]:
                        nspk+=1      
                    else:
                        break
                count[w, i, k] = nspk
    return count


def spike_binning(spike_time, event_time, windows=np.array([[-0.5, 0.5]]), spike_id=None):
    '''
    bin(count) the `spike_time` in `window` that around `event_time`

    work with: no, single or multiple `spike_id`
    work with: single or multiple `window` in `windows`

    Can be used to calculate PSTH, CCG and population decoding 

    related issue: https://github.com/chongxi/spiketag/issues/57
    --------
    Parameters
    ----------
    spike_time: a numpy array (T,) contains time stamps of spikes
    event_time: a numpy array (B,) contains time stamps of binning position
    spike_id:   a numpy array (T,) contains spike ids with (N,) unique labels
    window:    a numpy array (2,) contains the binning window 
    
    Returns
    -------
    B is #events(#bins), N is #cells, W is #window
    count: a numpy array representing the binning result (W, B, N)
    '''
    
    windows = np.array(windows)
    if spike_id is None:
        spike_id = np.zeros_like(spike_time)
    if windows.ndim==1:
        windows = windows.reshape(1, -1)
    return np.squeeze(_spike_binning(spike_time, event_time, spike_id, windows))



def spk_time_to_scv(spk_time_dict, ts, t_window=250e-3, sublist=None):
    if sublist is None:
        spk_time_list = list(spk_time_dict.values())
    else:
        spk_time_list = [spk_time_dict.get(key) for key in sublist]
    suv = scv_from_spk_time_list(spk_time_list, ts, t_window)
    return suv


@njit(cache=True, parallel=True, fastmath=True)
def scv_from_spk_time_list(spk_time_list, ts, t_window=250e-3):
    '''
    extract spike count vector from a list of spike trains
    '''
    N = len(spk_time_list)
    T = ts.shape[0]
    suv = np.zeros((T, N))
    for i in prange(T):
        for j in prange(N):
            suv[i, j] = np.sum(np.logical_and(spk_time_list[j] >= ts[i] - t_window,
                                              spk_time_list[j] < ts[i]))
    return suv[1:]


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
def bayesian_decoding(Fr, suv, t_window=100e-3):
    '''
    fast version of below labmda expression
    log_posterior = lambda x,y,i: np.nansum(np.log(Fr[:,x,y]) * suv[:,i]) - 100e-3*Fr[:,x,y].sum(axis=0)

    return both true (x,y) and log_posterior map

    Usage:
    post_2d = bayesian_decoding(Fr=Fr, suv=suv, t_window=100e-3)
    '''

    post_2d = np.zeros((suv.shape[0], Fr.shape[1], Fr.shape[2]))
    possion_matrix = t_window*Fr.sum(axis=0)
    log_fr = np.log(Fr) # make sure Fr[Fr==0] = 1e-12
    for i in prange(suv.shape[0]): # i is time point
        suv_weighted_log_fr = licomb_Matrix(suv[i].ravel(), log_fr)
        post_2d[i] = np.exp(suv_weighted_log_fr - possion_matrix)
    return post_2d


@njit(cache=True)
def bayesian_decoding_rt(Fr, suv, t_window=100e-3):
    '''
    fast version of below labmda expression
    log_posterior = lambda x,y,i: np.nansum(np.log(Fr[:,x,y]) * suv[:,i]) - 100e-3*Fr[:,x,y].sum(axis=0)

    return both true (x,y) and log_posterior map

    Usage:
    post_2d = bayesian_decoding_rt(Fr=Fr, suv=suv, t_window=100e-3)
    '''

    post_2d = np.zeros((Fr.shape[1], Fr.shape[2]))
    possion_matrix = t_window*Fr.sum(axis=0)
    log_fr = np.log(Fr) # make sure Fr[Fr==0] = 1e-12
    suv_weighted_log_fr = licomb_Matrix(suv.ravel(), log_fr)
    post_2d = np.exp(suv_weighted_log_fr - possion_matrix)
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


def get_hd(trajectory, speed_threshold, offset_hd=180):
    '''
    calcualte head direction from a very short trajectory
    input:
    trajectory: (N,2) array
    speed_threshold: a scalar to filter out the HDs for the final averaging
    output:
    hd:    scalar (inferred head direction of this trajectory)
    speed: scalar (inferred speed of this trajectory)
    '''
    # 1. Calculate both hd and speed (both are vectors)
    delta_pos = np.diff(trajectory, axis=0)
    hd = np.arctan2(delta_pos[:,0], -delta_pos[:,1])
    speed = np.linalg.norm(delta_pos, axis=1)
    # 2. filter out the bad points in the vector
    valid_idx = np.where(np.logical_and(hd!=0, speed>speed_threshold))[0]
    # 3. Calculate the mean value
    hd = np.mean(hd[valid_idx])
    speed = np.mean(speed[valid_idx])
    # 4. Offset the head-direction 
    # Tricky: (which screen animal is looking at?, is there a mirror imaging of the projector? etc)
    hd = (hd*180/np.pi + 360)    # radius to degrees
    hd = (hd+offset_hd)%360      # add the offset head direction
    return hd, speed
