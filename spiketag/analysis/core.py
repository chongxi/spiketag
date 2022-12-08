import numpy as np
import torch
import torch.nn.functional as F
from scipy import signal
from scipy.signal import butter, lfilter, freqz
from numba import njit, prange
from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)

def acorr(X, norm=False):
    '''
    FFT-based autocorrelation function for 1D or higher dimensional time series.
    When X is 1D, the autocorrelation return same results as `scipy.signal.correlate(X, X)` or `np.convolve(X, np.conj(X)[::-1])`
    Input:
        X: n by d (n time points, each has d dimension)
        norm: if True, normalize the autocorrelation that its maximum is 1
    Output:
        ACF (AutoCorrelation Function): a n-point sequence that represents the degree of similarity 
                                        between X and a lagged version of X over successive n time points.

    Note: this is function different from what was described in 
          https://stackoverflow.com/questions/4503325/autocorrelation-of-a-multidimensional-array-in-numpy
          our acorr function truly treat time series as ND array that the correlation between a d-dimension signal (a row) 
          and their lagged version (at different rows) are calculated.  
    '''
    from numpy.fft import fft, ifft
    if X.ndim == 1:
        X = X.reshape(-1,1)
    FT = fft(X.T, n=len(X)*2 + 1)
    ACF = ifft(FT*np.conj(FT)).sum(axis=0).real
    ACF = np.roll(ACF, len(X))
    if ACF[0] < 1e-6 or ACF[-1] < 1e-6:
        ACF = ACF[1:-1] # the first and last number is zero
    if norm:
        ACF = ACF/ACF.max()
    return ACF

def softmax(X):
    X = torch.tensor(X).double()
    X_exp = torch.exp(X-X.max())
    partition = torch.sum(X_exp, dim=1, keepdim=True)
    _softmax = X_exp / partition # The broadcast mechanism is applied here
    return _softmax.numpy()

def pos2speed(pos, ts=None):
    '''
    input:
        pos: (N,2) array
        ts:  (N,) array
    output:
        speed: (N, 2) array (the speed at time 0 is copied to make the array size the same)
    '''
    if ts is not None:
        speed = np.diff(pos, axis=0)/np.diff(ts).reshape(-1,1)
    else:
        speed = np.diff(pos, axis=0)
    speed = np.vstack((speed[0], speed))
    return speed

def sliding_window_to_feature(scv, n):
    '''
    used to stack scv, turn scv at different time (rows) into new feature (columns)
    `n` is the number of sliding window that is added to feature (column)

    from numpy.lib.stride_tricks import sliding_window_view
    this function is equivalent to "sliding_window_view(scv.ravel(), (n+1)*scv.shape[1])[::scv.shape[1]]"

    >>> x = np.arange(5*3).reshape(5,3)
    >>> x
    >>> array([[ 0,  1,  2],
               [ 3,  4,  5],
               [ 6,  7,  8],
               [ 9, 10, 11],
               [12, 13, 14]])
    >>> sliding_window_to_feature(x, 1)
    >>> array([[ 0,  1,  2,  3,  4,  5],
               [ 3,  4,  5,  6,  7,  8],
               [ 6,  7,  8,  9, 10, 11],
               [ 9, 10, 11, 12, 13, 14]])
    >>> sliding_window_to_feature(x, 2)
    >>> array([[ 0,  1,  2,  3,  4,  5,  6,  7,  8],
               [ 3,  4,  5,  6,  7,  8,  9, 10, 11],
               [ 6,  7,  8,  9, 10, 11, 12, 13, 14]])
    >>> sliding_window_to_feature(x, 3)  # sliding_window_view(x.ravel(), (3+1)*x.shape[1])[::x.shape[1]]
    >>> array([[ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11],
               [ 3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14]])
    >>> sliding_window_to_feature(x, 4)
    >>> array([[ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14]])
    '''
    assert(n<scv.shape[0]),f"n must be less than {scv.shape[0]} since array only has {scv.shape[0]} rows"
    _scv_list = [scv[i:i-n] if i<n else scv[n:] for i in range(n+1)]
    new_scv = np.hstack(_scv_list)
    return new_scv

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


@njit(cache=True)
def get_corr_field(pv, fields):
    '''
    Calculate the correlation field of a population vector and a rate vector for each position in a maze.
    
    * The population vector (pv) is a 1-dimensional array of shape (N,) that represents the spike count of
      a group of N neurons in the last few hundreds of milliseconds.
    * The rate vector (rv) is a 1-dimensional array of shape (N,) that represents the firing rate of each
      neuron at a specific position in the maze.
    * The rate vector is extracted from the fields array, which has shape (N, M, M) and contains the
      place fields (M,M) of N neuron. The place fields are the spatial firing rate of the neurons.
    * The function loops over each position in the maze (M,M) and calculates the correlation coefficient between
      the population vector and the rate vector at that position. The correlation coefficient is then stored
      in the corresponding position in the correlation field, which is a 2-dimensional array of shape (M, M)
      containing the correlation coefficients for each position in the maze.
      
    Args:
        pv (np.ndarray): The population vector of shape (N,). e.g., pc.scv[20]
        fields (np.ndarray): The place fields of shape (N, M, M), where M is the size of the maze. pc.fields in spiketag can
                             be used here. e.g., pc.fields
        
    Returns:
        np.ndarray: The correlation field of shape (M, M) containing the correlation coefficients between PV and RV for each
                     position in the maze.

    Example:
        # scv[20] is the population vector at time bin 20, pc.fields is the place fields, to calculate the correlation field:
        cf = get_corr_field(scv[20], pc.fields) 

    with njit, this function takes 7.5 ms (20x faster than the pure python version)
    '''
    cf = np.zeros((40, 40))
    for i in range(40):
        for j in range(40):
            rv = fields[:, j, i]
            cf[j, i] = np.corrcoef(pv, rv)[0, 1]
    return cf

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
            suv[i, j] = np.sum(np.logical_and(spk_time_list[j] >  ts[i] - t_window,
                                              spk_time_list[j] <= ts[i]))
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
def bayesian_decoding(Fr, suv, t_window=100e-3, mean_firing_rate=None):
    '''
    fast version of below labmda expression
    log_posterior = lambda x,y,i: np.nansum(np.log(Fr[:,x,y]) * suv[:,i]) - 100e-3*Fr[:,x,y].sum(axis=0)

    return both true (x,y) and log_posterior map

    Usage:
    post_2d = bayesian_decoding(Fr=Fr, suv=suv, t_window=100e-3)
    '''
    # mean_firing_rate = np.mean(suv)  # F_mean is the average firing rate of all cells over all time bins
    post_2d = np.zeros((suv.shape[0], Fr.shape[1], Fr.shape[2]))
    for i in prange(suv.shape[0]): # i is time point
        if mean_firing_rate is not None:
            firing_rate_ratio = np.mean(suv[i].ravel())/mean_firing_rate # firing rate modulation factor (e.q 46 in Zhang et al. 1998)
        else:
            firing_rate_ratio = 1
        suv_weighted_log_fr = licomb_Matrix(suv[i].ravel(), np.log(Fr))
        post_2d[i] = np.exp(suv_weighted_log_fr - firing_rate_ratio * t_window * Fr.sum(axis=0))
        # post_2d[i] = np.nan_to_num(post_2d[i]/post_2d[i].sum()) # nan_to_num because it is possible that post_2d[i].sum() = 0
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


############################################################################################
# Get LFP segment from the MUA data
# Wiener filter, low-pass filter, and downsampling to LFP
# Example:
# bf = bload()
# bf.load('./mua.bin', dtype='int32')
# ch, mua_fs, lfp_fs = 10, 25000, 1000
# t0, t1 = 101.0, 102.0 # in seconds
# tlfp, lfp = get_LFP(bf, t0, t1, ch, mua_fs, lfp_fs)
############################################################################################

def wiener_deconvolution_torch_gpu(signal, kernel):
    import torch.nn.functional as f
    signal = signal.cuda()/2**13  # FPGA used 13 bits for the fractional part
    kernel = torch.from_numpy(kernel).cuda()
    kernel = F.pad(kernel, (0, len(signal)-len(kernel)))
    H = torch.fft.fft(kernel)
    deconvolved = torch.fft.ifft(torch.fft.fft(
        signal) * torch.conj(H)/(H*torch.conj(H)))
    return deconvolved.real.cpu()


def butter_lowpass(cutoff, fs=25000, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a


def get_LFP(bf, t0, t1, ch, mua_fs=25000, lfp_fs=1000, cutoff=400):
    from ..base import mua_kernel
    # step 1: wiener deconvolution (25000 Hz) to reconstruct raw waveform
    offset = 10000
    mua_wav = bf.data[int(t0*25000)-offset:int(t1*25000)+offset, ch]
    raw_wav = wiener_deconvolution_torch_gpu(
        mua_wav, mua_kernel)[:len(mua_wav)][offset:-offset]
    raw_wav = signal.detrend(raw_wav)

    # step 2: downsample raw waveform to 1000 (or 1500) Hz
    raw_wav = signal.resample(raw_wav, int(raw_wav.shape[0]/mua_fs*lfp_fs))

    # step 3: low-pass filter to get lfp (also remove the ultra-low frequency)
    b, a = butter_lowpass(cutoff, fs=lfp_fs, order=5)
    lfp = signal.filtfilt(b, a, raw_wav, padlen=1000)
    tlfp = np.linspace(t0, t1-offset/mua_fs, lfp.shape[0], endpoint=False)
#     b, a = signal.iirnotch(1, 200, fs)
#     lfp = signal.filtfilt(b, a, lfp, padlen=1000)
    lfp = signal.detrend(lfp)
    return tlfp, lfp


def affine_pivot(img, angle, pivot=None, scale=1, padding=0, grid_expansion=2, center_at_pivot=True, return_transformation=False):
    """translate image to center at pivot, rotate angle degrees and then scale it

    Args:
    For just one image:
        img: (H, W) 
        angle: a float or int indicating the rotation degree, counter-clockwise 
        pivot: (x, y), if no pivot is given, the center of the image will be used
        padding: a float or int indicating the padding size, default is 0
        grid_expansion: a scalar deciding the grid scaling size, default is 2
        center_at_pivot: a boolean indicating whether to center the image at pivot
        return_transformation: a boolean indicating whether to return the transformation matrix

    For N images in img (a batch):
        img (numpy array or torch tensor): (N, C, H, W)
        angle (numpy array or torch tensor): (N,)
        pivot (numpy array or torch tensor): (N, 2)
        scale (numpy array or torch tensor): (N,)
        padding (numpy array or torch tensor): a scalar value, same padding for all images in img batch

    Variables:
        R (torch tensor): (N, 2, 3) rotation/scaling matrix
        r (torch tensor): (N,) angle to rotate (in radians) for N images
        grid: (torch tensor): (N, C, grid_expansion*(H+2*padding), grid_expansion*(W+2*padding), 2) flow field of rotated/scaled coordinates

    Note:
        - Without `padding`, the transformed image can be cropped.
        - Without `grid_expansion`, the transformed image can be distorted.
        These two parameters should be tuned together.
        Images are first padded to `H+2*padding`, `W+2*padding`, then sampled with a grid produced by a affine transformation (decided by angle, pivot and scale)
        the grid is expanded by `grid_expansion`, so that the grid is not only a grid of coordinates, but also a grid of coordinates with a grid_expansion times larger size
        the grid is then sampled with the original image, and the result is the transformed image
        
    Returns:
        af_img: (torch tensor): (N, C, grid_expansion*(H+2*padding), grid_expansion*(W+2*padding)) rotated/scaled image
    """
    # before processing, check if img is a torch tensor or numpy array
    # also check the shape of each input
    if type(img) == np.ndarray:
        img = torch.from_numpy(img).float()
    if img.dim() == 2: # (H, W) -> (1, 1, H, W)
        img = img.reshape(1,1,img.shape[0],img.shape[1]).float()
    elif img.dim() == 3: # (N, H, W) -> (N, 1, H, W) 
        img = img.reshape(-1, 1, img.shape[1], img.shape[2]).float()
    else:
        img = img.float()
    
    if type(angle) == np.ndarray:
        angle = torch.from_numpy(angle).float()
    if type(angle) == float or type(angle) == int:
        angle = torch.tensor([angle]).float()
    r = torch.deg2rad(angle).reshape(-1,)

    if pivot is None:
        pivot = torch.tensor([img.shape[2]//2, img.shape[3]//2], dtype=torch.float)
    if type(pivot) == np.ndarray:
        pivot = torch.from_numpy(pivot).float()
    if type(pivot) == tuple:
        pivot = torch.tensor(pivot, dtype=torch.float)
    if pivot.shape == (2,):
        pivot = pivot.reshape(1,2)

    if type(scale) == np.ndarray:
        scale = torch.from_numpy(scale).float()
    if type(scale) == float or type(scale) == int:
        scale = torch.tensor([scale]).float()

    # step1: pad all images by padding
    img = F.pad(img, [padding, padding, padding, padding], "constant", 0)
    N, C, H, W = img.shape

    # step2: construct the translation and scaling matrix 
    tx = (padding+pivot[:, 0]-W//2)/(W//2)
    ty = (padding+pivot[:, 1]-H//2)/(H//2)

    # (manually inversed already to save computing)
    T = torch.zeros((N, 3, 3))
    T[:, 0, 0] = 1/scale*grid_expansion
    T[:, 0, 2] = tx
    T[:, 1, 1] = 1/scale*grid_expansion
    T[:, 1, 2] = ty
    T[:, 2, 2] = 1

    # step3: construct the rotation matrix 
    R = torch.zeros((N, 3, 3))
    R[:, 0, 0] =  torch.cos(r)
    R[:, 0, 1] = -torch.sin(r)
    R[:, 1, 0] =  torch.sin(r)
    R[:, 1, 1] =  torch.cos(r)
    R[:, 2, 2] =  1

    # step4: construct the affine matrix
    if center_at_pivot:
        # translate, scale, rotate == rotate around the pivot poisition and the pivot position is the center of the image
        M = T@R
    else:
        # translate, scale, rotate and translate back == rotate around the pivot poisition (pivot position itself doesn't move)
        M = T@R@torch.linalg.inv(T)

    M = M[:, :2, :]

    # step4: apply the transformation matrix to generate a flow grid
    grid = F.affine_grid(M, size=(N, C, grid_expansion*H, grid_expansion*W), align_corners=True)
    # step5: sample (bilinear interpolation) the image with the flow grid to produce the rotated and scaled image
    af_img = F.grid_sample(img, grid, align_corners=True)

    if return_transformation:
        return M, grid, af_img
    else:
        return af_img


def rotate_scale(img, start, pivot, padding=0, target_distance=30, autoscale=True):
    """
    Rotate and scale an image to a `target_distance` from the pivot position, while the pivot position will be translated to the center of the image.
    The image will be rotated by the angle of the line connecting the start and the pivot position.
    After transformation, the new_pivot will be the center of the image, the new_start will be at the left side of the new_pivot.
    The image will be scaled such that the distance between the new_start and new_pivot is `target_distance`.
    The image will be padded by `padding` to make sure the image is not cropped.

    if autoscale is True, the image will be scaled such that the distance between the start and the pivot is `target_distance`.
    if autoscale is False, the image will not be scaled (only rotated and translated). 

    TODO: make it works for batch of images (N,C,H,W) and batch of start and pivot (N, 2)
    """

    vec = pivot - start
    distance = np.linalg.norm(vec)
    angle = np.array(np.angle(vec[0]+1j*vec[1], deg=True))

    if autoscale:
        scale = target_distance/distance
        plot_distance = target_distance
    else:
        scale = 1
        plot_distance = distance
    rotated_img = affine_pivot(img.copy(), angle=angle, pivot=pivot,
                               scale=scale, grid_expansion=1, padding=padding).squeeze().numpy()

    new_start = (rotated_img.shape[0]//2-plot_distance, rotated_img.shape[1]//2)
    new_pivot = (rotated_img.shape[0]//2, rotated_img.shape[1]//2)
    return rotated_img, new_start, new_pivot 
