from place_field import place_field
from place_field import info_bits, info_sparcity
from numba import njit, prange
import torch
import numpy as np


def spk_time_to_suv(spk_time, ts, delta_t=250e-3, sublist=None):
    if sublist is None:
        spk_time_list=list(spk_time.values())
    else:
        spk_time_list = [spk_time.get(key) for key in sublist]
    suv = suv_from_spk_time_list(spk_time_list, ts, delta_t)
    return suv

@njit(cache=True, parallel=True, fastmath=True)
def suv_from_spk_time_list(spk_time_list, ts, delta_t=250e-3):
    N = len(spk_time_list)
    T = ts.shape[0]
    suv = np.zeros((N,T))
    for j in prange(T):    
        for i in prange(N):
#             suv[i, j] = np.where(np.logical_and(spk_time_list[i]>=ts[j]-dt, spk_time_list[i]<ts[j]))[0].shape[0]
            suv[i, j] = np.sum(np.logical_and( spk_time_list[i] >  ts[j]-delta_t, 
                                               spk_time_list[i] <= ts[j]          ))
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
    '''
    Y = np.dot(w, X.reshape(w.shape[0], -1)).reshape(X.shape[1], X.shape[2])
    return Y


@njit(cache=True, parallel=True, fastmath=True)
def bayesian_decoding(Fr, suv, pos, pos_offset, bin_size, delta_t=100e-3):
    '''
    fast version of below labmda expression
    log_posterior = lambda x,y,i: np.nansum(np.log(Fr[:,x,y]) * suv[:,i]) - 100e-3*Fr[:,x,y].sum(axis=0)

    return both true (x,y) and log_posterior map

    Usage:
    1. for all suv
    true_xy, post_2d = bayesian_decoding(Fr=Fr, suv=suv[:, :], pos=pc.pos, pos_offset=pc.maze_original, 
                                         bin_size=pc.bin_size, delta_t=100e-3)

    2. for specific suv
    true_xy, post_2d = bayesian_decoding(Fr=Fr, suv=suv[:, 100:101], pos=pc.pos, pos_offset=pc.maze_original, 
                                         bin_size=pc.bin_size, delta_t=100e-3)
    '''

    true_xy = np.zeros((suv.shape[1],2))
    post_xy = np.zeros((suv.shape[1],2))
    post_2d = np.zeros((suv.shape[1], Fr.shape[1], Fr.shape[2]))
    possion_matrix = delta_t*Fr.sum(axis=0)
    log_fr = np.log(Fr) # make sure Fr[Fr==0] = 1e-12
    for i in range(suv.shape[1]): # i is time point
        suv_weighted_log_fr = licomb_Matrix(suv[:,i], log_fr)
        true_xy[i] = (pos[i]-pos_offset)//bin_size
        post_2d[i] = np.exp(suv_weighted_log_fr - possion_matrix)
    return true_xy, post_2d


def argmax_2d_tensor(X):
    values, indices = torch.max(torch.from_numpy(X.reshape(X.shape[0],-1)), 1)
    post_xy = np.vstack((indices.numpy()%X.shape[2], indices.numpy()//X.shape[2])).T
    return post_xy

def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

# def smooth(x, window_len=50, window='hanning'):
#     """smooth the data using a window with requested size.
    
#     This method is based on the convolution of a scaled window with the signal.
#     The signal is prepared by introducing reflected copies of the signal 
#     (with the window size) in both ends so that transient parts are minimized
#     in the begining and end part of the output signal.
    
#     input:
#         x: the input signal 
#         window_len: the dimension of the smoothing window; should be an odd integer
#         window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
#             flat window will produce a moving average smoothing.

#     output:
#         the smoothed signal
        
#     example:

#     t=linspace(-2,2,0.1)
#     x=sin(t)+randn(len(t))*0.1
#     y=smooth(x)
    
#     see also: 
    
#     numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
#     scipy.signal.lfilter
 
#     TODO: the window parameter could be the window itself if an array instead of a string
#     NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
#     """

#     if x.ndim != 1:
#         raise ValueError, "smooth only accepts 1 dimension arrays."

#     if x.size < window_len:
#         raise ValueError, "Input vector needs to be bigger than window size."


#     if window_len<3:
#         return x


#     if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
#         raise ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'"


#     s=np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
#     #print(len(s))
#     if window == 'flat': #moving average
#         w=np.ones(window_len,'d')
#     else:
#         w=eval('np.'+window+'(window_len)')

#     y=np.convolve(w/w.sum(), s, mode='same')
#     return y



