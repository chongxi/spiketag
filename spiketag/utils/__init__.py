from .utils import Timer, EventEmitter, Picker, key_buffer
from .tool import comet, Player, decoder_viewer
from .plotting import colorbar, colorline
from .cameras import XSyncCamera, YSyncCamera
from .conf import logger, debug, info, warning, error, critical
from . import conf
import numpy as np
from scipy.interpolate import interp1d


def order_label(vec):
    '''
    order label as 0->N for any vector
    '''
    _vec = vec.copy()
    _uniq = np.unique(_vec) 
    for i in np.argsort(_uniq):
        _vec[_vec==_uniq[i]] = i 
    return _vec


def shift_label(vec, const):
    '''
    except 0, every element add a constant
    '''
    return np.array([_v + const if _v!=0 else _v for _v in vec])


def fs2t(N, fs):
    dt = 1./fs
    t = np.arange(0, N*dt, dt)
    return t


def inNd(a, b, axis=0, assume_unique=False):
    '''
    Test whether each element of a Nd array is also present in a second array.
    Returns a boolean array the same length as `a` that is True where an element of `a` is in `b` and False otherwise.
    
    Parameters
    ----------
    a : (M,N) numpy array
    b : (M,N) numpy array 
    axis: axis through which a and b to be compared. Default is 0
    assume_unique : bool, optional
        If True, the input arrays are both assumed to be unique, which
        can speed up the calculation.  Default is False.

    Returns
    -------
    in1d : (M,) ndarray, bool
    The values `a[in1d,:]` are in `b` if axis=0.
    The values `a[:,in1d]` are in `b` if axis=1.
    '''
    if axis==0:
        a = np.asarray(a, order='C')
        b = np.asarray(b, order='C')
    if axis==1:
        a = np.asarray(a.T, order='C')
        b = np.asarray(b.T, order='C')
    a = a.ravel().view((np.str, a.itemsize * a.shape[1]))
    b = b.ravel().view((np.str, b.itemsize * b.shape[1]))
    # print a.shape
    # print b.shape
    return np.in1d(a, b, assume_unique)


def interpNd(data, fs_old, fs_new, method='quadratic'):
    N, nCh = data.shape
    t = fs2t(N, fs_old)
    new_t = np.arange(0, t[-1], 1/fs_new)
    new_data = np.zeros((new_t.shape[0], nCh))
    for i, datum in enumerate(data.T):
        print('resample {}th channel'.format(i))
        f = interp1d(t, datum, method)
        new_data[:,i] = f(new_t)
    return new_data


def searchsorted_nn(seq0, seq1, mode='right'):
    '''
    search seq1 in seq0
    return the `left` or `right` nearest neighbour of seq1 in seq0
    So the length of returned seq is the same as the length of seq1

    Test:
    a = np.cumsum(abs(np.random.randn(3,1)))
    b = np.cumsum(abs(np.random.randn(5,1))) 
    seq0, seq1 = a, b
    print(searchsorted_nn(seq0, seq1, mode='left'))
    plt.eventplot(seq0, lineoffsets=2.5)
    plt.eventplot(seq1)
    '''
    if mode == 'left':
        _seq = seq0[np.searchsorted(seq0, seq1) - 1]
        _seq[(_seq-seq1)>=0] = np.nan
        return _seq
    elif mode == 'right':
        idx = np.searchsorted(seq0, seq1)
        idx[idx==len(seq0)] = 0
        _seq = seq0[idx]
        _seq[(_seq-seq1)<=0] = np.nan
        return _seq
