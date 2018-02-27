from .utils import Timer, EventEmitter, Picker
from .cameras import YSyncCamera
from .conf import logger, debug, info, warning, error, critical
import numpy as np


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
