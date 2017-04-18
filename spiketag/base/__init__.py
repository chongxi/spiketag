from .MUA import MUA
from .SPK import SPK
from .FET import FET
from .CLU import CLU
from .SPKTAG import SPKTAG
from .Binload import bload
from .Probe import ProbeFactory  

#######################################################################
# This is just a patch for multiprocessing 
# to be used in class (python 2.7)
from multiprocessing import Pool
from functools import partial

def _pickle_method(method):
    func_name = method.im_func.__name__
    obj = method.im_self
    cls = method.im_class
    if func_name.startswith('__') and not func_name.endswith('__'): #deal with mangled names
        cls_name = cls.__name__.lstrip('_')
        func_name = '_' + cls_name + func_name
    return _unpickle_method, (func_name, obj, cls)

def _unpickle_method(func_name, obj, cls):
    for cls in cls.__mro__:
        try:
            func = cls.__dict__[func_name]
        except KeyError:
            pass
        else:
            break
    return func.__get__(obj, cls)

import copy_reg
import types
copy_reg.pickle(types.MethodType, _pickle_method, _unpickle_method)
#######################################################################



#######################################################################
# This function
# to calculate the avearge knn distance 
# from uncertain points to core clusters

def distance2clu(u, k=10):
    dis = []
    for key, value in app.model.clu[app.ch].index.items():
        if key > 0:
            X = app.model.fet[app.ch][value]
            print key, X.shape
            kd = KDTree(X, p=2)
            d  = kd.query(u, k)[0]
            dis.append(d.mean(axis=1))
    return np.vstack(np.asarray(dis))

#######################################################################
# from numba import jit

# @jit(cache=True)
#######################################################################
