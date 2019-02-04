from .MUA import MUA
from .SPK import SPK
from .FET import FET
from .CLU import CLU, status_manager
from .SPKTAG import SPKTAG
from .Binload import fs2t, bload
from .Probe import probe
import numpy as np


mua_kernel = np.array([-0.012767, -0.010065, -0.010603, -0.015188, -0.022775, -0.030649, -0.035635, -0.035833, -0.031967, -0.027421, -0.026643, -0.032443, -0.043479, -0.05342, -0.052658, -0.032253, 0.011402, 0.07339, 0.14032, 0.19411, 0.21834, 0.20454, 0.15583, 0.086166, 0.015233, -0.038734, -0.0654, -0.065287, -0.048177, -0.027911, -0.016061, -0.017338, -0.028613, -0.04164, -0.04788, -0.042934, -0.028503, -0.011142, 0.0013188, 0.0037242, -0.00405, -0.017387, -0.029505, -0.034875, -0.031708, -0.022382, -0.011779, -0.0045782, -0.0029516, -0.0058103, -0.0098116, -0.01144, -0.0089633, -0.0032537, 0.0028751, 0.0063325, 0.0054634, 0.00089644, -0.0048567, -0.0087553, -0.0087668, -0.0047873, 0.0014692, 0.0074606, 0.0111, 0.011659, 0.0098849, 0.0073795, 0.00565, 0.0053717, 0])
mua_kernel = mua_kernel.astype(np.float32)

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

# import copy_reg
# import types
# copy_reg.pickle(types.MethodType, _pickle_method, _unpickle_method)
#######################################################################



#######################################################################
# This function
# to calculate the avearge knn distance 
# from uncertain points to core clusters

# def distance2clu(u, k=10):
#     dis = []
#     for key, value in app.model.clu[app.ch].index.items():
#         if key > 0:
#             X = app.model.fet[app.ch][value]
#             print(key, X.shape)
#             kd = KDTree(X, p=2)
#             d  = kd.query(u, k)[0]
#             dis.append(d.mean(axis=1))
#     return np.vstack(np.asarray(dis))

#######################################################################
# from numba import jit

# @jit(cache=True)
#######################################################################

#######################################################################
# define some probes 
#######################################################################

