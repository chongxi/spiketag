from multiprocessing import Pool
import numpy as np
from sklearn.neighbors import NearestNeighbors
from hdbscan import HDBSCAN
from time import time
from ..utils.utils import Timer
from .CLU import CLU

#######################################################################
# This is just a patch for multiprocessing in class (python 2.7)
# patch for multiprocessing
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

class FET(object):
    """
    feature = FET(fet)
    fet: dictionary {chNo:fet[chNo], ...}
    fet[chNo]: n*m matrix, n is #samples, m is #features
    """
    def __init__(self, fet):
        self.fet = fet
        self.ch  = []
        self.nSamples = {}
        for ch, f in self.fet.items():
            self.nSamples[ch] = len(f)
            if len(f) > 0:
                self.ch.append(ch)

    def __getitem__(self, i):
        return self.fet[i]

    def get_rho(self):
        nbrs = NearestNeighbors(algorithm='ball_tree', metric='euclidean',
                                n_neighbors=25).fit(self.fet)

        dismat = np.zeros(self.fet.shape[0])
        for i in range(self.fet.shape[0]):
            dis,_ = nbrs.kneighbors(self.fet[i].reshape(1,-1), return_distance=True)
            dismat[i] = dis.mean()

        rho = 1/dismat
        self.rho = (rho-rho.min())/(rho.max()-rho.min())
        return self.rho

    def toclu(self, method='hdbscan', chNo=None, njobs=1):
        clu = {}
        if method == 'hdbscan':
            min_cluster_size = 5
            leaf_size = 10
            hdbcluster = HDBSCAN(min_cluster_size=min_cluster_size, 
                                 leaf_size=leaf_size,
                                 gen_min_span_tree=True, 
                                 algorithm='boruvka_kdtree')
            if chNo is None:
                if njobs!=1:
                    tic = time()
                    pool = Pool(njobs)
                    _clu = pool.map(self._toclu, self.ch)
                    toc = time()
                    print 'clustering finished, used {} seconds'.format(toc-tic)
                    for _chNo, __clu in zip(self.ch, _clu):
                        clu[_chNo] = CLU(__clu)
                    # return clu
                else:
                    tic = time()
                    for chNo in self.ch:
                        clu[chNo] = CLU(hdbcluster.fit_predict(self.fet[chNo]))
                    toc = time()
                    print 'clustering finished, used {} seconds'.format(toc-tic)
                return clu

            elif self.nSamples[chNo] != 0:
                clu[chNo] = CLU(hdbcluster.fit_predict(self.fet[chNo]))
                return clu
        else: # other methods 
            pass

    def _toclu(self, chNo, method='hdbscan'):
        from hdbscan import HDBSCAN
        min_cluster_size = 5
        leaf_size = 10
        hdbcluster = HDBSCAN(min_cluster_size=min_cluster_size, 
                     leaf_size=leaf_size,
                     gen_min_span_tree=False, 
                     algorithm='boruvka_kdtree')        
        clu = hdbcluster.fit_predict(self.fet[chNo])
        return clu


# if __name__ == '__main__':
#     f = np.load('fet.npy').item()
#     fet = FET(f)
#     t0 = ptime.time()
#     clu_0 = fet.toclu(method='hdbscan', njobs=16)
#     t1 = ptime.time()
#     print t1-t0

#     t0 = ptime.time()
#     clu_1 = fet.toclu(method='hdbscan', njobs=1)
#     t1 = ptime.time()
#     print t1-t0


    ############################
    # fet = []
    # # clu = []
    # for i, _f in f.items():
    #     if len(_f)>0:
    #         fet.append(_f)
    #         # clu.append(toclu(_f))

    # # # print clu
    # t0 = ptime.time()
    # pool = Pool(16)
    # clu = pool.map(toclu, fet)   
    # print clu
    # t1 = ptime.time()
    # print t1-t0
    #######################################

    # print clu

    # p = Pool(32)
    # print(p.map(f, [1, 2, 3]))
