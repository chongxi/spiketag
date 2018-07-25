from torch.multiprocessing import Pool, cpu_count
import numpy as np
from sklearn.neighbors import NearestNeighbors
from hdbscan import HDBSCAN
from time import time
from ..utils.utils import Timer
from ..utils.conf import info, warning
from .CLU import CLU

# def _pickle_method(method):
#     func_name = method.im_func.__name__
#     obj = method.im_self
#     cls = method.im_class
#     if func_name.startswith('__') and not func_name.endswith('__'): #deal with mangled names
#         cls_name = cls.__name__.lstrip('_')
#         func_name = '_' + cls_name + func_name
#     return _unpickle_method, (func_name, obj, cls)

# def _unpickle_method(func_name, obj, cls):
#     for cls in cls.__mro__:
#         try:
#             func = cls.__dict__[func_name]
#         except KeyError:
#             pass
#         else:
#             break
#     return func.__get__(obj, cls)

# import copy_reg
# import types
# copy_reg.pickle(types.MethodType, _pickle_method, _unpickle_method)


class FET(object):
    """
    feature = FET(fet)
    fet: dictionary {group_id:fet[group_id], ...}
    fet[group_id]: n*m matrix, n is #samples, m is #features
    """
    def __init__(self, fet):
        self.fet = fet
        self.group  = []
        self.nSamples = {}
        for g, f in self.fet.items():
            self.nSamples[g] = len(f)
            if len(f) > 0:
                self.group.append(g)
        # exclude channels which no spikes 
        self.fetlen = fet[self.group[0]].shape[1]

        self.hdbscan_hyper_param = {'method': 'hdbscan', 
                                    'min_cluster_size': 18,
                                    'leaf_size': 20}

    def __getitem__(self, i):
        return self.fet[i]

    def __setitem__(self, i, _fet_array):
        self.fet[i] = _fet_array

    def remove(self, group, ids):
        self.fet[group] = np.delete(self.fet[group], ids, axis=0)
 
    def toclu(self, method='hdbscan', njobs=1, *args, **kwargs):
        clu = {}
        
        group_id = kwargs['group_id'] if 'group_id' in kwargs.keys() else None
        fall_off_size = kwargs['fall_off_size'] if 'fall_off_size' in kwargs.keys() else None
        # print 'clustering method: {0}, group_id: {1}, fall_off_size: {2}'.format(method, group_id, fall_off_size)

        if method == 'hdbscan':
            min_cluster_size = self.hdbscan_hyper_param['min_cluster_size']
            leaf_size = self.hdbscan_hyper_param['leaf_size']
            if fall_off_size is not None:
                min_cluster_size = fall_off_size
            hdbcluster = HDBSCAN(min_cluster_size=min_cluster_size, 
                                 leaf_size=leaf_size,
                                 gen_min_span_tree=True, 
                                 algorithm='boruvka_kdtree')

            # automatic pool clustering
            if group_id is None:
                if njobs!=1:
                    info('clustering start with {} cpus'.format(njobs))
                    tic = time()
                    pool = Pool(njobs)
                    _clu = pool.map(self._toclu, self.group)
                    pool.close()
                    pool.join()
                    toc = time()
                    info('clustering finished, used {} sec'.format(toc-tic))
                    # info('get clustering from group_id {}:'.format(str(_group_id)))
                    for _group_id, __clu in zip(self.group, _clu):
                        clu[_group_id] = __clu
                else:
                    info('clustering start with {} cpus'.format(1))
                    tic = time()
                    for group_id in self.group:
                        clusterer = hdbcluster.fit(self.fet[group_id])
                        clu[group_id] = CLU(clusterer.labels_, clusterer)
                    toc = time()
                    info('clustering finished, used {} sec'.format(toc-tic))
                return clu

            # semi-automatic parameter selection for a specific channel
            elif self.nSamples[group_id] != 0:
                # fall_off_size in kwargs
                hdbcluster.min_cluster_size = fall_off_size
                clusterer = hdbcluster.fit(self.fet[group_id])
                return CLU(clusterer.labels_, clusterer)
        else: # other methods 
            warning('Clustering not support {} yet!!'.format(method)) 

    def _toclu(self, group_id, method='hdbscan'):
        if method == 'hdbscan':
            # tic = time()
            from hdbscan import HDBSCAN
            min_cluster_size = self.hdbscan_hyper_param['min_cluster_size']
            leaf_size = self.hdbscan_hyper_param['leaf_size']
            hdbcluster = HDBSCAN(min_cluster_size=min_cluster_size, 
                         leaf_size=leaf_size,
                         gen_min_span_tree=False, 
                         algorithm='boruvka_kdtree',
                         core_dist_n_jobs=1)        
            clusterer = hdbcluster.fit(self.fet[group_id])
            # toc = time()
            # info('fet._toclu(group_id={}, method={})  -- {} sec'.format(group_id, method, toc-tic))
            return CLU(clusterer.labels_, clusterer)
        #  elif method == 'reset':
            #  clu = np.zeros((self.fet[group_id].shape[0], )).astype(np.int64)
        else:
            warning('Clustering not support {} yet!!'.format(method))
        return clu
