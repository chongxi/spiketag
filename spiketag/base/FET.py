from multiprocessing import Pool, cpu_count
import numpy as np
from sklearn.neighbors import NearestNeighbors
from hdbscan import HDBSCAN
from time import time
from ..utils.utils import Timer
from ..utils.conf import info, warning
from .CLU import CLU

class FET(object):
    """
    feature = FET(fet)
    fet: dictionary {groupNo:fet[groupNo], ...}
    fet[groupNo]: n*m matrix, n is #samples, m is #features
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
        
        groupNo = kwargs['groupNo'] if 'groupNo' in kwargs.keys() else None
        fall_off_size = kwargs['fall_off_size'] if 'fall_off_size' in kwargs.keys() else None
        # print 'clustering method: {0}, groupNo: {1}, fall_off_size: {2}'.format(method, groupNo, fall_off_size)

        if method == 'hdbscan':
            min_cluster_size = self.hdbscan_hyper_param['min_cluster_size']
            leaf_size = self.hdbscan_hyper_param['leaf_size']
            if fall_off_size is not None:
                min_cluster_size = fall_off_size
            hdbcluster = HDBSCAN(min_cluster_size=min_cluster_size, 
                                 leaf_size=leaf_size,
                                 gen_min_span_tree=True, 
                                 algorithm='boruvka_kdtree')

            # automatic tatch clustering
            if groupNo is None:
                if njobs!=1:
                    tic = time()
                    pool = Pool(njobs)
                    _clu = pool.map(self._toclu, self.group)
                    pool.close()
                    pool.join()
                    toc = time()
                    info('clustering finished, used {} seconds'.format(toc-tic))
                    for _groupNo, __clu in zip(self.group, _clu):
                        clu[_groupNo] = __clu
                else:
                    tic = time()
                    for groupNo in self.group:
                        clusterer = hdbcluster.fit(self.fet[groupNo])
                        clu[groupNo] = CLU(clusterer.labels_, clusterer)
                    toc = time()
                    info('clustering finished, used {} seconds'.format(toc-tic))
                return clu

            # semi-automatic parameter selection for a specific channel
            elif self.nSamples[groupNo] != 0:
                # fall_off_size in kwargs
                hdbcluster.min_cluster_size = fall_off_size
                clusterer = hdbcluster.fit(self.fet[groupNo])
                return CLU(clusterer.labels_, clusterer)
        else: # other methods 
            warning('Clustering not support {} yet!!'.format(method)) 

    def _toclu(self, groupNo, method='hdbscan'):
        if method == 'hdbscan':
            from hdbscan import HDBSCAN
            min_cluster_size = self.hdbscan_hyper_param['min_cluster_size']
            leaf_size = self.hdbscan_hyper_param['leaf_size']
            hdbcluster = HDBSCAN(min_cluster_size=min_cluster_size, 
                         leaf_size=leaf_size,
                         gen_min_span_tree=False, 
                         algorithm='boruvka_kdtree',
                         core_dist_n_jobs=cpu_count())        
            clusterer = hdbcluster.fit(self.fet[groupNo])
            return CLU(clusterer.labels_, clusterer)
        elif method == 'reset':
            # FIXME choose the root of tree
            clu = np.zeros((self.fet[groupNo].shape[0], )).astype(np.int64)
        else:
            warning('Clustering not support {} yet!!'.format(method))
        return clu
