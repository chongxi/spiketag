from multiprocessing import Pool
import numpy as np
from sklearn.neighbors import NearestNeighbors
from hdbscan import HDBSCAN
from time import time
from ..utils.utils import Timer
from .CLU import CLU

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
        self.fetlen = fet[0].shape[1]

        self.hdbscan_hyper_param = {'method': 'hdbscan', 
                                    'min_cluster_size': 18,
                                    'leaf_size': 20}

    def __getitem__(self, i):
        return self.fet[i]

    def toclu(self, method='hdbscan', njobs=1, *args, **kwargs):
        clu = {}
        
        chNo = kwargs['chNo'] if 'chNo' in kwargs.keys() else None
        fall_off_size = kwargs['fall_off_size'] if 'fall_off_size' in kwargs.keys() else None
        # print 'clustering method: {0}, chNo: {1}, fall_off_size: {2}'.format(method, chNo, fall_off_size)

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
            if chNo is None:
                if njobs!=1:
                    tic = time()
                    pool = Pool(njobs)
                    _clu = pool.map(self._toclu, self.ch)
                    pool.close()
                    pool.join()
                    toc = time()
                    print 'clustering finished, used {} seconds'.format(toc-tic)
                    for _chNo, __clu in zip(self.ch, _clu):
                        clu[_chNo] = CLU(__clu)
                else:
                    tic = time()
                    for chNo in self.ch:
                        clu[chNo] = CLU(hdbcluster.fit_predict(self.fet[chNo]))
                    toc = time()
                    print 'clustering finished, used {} seconds'.format(toc-tic)
                return clu

            # semi-automatic parameter selection for a specific channel
            elif self.nSamples[chNo] != 0:
                # fall_off_size in kwargs
                hdbcluster.min_cluster_size = fall_off_size
                clu = CLU(hdbcluster.fit_predict(self.fet[chNo]))
                return clu
        else: # other methods 
            pass

    def _toclu(self, chNo, method='hdbscan'):
        if method == 'hdbscan':
            from hdbscan import HDBSCAN
            min_cluster_size = self.hdbscan_hyper_param['min_cluster_size']
            leaf_size = self.hdbscan_hyper_param['leaf_size']
            hdbcluster = HDBSCAN(min_cluster_size=min_cluster_size, 
                         leaf_size=leaf_size,
                         gen_min_span_tree=False, 
                         algorithm='boruvka_kdtree')        
            clu = hdbcluster.fit_predict(self.fet[chNo])
            return clu

    # def torho(self):
    #     nbrs = NearestNeighbors(algorithm='ball_tree', metric='euclidean',
    #                             n_neighbors=25).fit(self.fet)

    #     dismat = np.zeros(self.fet.shape[0])
    #     for i in range(self.fet.shape[0]):
    #         dis,_ = nbrs.kneighbors(self.fet[i].reshape(1,-1), return_distance=True)
    #         dismat[i] = dis.mean()

    #     rho = 1/dismat
    #     self.rho = (rho-rho.min())/(rho.max()-rho.min())
    #     return self.rho
