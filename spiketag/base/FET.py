from torch.multiprocessing import Pool, cpu_count
from torch import multiprocessing as multiprocessing
import numpy as np
from sklearn.neighbors import NearestNeighbors
# from hdbscan import HDBSCAN
import hdbscan
from time import time
from ..utils.utils import Timer
from ..utils.conf import info, warning
from .CLU import CLU



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

        self.clustering_func = {"reset":   self._reset,
                                "hdbscan": self._hdbscan,
                                "dpgmm":   self._dpgmm}

        self.hdbscan_hyper_param = {'method': 'hdbscan', 
                                    'min_cluster_size': 18,
                                    'leaf_size': 40,
                                    'eom_or_leaf': 'eom'}

        self.dpgmm_hyper_param = {'max_n_clusters': 10,
                                  'max_iter':       300}


    def __getitem__(self, i):
        return self.fet[i]

    def __setitem__(self, i, _fet_array):
        self.fet[i] = _fet_array

    def remove(self, group, ids):
        self.fet[group] = np.delete(self.fet[group], ids, axis=0)
 

    def toclu(self, method, params=None, group_id='all', njobs=24):
        clu_dict = {}
        print('clustering method: {0}, group_id: {1}'.format(method, group_id))

        '''
        1. When group_id is not provided, means parallel sorting on all groups
        '''
        if group_id is 'all':
            info('clustering for all groups with {} cpus'.format(njobs))
            tic = time()
            pool = Pool(njobs)
            _clu = pool.map(self.clustering_func[method], self.group)
            pool.close()
            pool.join()
            toc = time()
            info('clustering finished, used {} sec'.format(toc-tic))

            for _group_id, __clu in zip(self.group, _clu):
                clu_dict[_group_id] = __clu
            return clu_dict


        '''
        2. When group_id is provided, and background sorting (async and non-blocking) is required
        '''



        '''
        3. When group_id is provided, and sort in blocking mannter
        '''


    def _reset(self, group_id):
        clu = CLU(np.zeros((self.fet[group_id].shape[0], )).astype(np.int64))
        clu._id = group_id
        return clu 
            

    def _hdbscan(self, group_id):
        '''
        a pool method to hdbscan for each group_id
        '''
        import hdbscan
        min_cluster_size = self.hdbscan_hyper_param['min_cluster_size']
        leaf_size = self.hdbscan_hyper_param['leaf_size']
        eom_or_leaf = self.hdbscan_hyper_param['eom_or_leaf']
        hdbcluster = hdbscan.HDBSCAN(min_samples=2,
                     min_cluster_size=min_cluster_size, 
                     leaf_size=leaf_size,
                    #  alpha=0.1,
                     gen_min_span_tree=True, 
                     algorithm='boruvka_kdtree',
                     core_dist_n_jobs=1,
                     prediction_data=True,
                     cluster_selection_method=eom_or_leaf) # eom or leaf 
        clusterer = hdbcluster.fit(self.fet[group_id].astype(np.float64))
        probmatrix = hdbscan.all_points_membership_vectors(clusterer)
        # toc = time()
        # info('fet._toclu(group_id={}, method={})  -- {} sec'.format(group_id, method, toc-tic))
        clu = CLU(clu = clusterer.labels_, method='hdbscan',
                  clusterer=clusterer, probmatrix=probmatrix)
        clu._id = group_id
        return clu



    def _dpgmm(self, group_id):
        # TODO 
        pass






        # if method == 'dpgmm':
        #     if group_id is None:
        #         if njobs!=1:
        #             info('clustering start with {} cpus'.format(njobs))
        #             tic = time()
        #             pool = Pool(njobs)
        #             _clu = pool.map(self._toclu, self.group)
        #             pool.close()
        #             pool.join()
        #             toc = time()
        #             info('clustering finished, used {} sec'.format(toc-tic))
        #             # info('get clustering from group_id {}:'.format(str(_group_id)))
        #             for _group_id, __clu in zip(self.group, _clu):
        #                 clu[_group_id] = __clu
        #     else:
        #         from sklearn.mixture import BayesianGaussianMixture as DPGMM
        #         max_n_clusters = self.dpgmm_hyper_param['max_n_clusters']
        #         max_iter       = self.dpgmm_hyper_param['max_iter']
        #         dpgmm = DPGMM(
        #                     n_components=max_n_clusters, covariance_type='full', weight_concentration_prior=1e-3,
        #                     weight_concentration_prior_type='dirichlet_process', init_params="kmeans",
        #                     max_iter=max_iter, random_state=0) # init can be "kmeans" or "random"
        #         dpgmm.fit(self.fet[group_id])
        #         labels = dpgmm.predict(self.fet[group_id])
        #         return CLU(clu = labels, method='dpgmm')


        # elif method == 'hdbscan':
        #     min_cluster_size = self.hdbscan_hyper_param['min_cluster_size']
        #     leaf_size = self.hdbscan_hyper_param['leaf_size']
        #     eom_or_leaf = self.hdbscan_hyper_param['eom_or_leaf']
        #     hdbcluster = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, 
        #                          leaf_size=leaf_size,
        #                          gen_min_span_tree=True, 
        #                          algorithm='boruvka_kdtree',
        #                          core_dist_n_jobs=8,
        #                          prediction_data=True,
        #                          cluster_selection_method=eom_or_leaf)  # or leaf
            

        #     # automatic pool clustering
        #     if group_id is None:
        #         if njobs!=1:
        #             info('clustering start with {} cpus'.format(njobs))
        #             tic = time()
        #             pool = Pool(njobs)
        #             _clu = pool.map(self._toclu, self.group)
        #             pool.close()
        #             pool.join()
        #             toc = time()
        #             info('clustering finished, used {} sec'.format(toc-tic))
        #             # info('get clustering from group_id {}:'.format(str(_group_id)))
        #             for _group_id, __clu in zip(self.group, _clu):
        #                 clu[_group_id] = __clu
        #         # else:
        #         #     info('clustering start with {} cpus'.format(1))
        #         #     tic = time()
        #         #     for group_id in self.group:
        #         #         clusterer = hdbcluster.fit(self.fet[group_id].astype(np.float64))
        #         #         probmatrix = hdbscan.all_points_membership_vectors(clusterer)
        #         #         clu[group_id] = CLU(clu = clusterer.labels_, method='hdbscan',
        #         #                             clusterer=clusterer, probmatrix=probmatrix)
        #         #     toc = time()
        #         #     info('clustering finished, used {} sec'.format(toc-tic))
        #         return clu

        #     # semi-automatic parameter selection for a specific group
        #     elif self.nSamples[group_id] != 0:
        #         clusterer = hdbcluster.fit(self.fet[group_id].astype(np.float64))
        #         probmatrix = hdbscan.all_points_membership_vectors(clusterer)
        #         return CLU(clu = clusterer.labels_, method='hdbscan', 
        #                    clusterer=clusterer, probmatrix=probmatrix)
        # else: # other methods 
        #     warning('Clustering not support {} yet!!'.format(method)) 

