from torch.multiprocessing import Pool, cpu_count
from torch import multiprocessing as multiprocessing
import numpy as np
from sklearn.neighbors import NearestNeighbors
# from hdbscan import HDBSCAN
import hdbscan
import ipyparallel as ipp
from time import time
from ..utils.utils import Timer
from ..utils.conf import info, warning
from .CLU import CLU


class cluster():
    def __init__(self, clu_status):
        self.client = ipp.Client()
        self.cpu = self.client.load_balanced_view()
        self.clu_func = {'hdbscan': self._hdbscan,
                         'dpgmm':   self._dpgmm }
        self.clu_status = clu_status
    

    ############### Backend Engine Func ####################
    def fit(self, clu_method, fet, clu, **kwargs):
        self.fet = fet
        self.clu = clu
        func = self.clu_func[clu_method]
        print(func)
        print(kwargs)
        self.clu.emit('report', state='BUSY')                   # state report --- before async non-blocking clustering 
        ar = self.cpu.apply_async(func, fet=fet, **kwargs)
        def get_result(ar):
            labels = ar.get()
            group_id = self.clu.fill(labels)
            self.clu.emit('report', state='READY')              # state report --- before async non-blocking clustering 
            print(group_id, 'cluster finished')
            self.clu_status[group_id] = True
        ar.add_done_callback(get_result)
    ########################################################    
        
    @staticmethod
    def _dpgmm(fet, n_comp=8, max_iter=400):
        from sklearn.mixture import BayesianGaussianMixture as DPGMM
        dpgmm = DPGMM(
            n_components=n_comp, covariance_type='full', weight_concentration_prior=1e-3,
            weight_concentration_prior_type='dirichlet_process', init_params="kmeans",
            max_iter=100, random_state=0, verbose=0, verbose_interval=10) # init can be "kmeans" or "random"
        dpgmm.fit(fet)
        label = dpgmm.predict(fet)
        return label
    
    @staticmethod
    def _hdbscan(fet, min_cluster_size=18, leaf_size=40, eom_or_leaf='eom'):
        import hdbscan
        import numpy as np
        hdbcluster = hdbscan.HDBSCAN(min_samples=5,
                     min_cluster_size=min_cluster_size, 
                     leaf_size=leaf_size,
                     gen_min_span_tree=True, 
                     algorithm='boruvka_kdtree',
                     core_dist_n_jobs=4,
                     prediction_data=False,
                     cluster_selection_method=eom_or_leaf) # eom or leaf 
        clusterer = hdbcluster.fit(fet.astype(np.float64))
#         probmatrix = hdbscan.all_points_membership_vectors(clusterer)
        return clusterer.labels_+1


class FET(object):
    """
    feature = FET(fet)
    fet: dictionary {group_id:fet[group_id], ...}
    fet[group_id]: n*m matrix, n is #samples, m is #features
    """
    def __init__(self, fet):
        self.fet = fet
        self.group  = []
        self.clu    = {}
        self.clu_status = {}
        self.backend = []
        self.npts = {}
        for _grp_id, _fet in self.fet.items():
            self.npts[_grp_id] = len(_fet)
            if len(_fet) > 0:
                self.group.append(_grp_id)
                self._reset(_grp_id)
                self.clu_status[_grp_id] = False
            else:
                self.group.append(_grp_id)
                self.clu_status[_grp_id] = None

        # exclude channels which no spikes 
        self.fetlen = fet[self.group[0]].shape[1]

        self.hdbscan_hyper_param = {'min_cluster_size': 18,
                                    'leaf_size': 40,
                                    'eom_or_leaf': 'eom'}

        self.dpgmm_hyper_param = {'max_n_clusters': 10,
                                  'max_iter':       300}

    # def set_backend(self, method='ipyparallel'):
    #     self.backend = cluster()

    def __getitem__(self, i):
        return self.fet[i]

    def __setitem__(self, i, _fet_array):
        self.fet[i] = _fet_array

    def remove(self, group, ids):
        self.fet[group] = np.delete(self.fet[group], ids, axis=0)
 

    def toclu(self, method='dpgmm', group_id='all', **kwargs):
        
        # clu_dict = {}

        '''
        1. When group_id is not provided, means parallel sorting on all groups
        '''
        if group_id is 'all':
            for i in range(len(self.group)):
                group_id = self.group[i]
                self.toclu(method, group_id, **kwargs)
                # clu_dict[group_id] = self.clu[group_id]
            # info('clustering for all groups with {} cpus'.format(njobs))
            # tic = time()
            # ### TODO ###
            # toc = time()
            # info('clustering finished, used {} sec'.format(toc-tic))

        ## 2. When group_id is provided, and background sorting (async and non-blocking) is required
        else:
            self.backend.append(cluster(self.clu_status))
            self.backend[-1].fit(method, self.fet[group_id], self.clu[group_id], **kwargs)


    def _reset(self, group_id):
        '''
        A new CLU is generated for targeted group, with all membership set to 0
        '''
        clu = CLU(np.zeros((self.fet[group_id].shape[0], )).astype(np.int64))
        self.clu[group_id]     = clu
        self.clu[group_id]._id = group_id
        '''
        The first registraion after born for every clu
        ''' 
        @clu.connect
        def on_cluster(*args, **kwargs):
            print(clu._id, clu.membership)
