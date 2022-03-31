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
                         'dpgmm':   self._dpgmm,
                         'kmeans':  self._kmeans }
        self.clu_status = clu_status
    
    ############### Backend Engine Func ####################
    def fit(self, method, fet, clu, mode, minimum_spks=80, **kwargs):
        self.fet = fet
        self.clu = clu
        if self.fet.shape[0] >= minimum_spks:
            func = self.clu_func[method]
            self.clu.emit('report', state='BUSY')                   # state report --- before async non-blocking clustering 
            if mode == 'blocking':
                self.cpu.block = True
                labels = self.cpu.apply(func, fet=fet, **kwargs)
                group_id = self.clu.fill(labels)
                self.clu.emit('report', state='READY') 
                self.clu_status[group_id] = True
            elif mode == 'non-blocking':
                ar = self.cpu.apply_async(func, fet=fet, **kwargs)
                def get_result(ar):
                    labels = ar.get()
                    group_id = self.clu.fill(labels)
                    self.clu.emit('report', state='READY')              # state report --- before async non-blocking clustering 
                    # print(group_id, 'cluster finished')
                    self.clu_status[group_id] = True
                ar.add_done_callback(get_result)
        else:
            self.clu.emit('report', state='NONE')
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
    def _kmeans(fet, n_comp=8, max_iter=400):
        from sklearn.cluster import MiniBatchKMeans
        kmeans = MiniBatchKMeans(n_clusters=n_comp,
                                 max_no_improvement=20,
                                 random_state=0,
                                 batch_size=6)
        label = kmeans.fit_predict(fet)
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

    def toclu(self, method='dpgmm', mode='non_blocking', minimum_spks=80, **kwargs):
        self.clustering_mode = mode
        for i in range(len(self.group)):
            group_id = self.group[i]
            self._toclu(method, group_id, mode, minimum_spks, **kwargs)

    def _toclu(self, method, group_id, mode, minimum_spks=80, **kwargs):
        self.backend.append(cluster(self.clu_status))
        self.backend[-1].fit(method = method, 
                             fet = self.fet[group_id], 
                             clu = self.clu[group_id], 
                             mode = mode,
                             minimum_spks = minimum_spks,
                             **kwargs)

    def reset(self):
        for g in self.group:
            self._reset(g)

    def _reset(self, group_id):
        '''
        A new CLU is generated for targeted group, with all membership set to 0
        '''
        clu = CLU(np.zeros((self.fet[group_id].shape[0], )).astype(np.int64))
        self.clu[group_id]     = clu
        self.clu[group_id]._id = group_id
        ### The first registraion after born for every clu
        # ! uncomment below three lines if you want to use the first registration for debugging
        # @clu.connect
        # def on_cluster(*args, **kwargs):
        #     print(clu._id, clu.membership)

    @property
    def nclus(self):
        self._nclus = []
        for i in self.group:
            n = self.clu[i].nclu
            self._nclus.append(n)
        self._nclus = np.array(self._nclus) - 1
        return self._nclus

    def assign_clu_global_labels(self):
        '''
        assign global labels to each clu (fet.clu[group_id].membership_global)
        while return a look up table {group_id: {local_label:global_label}}
        '''
        base = 0
        global_label_lut = {} # a look up table {local label : global label} 
        for g in self.group:
            label_dict = {}
            old_labels = np.unique(self.clu[g].membership)
            new_labels = np.unique(self.clu[g].membership) + base
            base = max(max(new_labels), base)
            new_labels[new_labels == new_labels.min()] = 0  # 0 won't change
    #         print(g, new_labels)
            for i, j in zip(old_labels, new_labels):
                label_dict[i] = j
            global_label_lut[g] = label_dict
            # assign global labels
            self.clu[g].membership_global = np.vectorize(label_dict.get)(self.clu[g].membership)
        return global_label_lut
