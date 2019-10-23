import numpy as np
from sklearn.neighbors import KDTree
from tqdm import tqdm
from .Model import MainModel
from ..analysis import spk_time_to_scv 
from .View import MainView
from ..base import CLU
from ..utils import warning, conf
from ..utils.utils import Timer
from ..base.SPK import _transform
from ..fpga import xike_config
from ..analysis.place_field import place_field
from ..view import scatter_3d_view
from ..utils.conf import info 
from playground.view import maze_view


class controller(object):
    '''
    First stage:  (channel map in the prb file)
    >> from spiketag.fpga import xike_config
    >> fpga = xike_config(probe=prb)

    Second stage: (ch_ref and thres)
    >> ctrl = controller(fpga=True, **kwargs)     #1. load pre-recorded files
    >> th = ctrl.model.mua.get_threshold(beta=5.5)/float(2**13)
    >> ctrl.fpga.thres[:] = th
    >> plt.plot(th, '-o')
    >> ctrl.fpga.ch_ref[0:88] = 44.
    >> ctrl.fpga.ch_ref[88:156] = 137.
    >> ctrl.fpga.ch_ref[156:160] = 160.

    Third stage:  (sorting for spike model including transformer, vq and labels):
    >> ctrl = controller(fpga=True, **kwargs)     #1. load pre-recorded files
    >> ctrl.show()                                #2. sorting (label the `done` state)
    >> ctrl.compile()                             #3. FPGA transformer (y=a(Px+b)) and vq parameters (vqs and labels)
    '''
    def __init__(self, fpga=False, *args, **kwargs):

        self.model = MainModel(*args, **kwargs)
        self.prb   = self.model.probe
        self.view  = MainView(prb=self.prb, model=self.model)
        self.current_group = 0

        # place fields
        self.fields = {}
        for group_id in range(self.n_group):
            self.fields[group_id] = {}

        # vq
        self.vq = {}
        self.vq['points'] = {}
        self.vq['labels'] = {}
        self.vq['scores'] = {}
        self._vq_npts = 500  # size of codebook to download to FPGA, there are many codebooks


        if fpga is True:
            # initialize FPGA channel grouping
            # both ch_hash and ch_grpNo are configured
            # every channel has a `ch_hash` and a `ch_grpNo` 
            self.fpga = xike_config(probe=self.prb)  # this will automatically download the prb map into the FPGA
            
            
        @self.view.prb.connect
        def on_select(group_id, chs):
            # print(group_id, chs)
            self.current_group = group_id
            nspks = self.model.gtimes[self.current_group].shape[0]
            self.view.status_bar.setStyleSheet("color:red")
            self.view.status_bar.showMessage('loading group {}:{}. It contains {} spikes'.format(group_id, chs, nspks))
            self.show(group_id)
            self.view.status_bar.setStyleSheet("color:black")
            self.view.status_bar.showMessage('group {}:{} are loaded. It contains {} spikes'.format(group_id, chs, nspks))
            self.view.setWindowTitle("Spiketag: {} units".format(self.unit_done)) 


        @self.view.clu_view.clu_manager.connect
        def on_select(group_id):
            self.view.prb_view.select(group_id)

        @self.view.clu_view.clu_manager.connect
        def on_backend(method, **params):
            self.recluster(method=method, **params)


        @self.view.spkview.event.connect
        def on_show(content):
            if content == 'ephys_full':
                if self.clu.selectlist.shape[0] == 1:
                    _time = self.model.gtimes[self.current_group][self.clu.selectlist[0]]/self.prb.fs
                    _span = 0.4
                    self.model.mua.show(self.prb.chs, span=_span, time=_time)
                    _highlight_point = int(_span*self.prb.fs)
                    #  chs, timelist, colorlist=None, mask_others=False
                    _cluNo = self.clu.membership[self.clu.selectlist][0]
                    _highlight_color = np.array(self.view.spkview.palette[_cluNo])
                    _highlight_color = np.append(_highlight_color, 1)
                    self.model.mua.wview.highlight(chs=self.prb[self.current_group], 
                                                   timelist=[[_highlight_point-10, _highlight_point+15]],
                                                   colorlist=_highlight_color, mask_others=True)


        @self.view.spkview.event.connect
        def on_magnet(sink_id, source_id, k):
            # print('sink_id {}, k {}'.format(sink_id, k))
            self.transfer(sink_id, source_id, k)

        @self.view.spkview.event.connect
        def on_trim(source_id, k):
            self.trim(source_id, k)
            
        @self.view.spkview.event.connect
        def on_dismiss(source_id):
            self.dismiss(source_id)
        
        @self.view.spkview.event.connect
        def on_build_vq():
            self.build_vq()

        @self.view.spkview.event.connect
        def on_clip(idx):
            idx = np.array(idx)
            print('delete {} spikes'.format(idx.shape))
            self.delete_spk(spk_idx=idx)
            self.update_view()

        @self.view.spkview.event.connect
        def on_recluster(method, **params):
            self.recluster(method=method, **params)

        @self.view.spkview.event.connect
        def on_refine(method, args):
            self.refine(method, args)

        @self.view.ampview.event.connect
        def on_refine(method, args):
            self.refine(method, args)

        @self.view.ampview.event.connect
        def on_clip(thres):
            idx = np.where(self.model.spk[self.current_group][:,8,:].min(axis=1)>thres)[0]
            print('delete {} spikes'.format(idx.shape))
            self.delete_spk(spk_idx=idx)

        @self.view.traceview.event.connect
        def on_view_trace():
            if self.current_group>min(self.prb.grp_dict.keys()) and self.current_group<max(self.prb.grp_dict.keys()):
                vchs = np.hstack((self.prb[self.current_group-1], self.prb[self.current_group], self.prb[self.current_group+1]))
            elif self.current_group == min(self.prb.grp_dict.keys()):
                vchs = np.hstack((self.prb[self.current_group], self.prb[self.current_group+1]))
            elif self.current_group == max(prb.grp_dict.keys()):
                vchs = np.hstack((self.prb[self.current_group-1], self.prb[self.current_group]))
            if len(self.view.spkview.selected_spk) == 1:
                current_time = self.model.mua.spk_times[self.current_group][self.view.spkview.selected_spk]/self.model.mua.fs
                self.model.mua.show(time = current_time, chs=vchs)

    @property
    def current_group(self):
        return self._current_group

    @current_group.setter
    def current_group(self, group_id):
        self._current_group = group_id
        # self.show(group_id)

    @property
    def n_group(self):
        return len(self.prb.grp_dict.keys())


    @property
    def spk(self):
        return self.model.spk[self.current_group]   # spk ndarray

    @property
    def fet(self):
        return self.model.fet[self.current_group]   # fet ndarray

    @property
    def clu(self):
        return self.model.clu[self.current_group]   # CLU

    @property
    def nclu(self):
        return len(self.model.clu[self.current_group].index_id)

    @property
    def group_done(self):
        self._group_done = np.where(np.array(self.model.clu_manager.state_list)==3)[0]
        print('{} groups done'.format(self._group_done.shape[0]))
        return self._group_done

    @property
    def unit_done(self):
        return sum(list(self.spk_times_all[1].values()))

    @property
    def selected_spk_times(self):
        return self.model.gtimes[self.current_group][self.clu.selectlist]/self.model.mua.fs

    def get_spk_times(self, group_id=-1, cluster_id=1):
        if group_id==-1:
            group_id = self.current_group
        idx = self.model.clu[group_id][cluster_id]
        spk_times = self.model.gtimes[group_id][idx]/float(self.model.mua.fs)
        return spk_times


    @property
    def spk_time(self):
        self._spk_time = {}
        cluNo = self.model.clu[self.current_group].index.keys()
        for i in cluNo:
            self._spk_time[i] = self.get_spk_times(cluster_id=i)
        return self._spk_time


    @property
    def spk_times_all(self):
        self._spk_time_all = {}
        self._nclu_all = {}
        for grp_id in range(self.prb.n_group):
            if self.model.clu_manager.state_list[grp_id]==3:
                _spk_times = {}
                cluNo = self.model.clu[grp_id].index.keys()
                for i in cluNo:
                    if i>0:  # first one is noise
                        _spk_times[i-1] = self.get_spk_times(grp_id, cluster_id=i)  # start from 0
                self._spk_time_all[grp_id] = _spk_times
                self._nclu_all[grp_id] = len(cluNo) - 1 # first one is noise
        return self._spk_time_all, self._nclu_all


    @property
    def spk_times_all_in_one(self):
        spk_times, _ = self.spk_times_all
        i = 0
        self._spk_times_all_in_one = {}
        for grp_id, _spk_times in spk_times.items():
            for cluNo, _spk_time in _spk_times.items():
                self._spk_times_all_in_one[i] = _spk_time
                i += 1
        return self._spk_times_all_in_one

    @property
    def spk_times_all_in_one_array(self):
        return np.array(list(self.spk_times_all_in_one.values()))

    def spk_count_vector(self, bin_size=33.33, ts=None):
        if ts is None:
            ts = self.model.pc.ts
        self._scv = spk_time_to_scv(self.spk_times_all_in_one, delta_t=bin_size, ts=ts)
        return self._scv

    def delete_spk(self, spk_idx):
        i = self.current_group
        self.model.mua.spk_times[i] = np.delete(self.model.mua.spk_times[i], spk_idx, axis=0)
        self.model.spk[i] = np.delete(self.model.spk[i], spk_idx, axis=0)
        self.model.fet[i] = self.model.spk.tofet(i, method=self.model.fet_method, ncomp=self.model._fetlen)
        self.model.clu[i].delete(spk_idx)
        self.view.prb_view.select(i) 
        

    def recluster(self, method, **params):
        group_id = self.current_group
        self.model.cluster(group_id, method, **params)

    def gmm_cluster(self, N=None):
        '''
        before running this, make sure the first cluster is noise, and the rest clusters are correctly positioned
        '''
        from sklearn.mixture import GaussianMixture as GMM
        if N is None:
            N = self.clu.nclu - 1
            means_init = []
            for i in range(N):
                means_init.append(self.fet[self.clu[i+1]].mean(axis=0))
            means_init = np.vstack(means_init)
            gmm = GMM(n_components=N, covariance_type='full', means_init=means_init)
        else:
            gmm = GMM(n_components=N, covariance_type='full')
        gmm.fit(self.fet)
        label = gmm.predict(self.fet)
        self.clu.membership = label
        self.clu.__construct__()
        self.clu.emit('cluster')
        return gmm


    def dpgmm_cluster(self, max_n_clusters = 30, max_iter=300):
        from sklearn.mixture import BayesianGaussianMixture as DPGMM
        dpgmm = DPGMM(
            n_components=max_n_clusters, covariance_type='full', weight_concentration_prior=1e-3,
            weight_concentration_prior_type='dirichlet_process', init_params="kmeans",
            max_iter=max_iter, random_state=0, verbose=1, verbose_interval=10) # init can be "kmeans" or "random"
        dpgmm.fit(self.fet)
        label = dpgmm.predict(self.fet)
        self.clu.membership = label
        self.clu.__construct__()
        self.clu.emit('cluster')
        return dpgmm                

    def kmm_cluster(self, N=100):
        from sklearn.cluster import MiniBatchKMeans
        kmm = MiniBatchKMeans(N)
        kmm.fit(self.fet)
        label = kmm.predict(self.fet)
        self.clu.membership = label
        self.clu.__construct__()
        self.clu.emit('cluster')
        return kmm    


    def update_view(self):
        i = self.current_group
        self.view.set_data(i, self.model.mua, self.model.spk[i], self.model.fet[i], self.model.clu[i])
        self.view.setWindowTitle("Spiketag: {} units".format(self.unit_done)) 


    def sort(self, clu_method='hdbscan'):
        if clu_method == 'bg_cluster':
            self.model.sort(clu_method='reset')
            self.run_bg_clustering()
        else:
            self.model.sort(clu_method=clu_method)            


    def run_bg_clustering(self, method='parallel'):
        if method == 'parallel':
            self.bg_cluster = DPGMM_IPY(cpu_No=self.prb.n_group) 
            for group_id in range(self.prb.n_group):
                self.bg_cluster.set_data(group_id, {'data':self.model.fet[group_id]}) 
            self.bg_cluster.run_all(nclu_max=7)
        if method == 'sequential':
            pass
            #TODO

    def bg_dpgmm(self, max_n_clusters=8):
        self.bg_cluster = DPGMM_IPY(cpu_No=self.prb.n_group) 
        cpu_id = self.bg_cluster.cpu_available.min()
        self.bg_cluster[cpu_id]['data'] = self.model.fet[self.current_group]
        self.bg_cluster[cpu_id].execute('label = dpgmm({})'.format(max_n_clusters))
        # labels = self.bg_cluster[cpu_id]['label']
        # self.clu.fill(labels)

    def show(self, group_id=None):
        if group_id is None:
            self.update_view()
            self.view.show()
        else:
            self.view.set_data(group_id, self.model.mua, self.model.spk[group_id], self.model.fet[group_id], self.model.clu[group_id])
            self.view.show()

    def save(self, filename):
        self.model.tofile(filename)

    #####################################
    ####  sorting improvement
    #####################################
    def select_noise(self, thres=0.3):
        noise_leve = []
        group_id = self.current_group
        for i in range(self.model.spk[group_id].shape[0]):
            current_pts = self.model.mua.spk_times[group_id][i]
            noise_leve.append(self.model.mua.data[current_pts, self.prb.chs].mean()/self.model.mua.data[current_pts, self.prb[group_id]].mean())
        noise_leve = np.array(noise_leve)
        idx = np.where(abs(noise_leve)>thres)[0]
        self.clu.select(idx)


    def refine(self, method, args):
        # method is time_threshold, used to find burst
        # args here is the time_threshold for bursting or a sequential firing for single neuron
        time_thr = args*self.prb.fs
        # print time_thr
        sequence = np.where(np.diff(self.model.gtimes[self.current_group][np.unique(self.clu.selectlist)])<time_thr)[0]
        idx_tosel = np.hstack((sequence, sequence-1)) + 1
        spk_tosel = np.unique(self.clu.selectlist)[idx_tosel]
        self.clu.select(spk_tosel)


    def transfer(self, sink_clu_id, source_clu_id, k=1):
        '''
        transfer source to sink the N*sink.shape[0] NN pts
        '''
        # if source_clu_id == sink_clu_id:
        #     source_clu_id = 0
        source = self.fet[self.clu[source_clu_id]]
        sink   = self.fet[self.clu[sink_clu_id]]
        KT = KDTree(source)
        nn_ids = KT.query(sink, k, dualtree=True)[1].ravel()
        global_nn_ids = self.clu.local2global({source_clu_id:nn_ids})
        collective_ids = np.hstack((global_nn_ids, self.clu[sink_clu_id]))
        self.clu.select(collective_ids)
        # sink = np.append(sink, source[nn_ids], axis=0)
        # source = np.delete(source, nn_ids, axis=0)
        # return source, sink
        # self.model.fet[self.current_group]

    def trim(self, source_clu_id, k):
        '''
        trim a source cluster
        '''
        pts = k*30
        source = self.fet[self.clu[source_clu_id]]
        KT = KDTree(source)
        dis, _ = KT.query(source, 10, dualtree=True) # use 10 pts to calculate average distance
        distance = dis[:, 1:].mean(axis=1) # the first column is the distance to each point itself, which is 0
        low_density_idx = self.clu.local2global({source_clu_id:np.argsort(distance)[::-1][:pts]})
        self.clu.select(low_density_idx)

    # cluNo is a noisy cluster, usually 0, assign it's member to other clusters
    # using knn classifier: for each grey points:
    # 1. Get its knn in each other clusters (which require KDTree for each cluster)
    # 2. Get the mean distance of these knn points in each KDTree (Cluster)
    # 3. Assign the point to the closet cluster
    def dismiss(self, cluNo=0, k=15):
        # get the features of targeted cluNo
        #  X = self.fet[self.clu[cluNo]]
        # classification on these features
            lables_X = self.model.predict(self.current_group, self.clu[cluNo], method='knn', k=15)
        # reconstruct current cluster membership
            self.clu.membership[self.clu[cluNo]] = lables_X
            if self.clu.membership.min()>0:
                self.clu.membership -= 1
            self.clu.__construct__()
            self.update_view()


    def get_fields(self, group_id=-1, kernlen=21, std=3):
        if group_id == -1:
            group_id = self.current_group
        # spk_time = {}
        # for clu_id in self.model.clu[group_id].index.keys():
        #     spk_time[clu_id] = self.get_spk_times(group_id, clu_id)
        self.model.pc.get_fields(self.spk_times, kernlen, std)
        self.fields[group_id] = self.model.pc.fields


    def plot_fields(self, N=4, size=3):
        self.model.pc.plot_fields(N, size)


    def field_reorder(self, group_id=-1, thres=1):
        '''
        1. merge clusters based on its spatial information (bits), merge if lower than thres bits
        2. reorder clusters according to spatial bits
        '''
        if group_id == -1:
            group_id = self.current_group
        self.model.pc.get_fields(self.spk_time)
        self.model.pc.rank_fields('spatial_bit_smoothed_spike')
        # 1. merge low bits clusters
        low_spatial_clus = np.where(self.model.pc.metric['spatial_bit_smoothed_spike']<thres)[0]
        self.clu.merge(low_spatial_clus)
        # 2. reorder clusters according to bits (0: lowest noise, 1: highest, 2: second highest ... )
        self.model.pc.get_fields(self.spk_time)
        self.model.pc.rank_fields('spatial_bit_smoothed_spike')        
        sorted_idx = np.roll(self.model.pc.sorted_fields_id, 1) # put noise first as grey cluster
        self.clu.reorder(sorted_idx)


    def replay(self, maze_folder, neuron_id, replay_speed=10, replay_start_time=0., mirror=True, spk_time=None):
        self.nav_view = maze_view()
        self.nav_view.load_maze(maze_folder+'maze_2d.obj',
                                maze_folder+'maze_2d.coords',
                                mirror=mirror)

        t, pos = self.model.ts.copy(), self.model.pos.copy()
        if mirror:
            pos[:,1] = -pos[:,1]
        pos = self.nav_view._to_jovian_coord(pos).astype(np.float32)

        self.nav_view.replay_t = t
        self.nav_view.replay_pos = pos
        if spk_time is None:
            self.nav_view.load_neurons(spk_time=self.spk_time)
        else:
            self.nav_view.load_neurons(spk_time=spk_time)

        self.nav_view.neuron_id = neuron_id
        self.nav_view.replay_speed = replay_speed
        self.nav_view.replay_time = replay_start_time

        self.nav_view.view.camera.azimuth = 0.
        self.nav_view.view.camera.elevation = -90.
        self.nav_view.show()


    ##### FPGA related #####
    def set_threshold(self, beta=4.0):
        self.fpga.thres[:] = self.model.mua.get_threshold(beta)
        for ch in self.prb.mask_chs:
            self.fpga.thres[ch] = -5000. 
        for ch in self.prb.bad_chs:
            self.fpga.thres[ch] = -5000.

    def _transform(self, x, P, shift, scale):
        return _transform(x, P, shift, scale) 

    def construct_transformer(self, group_id, ndim=4):
        _pca_comp, _shift, _scale = self.model.construct_transformer(group_id, ndim)
        return _pca_comp, _shift, _scale      

    def set_transformer(self, group_id, random=False):
        _pca_comp, _shift, _scale = self.construct_transformer(group_id=group_id, ndim=4)
        self.fpga._config_FPGA_transformer(grpNo=group_id, P=_pca_comp, b=_shift, a=_scale) 
        # assert(np.allclose(self.fpga.pca[i], _pca_comp, atol=1e-3))
        # assert(np.allclose(self.fpga.shift[i], _shift,  atol=1e-3))
        # assert(np.allclose(self.fpga.scale[i], _scale,  atol=1e-3))
            
    def check_transformer(self, group_id):
        _spk = self.model.spk[group_id].T.reshape(76, -1).T
        _fet = self._transform(_spk, self.fpga.pca[group_id], 
                                     self.fpga.shift[group_id], 
                                     self.fpga.scale[group_id])
        from spiketag.view import scatter_3d_view
        _fetview = scatter_3d_view()
        _fetview.set_data(_fet)
        _fetview.show()

    def build_vq(self, grp_id=None, n_dim=4, n_vq=None, show=True, method='proportional', fpga=False):
        import warnings
        warnings.filterwarnings('ignore')
        # get the vq and vq labels
        from sklearn.cluster import MiniBatchKMeans

        if grp_id is None:
            grp_id = self.current_group

        vq = []
        if n_vq is None:
            if method == 'proportional':
                k = self.model.nspk_per_clu[grp_id].sum() / self._vq_npts
                n_vq = np.around(self.model.nspk_per_clu[grp_id] / k).astype(np.int32)
            elif method == 'equal':
                k = int(self._vq_npts/self.model.clu[grp_id].nclu)
                n_vq = np.ones((self.model.clu[grp_id].nclu,)).astype(np.int32) * k
            err = n_vq.sum() - self._vq_npts
            n_vq[-1] -= err
            assert(n_vq.sum()==500)

        for _clu_id in tqdm(self.model.clu[grp_id].index_id):
            km = MiniBatchKMeans(n_vq[_clu_id])
            X = self.model.fet[grp_id][self.model.clu[grp_id].index[_clu_id]][:,:n_dim]
            km.fit(X)
            vq.append(km.cluster_centers_)

        self.vq['points'][grp_id] = np.vstack(vq)
        self.vq['labels'][grp_id] = self._predict(grp_id, np.vstack(vq), n_dim)
        self.vq['scores'][grp_id] = self._validate_vq(grp_id, n_dim)
        info('group {}: accuracy:{}'.format(grp_id, self.vq['scores'][grp_id]))
        assert(self.vq['labels'][grp_id].max() == self.model.clu[grp_id].nclu - 1)

        if show:
            self.vq_view = scatter_3d_view()
            self.vq_view._size = 5
            self.vq_view.set_data(self.vq['points'][grp_id], 
                                  self.vq['labels'][grp_id])
            self.vq_view.transparency = 0.9
            self.vq_view.show()
        
        if fpga:
            self.fpga.vq[grp_id] = self.vq['points'][grp_id]
            self._update_labels()
            for grpNo in tqdm(self.vq['labels'].keys(), desc='compile to fpga'):
                self.fpga.label[grpNo] = self.vq['labels'][grpNo]


    def _update_labels(self):
        '''
        This update the global labels after each time the vq['labels'] is updated by any group
        '''
        base_label = 0  # start from zero
        for _grpNo, _labels in self.vq['labels'].items():
            _labels += base_label
            _labels[_labels==base_label] = 0
            if _labels.max() != 0:
                base_label = _labels.max() # base_label gets up each group by its #unit

        
    def _validate_vq(self, grp_id, n_dim=4):
        from sklearn.neighbors import KNeighborsClassifier as KNN
        knn = KNN(n_neighbors=1)
        knn.fit(self.vq['points'][grp_id], self.vq['labels'][grp_id])
        _score = knn.score(self.model.fet[grp_id][:,:n_dim], self.model.clu[grp_id].membership)
        return _score

    def _predict(self, grp_id, points, n_dim=4):
        self.model.construct_kdtree(grp_id, n_dim)
        d = []
        for _kd in self.model.kd.keys():
            tmp = _kd.query(points, 10)[0]
            d.append(tmp.mean(axis=1))
        d = np.vstack(np.asarray(d))
        labels = np.asarray(list(self.model.kd.values()))[np.argmin(d, axis=0)]
        return labels

    def set_vq(self, vq_method='proportional'):
        # step 1: set FPGA transfomer and build vq 
        for grp_id in range(self.prb.n_group):  # set_vq condition for a group: at least 500 spikes and in a `done` state
            if self.model.gtimes[grp_id].shape[0] > 500 and self.model.clu_manager.state_list[grp_id]==3:
                self.set_transformer(group_id=grp_id)
                self.build_vq(grp_id=grp_id, show=False, method=vq_method)
            else:
                pass

        # step 2: change labels such that each group has a different range that no overlapping
        self._update_labels()

        # step 3: set FPGA vq
        for grpNo in tqdm(self.vq['points'].keys(), desc='compile to fpga'):
            x = self.vq['points'][grpNo]
            y = self.vq['labels'][grpNo]
            self.fpga.vq[grpNo]    = x
            self.fpga.label[grpNo] = y
            # print('group {} vq configured with shape {}'.format(grpNo, x.shape))

    def reset_vq(self):
        # step 1: set FPGA transfomer
        for grp_id in range(self.prb.n_group):
            self.fpga.scale[grp_id] = 0  # this will ban the tranformer and check fpga.transformer status
            if self.model.gtimes[grp_id].shape[0] > 500 and self.model.clu_manager.state_list[grp_id]==3:
                self.set_transformer(group_id=grp_id)
                self.fpga.label[grp_id] = np.zeros((500,))

    def compile(self, vq_method='proportional'):
        self.reset_vq()
        self.set_vq(vq_method)
        self.fpga.n_units = self.unit_done
        print('FPGA is compiled')
        print('{} units are ready for real-time spike assignment'.format(self.fpga.n_units))
