import numpy as np
from .Model import MainModel
from .View import MainView
from ..base import CLU
from ..utils import warning, conf
from ..utils.utils import Timer
from ..base.SPK import _transform
from ..fpga import xike_config
from ..analysis.place_field import place_field
from ..view import scatter_3d_view
from playground.view import maze_view


class controller(object):
    def __init__(self, fpga=False, *args, **kwargs):

        self.model = MainModel(*args, **kwargs)
        self.prb   = self.model.probe
        self.view  = MainView(self.prb)
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
        self._vq_npts = 100  # size of codebook to download to FPGA, there are many codebooks


        if fpga is True:
            # initialize FPGA channel grouping
            # both ch_hash and ch_grpNo are configured
            # every channel has a `ch_hash` and a `ch_grpNo` 
            self.fpga = xike_config(probe=self.prb, offset_value=32)
            
        @self.view.prb.connect
        def on_select(group_id, chs):
            print(group_id, chs)
            self.current_group = group_id
            self.show(group_id)

        @self.view.ampview.clip.connect
        def on_clip(thres):
            idx = np.where(self.model.spk[self.current_group].min(axis=1).min(axis=1)>thres)[0]
            print('delete {} spikes'.format(idx.shape))
            self.delete_spk(spk_idx=idx)
            self.recluster()

        @self.view.spkview.event.connect
        def on_clip(idx):
            idx = np.array(idx)
            print('delete {} spikes'.format(idx.shape))
            self.delete_spk(spk_idx=idx)
            self.update_view()

        @self.view.spkview.event.connect
        def on_recluster():
            self.recluster()

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
    def spk_time(self):
        self._spk_time = {}
        cluNo = self.model.clu[self.current_group].index.keys()
        for i in cluNo:
            self._spk_time[i] = self.get_spk_times(cluster_id=i)
        return self._spk_time
        # for group_id in range(self.n_group):
        #     self._spk_times[group_id] = {}
        #     for clu_id in self.model.clu[group_id].index.keys():
        #         self._spk_times[group_id][clu_id] = self.get_spk_times(group_id, i)

    def get_spk_times(self, group_id=-1, cluster_id=1):
        if group_id==-1:
            group_id = self.current_group
        idx = self.model.clu[group_id][cluster_id]
        spk_times = self.model.gtimes[group_id][idx]/float(self.model.mua.fs)
        return spk_times

    def delete_spk(self, spk_idx):
        i = self.current_group
        self.model.mua.spk_times[i] = np.delete(self.model.mua.spk_times[i], spk_idx, axis=0)
        self.model.spk[i] = np.delete(self.model.spk[i], spk_idx, axis=0)
        self.model.fet[i] = self.model.spk._tofet(i, method='pca')
        self.model.clu[i].delete(spk_idx)

    def recluster(self, fall_off_size=None):
        i = self.current_group
        if fall_off_size is None:
            self.model.cluster(group_id=i, method='hdbscan', fall_off_size=self.model._fall_off_size)
        else:
            self.model.cluster(group_id=i, method='hdbscan', fall_off_size=fall_off_size)
        self.update_view()

    def update_view(self):
        i = self.current_group
        self.view.set_data(i, self.model.mua, self.model.spk[i], self.model.fet[i], self.model.clu[i])

    def sort(self):
        self.model.sort()

    def show(self, group_id=None):
        if group_id is None:
            self.update_view()
            self.view.show()
        else:
            self.view.set_data(group_id, self.model.mua, self.model.spk[group_id], self.model.fet[group_id], self.model.clu[group_id])
            self.view.show()

    def save(self, filename):
        self.model.tofile(filename)


    #### Analysis ####
    # def load_logfile(self, logfile, session_id=0, v_cutoff=5):
    #     self.pc = place_field(logfile=logfile, session_id=session_id, v_cutoff=v_cutoff)


    def get_fields(self, group_id=-1, kernlen=21, std=3):
        if group_id == -1:
            group_id = self.current_group
        spk_time = {}
        for clu_id in self.model.clu[group_id].index.keys():
            spk_time[clu_id] = self.get_spk_times(group_id, clu_id)
        self.model.pc.get_fields(spk_time, kernlen, std)
        self.fields[group_id] = self.model.pc.fields

    def plot_fields(self, N=4, size=3):
        self.model.pc.plot_fields(N, size)


    def replay(self, maze_folder, neuron_id, replay_speed=10, replay_start_time=0.):
        self.nav_view = maze_view()
        self.nav_view.load_maze(maze_folder+'maze_2d.obj',
                                maze_folder+'maze_2d.coords',
                                mirror=False)

        t, pos = self.model.ts, self.model.pos
        pos = self.nav_view._to_jovian_coord(pos).astype(np.float32)
        self.nav_view.replay_t = t
        self.nav_view.replay_pos = pos
        self.nav_view.load_neurons(spk_time=self.spk_time)

        self.nav_view.neuron_id = neuron_id
        self.nav_view.replay_speed = replay_speed
        self.nav_view.replay_time = replay_start_time

        self.nav_view.view.camera.azimuth = 0.
        self.nav_view.view.camera.elevation = 90.
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

    def set_transformer(self):
        for i in self.model.groups:
            _pca_comp, _shift, _scale = self.construct_transformer(group_id=i, ndim=4)
            self.fpga._config_FPGA_transformer(grpNo=i, P=_pca_comp, b=_shift, a=_scale) 
            assert(np.allclose(self.fpga.pca[i], _pca_comp, atol=1e-3))
            assert(np.allclose(self.fpga.shift[i], _shift,  atol=1e-3))
            assert(np.allclose(self.fpga.scale[i], _scale,  atol=1e-3))
            

    def build_vq(self, all=False):
        import warnings
        warnings.filterwarnings('ignore')
        # get the vq and vq labels
        from sklearn.cluster import MiniBatchKMeans
        km = MiniBatchKMeans(self._vq_npts)
        X = self.fet
        km.fit(X)
        self.points = km.cluster_centers_   # these are vq
        self.labels = self._predict(self.points) # these are vq labels
        self.scores = self._validate_vq()

        self.vq['points'][self.current_group] = self.points
        self.vq['labels'][self.current_group] = self.labels
        self.vq['scores'][self.current_group] = self.scores

        self.vq_view = scatter_3d_view()
        self.vq_view._size = 5
        self.vq_view.set_data(self.points, CLU(self.labels))
        self.vq_view.transparency = 0.8
        self.vq_view.show()


    def _validate_vq(self):
        from sklearn.neighbors import KNeighborsClassifier as KNN
        knn = KNN(n_neighbors=1)
        knn.fit(self.points, self.labels)
        return knn.score(self.fet, self.clu.membership)


    def _predict(self, points):
        self.model.construct_kdtree(self.current_group)
        d = []
        for _kd in self.model.kd.iterkeys():
            tmp = _kd.query(points, 10)[0]
            d.append(tmp.mean(axis=1))
        d = np.vstack(np.asarray(d))
        labels = np.asarray(self.model.kd.values())[np.argmin(d, axis=0)]
        return labels







class Sorter(object):
	"""docstring for Sorter"""
	def __init__(self, *args, **kwargs):
		self.model = MainModel(*args, **kwargs)
		self.view  = MainView(self.model.probe.n_group, 
                                      self.model.probe.get_chs,
                                      self.model.probe.fs,
                                      self.model.spk.spklen,
                                      self.model.mua.scale)

                # register action for param view
		self.view.param_view.signal_group_changed.connect(self.update_group)
		self.view.param_view.signal_get_fet.connect(self.update_fet)
		self.view.param_view.signal_recluster.connect(self.update_clu)
		self.view.param_view.signal_refine.connect(self.refine)
		self.view.param_view.signal_build_vq.connect(self.build_vq)
		self.view.param_view.signal_apply_to_all.connect(self.check_apply_to_all)
                self.view.param_view.signal_trace_view_zoom.connect(self.trace_view_zoom)

                #register action for spike view
                self.view.spk_view.events.model_modified.connect(self.on_model_modified)

		self.showmua = False
		self.current_group = 0
		self.vq = {}
		self.vq['points'] = {}
		self.vq['labels'] = {}
		self.vq['scores'] = {}
		self._vq_npts = 100  # size of codebook to download to FPGA, there are many codebooks

	def check_apply_to_all(self):
		self.apply_to_all = self.view.param_view._apply_to_all

	def set_model(self, model):
		if model is not None:
			self.model = model
	        if self.current_group not in self.model.clu:
                    # TODO: find some way to popup this information
                    warning(" group {} has no spikes! ".format(self.current_group))
                elif self.showmua is False:
                    self.view.set_data(data=self.model.mua.data[:,self.model.probe[self.current_group]], 
                                       spk=self.model.spk[self.current_group], 
				       fet=self.model.fet[self.current_group], 
				       clu=self.model.clu[self.current_group], 
                                       spk_times=self.model.gtimes[self.current_group])
		else:
                    self.view.set_data(gdata=self.model.mua.data[:,self.model.probe[self.current_group]], 
                                       mua=self.model.mua, 
                                       spk=self.model.spk[self.current_group], 
                                       fet=self.model.fet[self.current_group], 
                                       clu=self.model.clu[self.current_group],
                                       spk_times=self.model.gtimes(self.current_group))

	def refresh(self):
		self.set_model(self.model)


	def run(self):
	        self.update_group()
	        self.view.show()

	def save(self):
		self.model.tofile()


	def tofile(self, filename):
		# use model.tofile
		# Will update the manmual modification to spktag structured numpy array
		self.model.tofile(filename)

	@property
	def selected(self):
		self._selected = self.view.spk_view.selected_spk
		return self._selected
	
	@property
	def spk(self):
		return self.model.spk[self.current_group]   # ndarray

	@property
	def fet(self):
		return self.model.fet[self.current_group]   # ndarray

	@fet.setter
	def fet(self, fet_value):
		self.model.fet.fet[self.current_group] = fet_value
		self.refresh()

	@property
	def clu(self):
		return self.model.clu[self.current_group]   # CLU

	@clu.setter
	def clu(self, clu_membership):
		self.clu.membership = clu_membership
		self.clu.__construct__()
		self.refresh()

        @property
        def time(self):
                return self.model.gtimes[self.current_group]

        def show_context(self, chs):
            wview = wave_view(self.model.mua.data[self.time[self.selected]-2000:self.time[self.selected]+2000, :], fs=self.model.probe.fs, ncols=1, chs=chs)
            wview.show()

        def show_mua(self, chs, spks=None):
            '''
            spks: (t,ch) encodes pivital
            array([[  37074,   37155,   37192, ..., 1602920, 1602943, 1602947],
                   [     58,      49,      58, ...,      58,      75,      77]], dtype=int32)
            '''
            wview = wave_view(self.model.mua.data, fs=self.model.probe.fs, ncols=1, chs=chs, spks=spks)
            wview.show()

	def update_group(self):
		# ---- update chosen ch and get cluster ----
		self.current_group = self.view.param_view.group.value()
		# print '{} selected'.format(self.ch)
		self.set_model(self.model)


	# cluNo is a noisy cluster, usually 0, assign it's member to other clusters
	# using knn classifier: for each grey points:
	# 1. Get its knn in each other clusters (which require KDTree for each cluster)
	# 2. Get the mean distance of these knn points in each KDTree (Cluster)
	# 3. Assign the point to the closet cluster
	def refine(self, cluNo=0, k=30):
		# get the features of targeted cluNo
		#  X = self.fet[self.clu[cluNo]]
		# classification on these features
                lables_X = self.model.predict(self.current_group, self.clu[cluNo], method='knn', k=10)
		# reconstruct current cluster membership
		self.clu.membership[self.clu[cluNo]] = lables_X
		if self.clu.membership.min()>0:
			self.clu.membership -= 1
		self.clu.__construct__()
		self.refresh()


	# cluNo is a good cluster, absorb its member from other clusters
	def absorb(self, cluNo=0):
		# TODO: 
		pass


	def update_fet(self):
		fet_method = str(self.view.param_view.fet_combo.currentText())
                # print fet_method
		fet_len    = self.view.param_view.fet_No.value()
                # print fet_len
		self.model.fet[self.current_group] = self.model.spk.tofet(group_id=self.current_group, method=fet_method, ncomp=fet_len)
		self.refresh()


	def update_clu(self):
		clu_method = str(self.view.param_view.clu_combo.currentText())
		self.model.cluster(method = clu_method,
						   group_id   = self.current_group, 
						   fall_off_size = self.view.param_view.clu_param.value())
		self.refresh()


	def build_vq(self):
		import warnings
		warnings.filterwarnings('ignore')
		# get the vq and vq labels
		from sklearn.cluster import MiniBatchKMeans
		km = MiniBatchKMeans(self._vq_npts)
		X = self.fet
		km.fit(X)
		self.points = km.cluster_centers_   # these are vq
		self.labels = self._predict(self.points) # these are vq labels
		self.scores = self._validate_vq()

		self.vq['points'][self.current_group] = self.points
		self.vq['labels'][self.current_group] = self.labels
		self.vq['scores'][self.current_group] = self.scores

		self.vq_view = scatter_3d_view()
		self.vq_view._size = 5
		self.vq_view.set_data(self.points, CLU(self.labels))
		self.vq_view.transparency = 0.8
		self.vq_view.show()

		self.view.gui.status_message = str(self.scores)

        def trace_view_zoom(self):
                self.view.trace_view.locate_buffer = self.view.param_view.trace_view_zoom.value()


        def on_model_modified(self, e):
            if e.type == 'delete':
                with Timer("[CONTROL] Control -- remove spk from model.", verbose = conf.ENABLE_PROFILER):
                    self.model.remove_spk(self.current_group, self.view.spk_view.selected_spk)
                with Timer("[CONTROL] Control -- refresh view after delete.", verbose = conf.ENABLE_PROFILER): 
                    self.refresh()
            if e.type == 'refine':
                self.model.refine(self.current_group, self.view.spk_view.selected_spk)

	def _validate_vq(self):
		from sklearn.neighbors import KNeighborsClassifier as KNN
		knn = KNN(n_neighbors=1)
		knn.fit(self.points, self.labels)
		return knn.score(self.fet, self.clu.membership)


	def _predict(self, points):
		self.model.construct_kdtree(self.current_group)
		d = []
		for _kd in self.model.kd.iterkeys():
			tmp = _kd.query(points, 10)[0]
			d.append(tmp.mean(axis=1))
		d = np.vstack(np.asarray(d))
                labels = np.asarray(self.model.kd.values())[np.argmin(d, axis=0)]
		return labels

        def _transform(self, x, P, shift, scale):
            return _transform(x, P, shift, scale) 

        def construct_transformer(self, group_id, ndim=4):
            _pca_comp, _shift, _scale = self.model.construct_transformer(group_id, ndim)
            return _pca_comp, _shift, _scale            

