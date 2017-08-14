import numpy as np
from .Model import MainModel
from .View import MainView
from ..view import scatter_3d_view
from ..view import wave_view
from ..base import CLU
from ..utils import warning, conf
from ..utils.utils import Timer


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
		X = self.fet[self.clu[cluNo]]
		# classification on these features
		lables_X = self.model.predict(self.current_group, X, method='knn', k=10)
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
		fet_len    = self.view.param_view.fet_No.value()
		self.model.fet[self.current_group] = self.model.spk.tofet(groupNo=self.current_group, method=fet_method, ncomp=fet_len)
		self.refresh()


	def update_clu(self):
		clu_method = str(self.view.param_view.clu_combo.currentText())
		self.model.cluster(method = clu_method,
						   groupNo   = self.current_group, 
						   fall_off_size = self.view.param_view.clu_param.value())
		self.refresh()


	def build_vq(self):
		# print 'build vector quantization'
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


	def _validate_vq(self):
		from sklearn.neighbors import KNeighborsClassifier as KNN
		knn = KNN(n_neighbors=1)
		knn.fit(self.points, self.labels)
		return knn.score(self.fet, self.clu.membership)


	def _predict(self, points):
		self.model.construct_kdtree(self.current_group)
		d = []
		for _kd in self.model.kd:
			tmp = _kd.query(points, 10)[0]
			d.append(tmp.mean(axis=1))
		d = np.vstack(np.asarray(d))
		labels = np.argmin(d, axis=0)
		return labels
