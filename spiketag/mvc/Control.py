from .Model import MainModel
from .View import MainView


class Sorter(object):
	"""docstring for Sorter"""
	def __init__(self, *args, **kwargs):
		self.model = MainModel(*args, **kwargs)
		self.view  = MainView()
		self.view.param_view.signal_ch_changed.connect(self.update_ch)
		self.view.param_view.signal_recluster.connect(self.update_clu)	
		self.view.param_view.signal_refine.connect(self.dismiss)
		self.showmua = False
		self.ch = 0
		self.set_model(self.model)


	def set_model(self, model):
		ch = self.ch
		if model is not None:
			self.model = model
		if self.showmua is False:
			self.view.set_data(spk=self.model.spk[ch], 
				               fet=self.model.fet[ch], 
				               clu=self.model.clu[ch])
		else:
			self.view.set_data(mua=self.model.mua, 
							   spk=self.model.spk[ch], 
							   fet=self.model.fet[ch], 
							   clu=self.model.clu[ch])


	def refresh(self):
		self.set_model(self.model)


	def run(self):
		self.view.show()
		self.update_ch()


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
		return self.model.spk[self.ch]   # ndarray

	@property
	def fet(self):
		return self.model.fet[self.ch]   # ndarray

	@fet.setter
	def fet(self, fet_value):
		self.model.fet.fet[self.ch] = fet_value
		self.refresh()

	@property
	def clu(self):
		return self.model.clu[self.ch]   # CLU

	@clu.setter
	def clu(self, clu_membership):
		self.clu.membership = clu_membership
		self.clu.__construct__()
		self.refresh()


	def update_ch(self):
		# ---- update chosen ch and get cluster ----
		self.ch = self.view.param_view.ch.value()
		# print '{} selected'.format(self.ch)
		self.set_model(self.model)


	# cluNo is a noisy cluster, usually 0, assign it's member to other clusters
	# using knn classifier: for each grey points:
	# 1. Get its knn in each other clusters (which require KDTree for each cluster)
	# 2. Get the mean distance of these knn points in each KDTree (Cluster)
	# 3. Assign the point to the closet cluster
	def dismiss(self, cluNo=0, k=30):
		# get the features of targeted cluNo
		X = self.fet[self.clu[cluNo]]
		# classification on these features
		lables_X = self.model.predict(self.ch, X, method='knn', k=10)
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


	def update_clu(self):
		clu_method = str(self.view.param_view.clu_combo.currentText())
		self.model.cluster(method = clu_method,
						   chNo   = self.ch, 
						   fall_off_size = self.view.param_view.clu_param.value())
		self.refresh()