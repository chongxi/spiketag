from phy import gui
from ..view import *


class MainView(object):
	"""docstring for View"""
	def __init__(self):
		self.gui = gui.GUI()
		self.param_view = param_widget()
		self.gui.add_view(self.param_view, position='left', name='params')
		self.spk_view = spike_view()
		self.scatter_view = scatter_3d_view()
		self.gui.add_view(self.scatter_view)
		self.gui.add_view(self.spk_view)

	def set_data(self, mua=None, spk=None, fet=None, clu=None):
		self.spk_view.set_data(spk, clu)
		self.scatter_view.set_data(fet, clu)
		# if spk is not None and clu is None:
		# 	self.spk_view.set_data(spk[self.ch])
		# if fet is not None and clu is None:
		# 	self.scatter_view.set_data(fet[self.ch])
		# if spk is not None and clu is not None:
		# 	self.spk_view.set_data(spk[self.ch], clu[self.ch])
		# if fet is not None and clu is not None:
		# 	self.scatter_view.set_data(fet[self.ch], clu[self.ch])


	def show(self):
		self.gui.show()

