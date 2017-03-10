from phy import gui
from phy.gui import Actions
from ..view import *
import PyQt4
from PyQt4 import QtGui

class MainView(object):
	"""docstring for View"""
	def __init__(self):
		self.gui = gui.GUI()
		self.param_view = param_widget()
		self.spk_view = spike_view()
		self.scatter_view = scatter_3d_view()
                self.wave_view = wave_view()
                self.correlogram_view = correlogram_view()
                self.cluster_view = cluster_view()
                self.gui.add_view(self.param_view, name='Dashboard')
                self.gui.add_view(self.scatter_view, name='ScatterView')
                self.gui.add_view(self.spk_view, name='SpikeView')
                self.gui.add_view(self.wave_view, name='WaveView')
                self.gui.add_view(self.correlogram_view, name='CorrelogramView')
                self.gui.add_view(self.cluster_view, name='ClusterView')

	def set_data(self, ch=None, mua=None, spk=None, fet=None, clu=None):
                self.spk_view.set_data(spk, clu)
		self.scatter_view.set_data(fet, clu)
                self.wave_view.set_data(ch, clu)
                self.correlogram_view.set_data(ch, clu)
                self.cluster_view.set_data(clu)
		# if spk is not None and clu is None:
		# 	self.spk_view.set_data(spk[self.ch])
		# if fet is not None and clu is None:
		# 	self.scatter_view.set_data(fet[self.ch])
		# if spk is not None and clu is not None:
		# 	self.spk_view.set_data(spk[self.ch], clu[self.ch])
		# if fet is not None and clu is not None:
		# 	self.scatter_view.set_data(fet[self.ch], clu[self.ch])

        def bind_data(self, data=None, spktag=None):
                self.wave_view.bind(data,spktag)
                self.correlogram_view.bind(data,spktag)

	def show(self):
		self.gui.show()
                # the widget of phy can not show by GUI
                self.cluster_view.show()
