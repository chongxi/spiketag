from phy import gui
from phy.gui import Actions
from ..view import *
import PyQt4
from PyQt4 import QtGui
import os
from . import root_dir

class MainView(object):
	"""docstring for View"""
	def __init__(self, n_group, group2chs, fs, spklen, scale):
                self.gui = gui.GUI(name='spiketag', config_dir=root_dir())
		self.param_view = param_widget(n_group, group2chs)
		self.spk_view = spike_view()
		self.scatter_view = scatter_3d_view()
                self.trace_view = trace_view(fs=fs, spklen=spklen)
                self.correlogram_view = correlogram_view(fs=fs)
                self.cluster_view = cluster_view()
                self.amplitude_view = amplitude_view(fs=fs, scale=scale)
                self.raster_view = raster_view(fs=fs)
                self.ctree_view = ctree_view()
                #  self.firing_rate_view = firing_rate_view(fs=fs)
                self.gui.add_view(self.param_view)
                self.gui.add_view(self.scatter_view)
                self.gui.add_view(self.spk_view)
                self.gui.add_view(self.trace_view)
                self.gui.add_view(self.correlogram_view)
                self.gui.add_view(self.amplitude_view)
                self.gui.add_view(self.raster_view)
                self.gui.add_view(self.cluster_view)
                self.gui.add_view(self.ctree_view)
                #  self.gui.add_view(self.firing_rate_view)
	def set_data(self, data=None, mua=None, spk=None, fet=None, clu=None, spk_times=None):
                self.spk_view.set_data(spk, clu)
		self.scatter_view.set_data(fet, clu)
                self.trace_view.set_data(data, clu, spk_times)
                self.correlogram_view.set_data(clu, spk_times)
                self.amplitude_view.set_data(spk, clu, spk_times)
                self.raster_view.set_data(clu, spk_times)
                self.cluster_view.set_data(clu)
                self.ctree_view.set_data(clu)
                #  self.firing_rate_view.set_data(clu, spk_times)

	def show(self):
		self.gui.show()
                # the widget of phy can not show by GUI
                self.cluster_view.show()

