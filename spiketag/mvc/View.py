from phy import gui
from phy.gui import Actions
from ..view import *
import PyQt4
from PyQt4 import QtGui
import os

class MainView(object):
	"""docstring for View"""
	def __init__(self):
                self.gui = gui.GUI(config_dir=self.get_config_dir())
		self.param_view = param_widget()
		self.spk_view = spike_view()
		self.scatter_view = scatter_3d_view()
                self.wave_view = wave_view()
                self.correlogram_view = correlogram_view()
                self.cluster_view = cluster_view()
                self.amplitude_view = amplitude_view()
                self.raster_view = raster_view()
                self.firing_rate_view = firing_rate_view()
                self.gui.add_view(self.param_view)
                self.gui.add_view(self.scatter_view)
                self.gui.add_view(self.spk_view)
                self.gui.add_view(self.wave_view)
                self.gui.add_view(self.correlogram_view)
                self.gui.add_view(self.amplitude_view)
                self.gui.add_view(self.raster_view)
                self.gui.add_view(self.cluster_view)
                self.gui.add_view(self.firing_rate_view)
	def set_data(self, ch=None, mua=None, spk=None, fet=None, clu=None):
                self.spk_view.set_data(spk, clu)
		self.scatter_view.set_data(fet, clu)
                self.wave_view.set_data(ch, clu)
                self.correlogram_view.set_data(ch, clu)
                self.amplitude_view.set_data(ch, spk, clu)
                self.raster_view.set_data(ch, clu)
                self.cluster_view.set_data(clu)
                self.firing_rate_view.set_data(ch, clu)
		# if spk is not None and clu is None:
		# 	self.spk_view.set_data(spk[self.ch])
		# if fet is not None and clu is None:
		# 	self.scatter_view.set_data(fet[self.ch])
		# if spk is not None and clu is not None:
		# 	self.spk_view.set_data(spk[self.ch], clu[self.ch])
		# if fet is not None and clu is not None:
		# 	self.scatter_view.set_data(fet[self.ch], clu[self.ch])

        def bind_data(self, data=None, spktag=None):
                self.wave_view.bind(data, spktag)
                self.correlogram_view.bind(spktag)
                self.amplitude_view.bind(data, spktag)
                self.raster_view.bind(spktag)
                self.firing_rate_view.bind(spktag)
	def show(self):
		self.gui.show()
                # the widget of phy can not show by GUI
                self.cluster_view.show()

        def get_config_dir(self):
            '''
              TODO
              get the template gui state config dir, this is for temparory now.
            '''
            
            path = ''
            for fn in os.getcwd().split(os.path.sep):
                if len(fn) == 0:
                    path = os.path.join(os.path.sep, path)
                else:
                    path = os.path.join(path, fn)
    
                if fn == 'spiketag':
                    break

            return  os.path.join(path, 'spiketag/res')

