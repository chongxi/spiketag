# from phy import gui
# from phy.gui import Actions
# from ..view import *
# import PyQt4
# from PyQt4 import QtGui
# import os
# from . import root_dir

# class MainView(object):
# 	"""docstring for View"""
# 	def __init__(self, n_group, group2chs, fs, spklen, scale):
#                 self.gui = gui.GUI(name='spiketag', config_dir=root_dir())
# 		self.param_view = param_widget(n_group, group2chs)
# 		self.spk_view = spike_view()
# 		self.scatter_view = scatter_3d_view()
#                 self.trace_view = trace_view(fs=fs, spklen=spklen)
#                 self.correlogram_view = correlogram_view(fs=fs)
#                 self.cluster_view = cluster_view()
#                 self.amplitude_view = amplitude_view(fs=fs, scale=scale)
#                 self.raster_view = raster_view(fs=fs)
#                 self.ctree_view = ctree_view()
#                 #  self.firing_rate_view = firing_rate_view(fs=fs)
#                 self.gui.add_view(self.param_view)
#                 self.gui.add_view(self.scatter_view)
#                 self.gui.add_view(self.spk_view)
#                 self.gui.add_view(self.trace_view)
#                 self.gui.add_view(self.correlogram_view)
#                 self.gui.add_view(self.amplitude_view)
#                 self.gui.add_view(self.raster_view)
#                 self.gui.add_view(self.cluster_view)
#                 self.gui.add_view(self.ctree_view)
#                 #  self.gui.add_view(self.firing_rate_view)
# 	def set_data(self, data=None, mua=None, spk=None, fet=None, clu=None, spk_times=None):
#                 self.spk_view.set_data(spk, clu)
# 		self.scatter_view.set_data(fet, clu)
#                 self.trace_view.set_data(data, clu, spk_times)
#                 self.correlogram_view.set_data(clu, spk_times)
#                 self.amplitude_view.set_data(spk, clu, spk_times)
#                 self.raster_view.set_data(clu, spk_times)
#                 self.cluster_view.set_data(clu)
#                 self.ctree_view.set_data(clu)
#                 #  self.firing_rate_view.set_data(clu, spk_times)

# 	def show(self):
# 		self.gui.show()
#                 # the widget of phy can not show by GUI
#                 self.cluster_view.show()

import sys
from ..view import probe_view, spike_view, scatter_3d_view, amplitude_view, ctree_view, trace_view, correlogram_view
from PyQt5 import QtCore
from PyQt5.QtCore import Qt, QThread, QEventLoop
from PyQt5.QtWidgets import QMainWindow, QAction, QFileDialog, QWidget, QSplitter, QComboBox, QTextBrowser, QSlider, QPushButton, QTableWidget, QVBoxLayout, QHBoxLayout, QLineEdit, QLabel, QGridLayout
from PyQt5.QtGui import QIcon


class MainView(QWidget):

    def __init__(self, prb):
        super(MainView, self).__init__()
        self.prb = prb
        self.initUI()


    def initUI(self):

        hbox = QHBoxLayout(self)
        self.splitter0 = QSplitter(Qt.Horizontal)
        self.splitter1 = QSplitter(Qt.Horizontal)
#         textedit = QTextEdit()
#         self.splitter1.addWidget(self.topleft)
#         self.splitter1.addWidget(textedit)
#         self.splitter1.setSizes([100,200])
        self.splitter2 = QSplitter(Qt.Horizontal)
        self.splitter_fet = QSplitter(Qt.Vertical)
        # self.splitter2.addWidget(self.splitter0)

        self.splitter3 = QSplitter(Qt.Vertical)

        # self.splitter3.addWidget(self.splitter0)
        self.splitter3.addWidget(self.splitter1)
        self.splitter3.addWidget(self.splitter2)

        self.prb_view = probe_view()
        self.prb_view.set_data(self.prb, font_size=35)
        self.splitter0.addWidget(self.prb_view.native)
        self.splitter0.addWidget(self.splitter3)
#         self.splitter2.addWidget(self.bottom)

        hbox.addWidget(self.splitter0)

        self.setLayout(hbox)
        # QApplication.setStyle(QStyleFactory.create('Cleanlooks'))

        self.setGeometry(300, 300, 1000, 500)
        self.setWindowTitle('spiketag')
#         self.show()

        self.spkview = spike_view()
        self.fetview0 = scatter_3d_view()
        self.fetview1 = scatter_3d_view()
        self.ampview = amplitude_view(fs=self.prb.fs, scale=1)
        self.corview = correlogram_view(fs=self.prb.fs)
        self.treeview = ctree_view()
        self.traceview = trace_view(fs=self.prb.fs)
        
        self.splitter1.addWidget(self.traceview.native)
        self.splitter1.addWidget(self.splitter_fet)
        self.splitter_fet.addWidget(self.fetview0.native)
        self.splitter_fet.addWidget(self.fetview1.native)
        self.splitter1.addWidget(self.spkview.native)

        self.splitter2.addWidget(self.corview.native) 
        self.splitter2.addWidget(self.treeview.native)
        self.splitter2.addWidget(self.ampview.native)


    def set_data(self, group_id, mua, spk, fet, clu):
        ### init view and set_data
        
        chs = self.prb[group_id]
        self.spkview.set_data(spk, clu)
        # self.fetview0.set_data(fet, clu)
        # if fet.shape[1]>3:
        self.fetview0.dimension = [0,1,2]
        self.fetview0.set_data(fet, clu)   #[:,[0,1,2]].copy()
        self.fetview1.dimension = [0,1,3]
        self.fetview1.set_data(fet, clu)   #[:,[0,1,3]].copy()
        # else:
        self.ampview.set_data(spk, clu, mua.spk_times[group_id])
        self.treeview.set_data(clu) 
        self.traceview.set_data(mua.data[:,chs], clu, mua.spk_times[group_id])
        try:
            self.corview.set_data(clu, mua.spk_times[group_id])
        except Exception as e:
            pass
        
#         self.traceview.locate_buffer = 2000
