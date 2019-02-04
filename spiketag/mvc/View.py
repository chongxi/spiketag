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
from ..view import probe_view, cluster_view, spike_view, scatter_3d_view, amplitude_view, ctree_view, trace_view, correlogram_view
from PyQt5 import QtCore
from PyQt5.QtCore import Qt, QThread, QEventLoop
from PyQt5.QtWidgets import QStatusBar, QMainWindow, QAction, QFileDialog, QWidget, QSplitter, QComboBox, QTextBrowser, QSlider, QPushButton, QTableWidget, QVBoxLayout, QHBoxLayout, QLineEdit, QLabel, QGridLayout
from PyQt5.QtGui import QIcon


class MainView(QMainWindow):

    def __init__(self, prb, model=None):
        super(MainView, self).__init__()
        self.prb = prb
        if model is not None:
            self._model = model
        self.initUI()


    def initUI(self):

        self.setWindowTitle("Spiketag") 

        self.status_bar = QStatusBar(self)
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage('Ready')

        self.centralWidget = QWidget(self)          
        self.setCentralWidget(self.centralWidget)   

        hbox = QHBoxLayout(self)
        self.splitter0 = QSplitter(Qt.Horizontal)
        self.splitter1 = QSplitter(Qt.Horizontal)
        self.splitter2 = QSplitter(Qt.Horizontal)
        self.splitter_fet = QSplitter(Qt.Vertical)
        self.splitter3 = QSplitter(Qt.Vertical)
        self.splitter3.addWidget(self.splitter1)
        self.splitter3.addWidget(self.splitter2)
        self.splitter3.setSizes([70,30])  # uppper and down (up, down)


        self.splitter_prb_cpu = QSplitter(Qt.Vertical)
        self.prb_view = probe_view()
        self.clu_view = cluster_view()
        self.prb_view.set_data(self.prb, font_size=35)
        self.clu_view.set_data(self._model.clu_manager)
        self.splitter_prb_cpu.addWidget(self.prb_view.native)
        self.splitter_prb_cpu.addWidget(self.clu_view.native)
        self.splitter0.addWidget(self.splitter_prb_cpu)
        self.splitter0.addWidget(self.splitter3)
        self.splitter0.setSizes([20, 180]) # prb_view and all others (left, right)

        hbox.addWidget(self.splitter0)

        self.centralWidget.setLayout(hbox)

        self.spkview = spike_view()
        self.fetview0 = scatter_3d_view()
        self.fetview1 = scatter_3d_view()
        self.ampview = amplitude_view(fs=self.prb.fs, scale=1)
        self.corview = correlogram_view(fs=self.prb.fs)
        self.treeview = ctree_view()
        self.traceview = trace_view(data=self._model.mua.data, fs=self.prb.fs)
        
        self.splitter1.addWidget(self.traceview.native)
        self.splitter1.addWidget(self.splitter_fet)
        self.splitter_fet.addWidget(self.fetview0.native)
        self.splitter_fet.addWidget(self.fetview1.native)
        self.splitter1.addWidget(self.spkview.native)
        self.splitter1.setSizes([30,50,80])  # trace_view, fet_view, spk_view

        self.splitter2.addWidget(self.corview.native) 
        self.splitter2.addWidget(self.treeview.native)
        self.splitter2.addWidget(self.ampview.native)
        self.splitter2.setSizes([40,40,100])  # corview, treeview, ampview


    def set_data(self, group_id, mua, spk, fet, clu):
        ### init view and set_data

        self.clu = clu
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
        self.traceview.set_data(chs, clu, mua.spk_times[group_id])
        try:
            self.corview.set_data(clu, mua.spk_times[group_id])
        except Exception as e:
            pass

        self.traceview.locate_buffer = 1500

        if self.clu._event_reg_enable is True:
            self.register_event()


    def register_event(self):
        self.spkview.register_event()
        self.fetview0.register_event()
        self.fetview1.register_event()
        self.ampview.register_event()
        self.traceview.register_event()
        self.corview.register_event()
        self.clu._event_reg_enable = False


    def set_data_from_model(self, group_id):
        ### init view and set_data
        model = self._model
        chs = self.prb[group_id]
        self.spkview.set_data(model.spk[group_id], model.clu[group_id])
        self.fetview0.dimension = [0,1,2]
        self.fetview0.set_data(model.fet[group_id], model.clu[group_id])   #[:,[0,1,2]].copy()
        self.fetview1.dimension = [0,1,3]
        self.fetview1.set_data(model.fet[group_id], model.clu[group_id])   #[:,[0,1,3]].copy()
        # else:
        self.ampview.set_data(model.spk[group_id], model.clu[group_id], model.mua.spk_times[group_id])
        self.treeview.set_data(model.clu[group_id]) 
        self.traceview.set_data(self.prb[group_id], model.clu[group_id], model.mua.spk_times[group_id])
        # try:
        #     self.corview.set_data(model.clu[group_id], model.mua.spk_times[group_id])
        # except Exception as e:
        #     pass

        self.traceview.locate_buffer = 2000