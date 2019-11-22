import sys
from ..view import probe_view, cluster_view, spike_view, scatter_3d_view, amplitude_view, ctree_view, trace_view, correlogram_view, pf_view
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
        
        self.setAutoFillBackground(True)
        p = self.palette()
        p.setColor(self.backgroundRole(), Qt.darkGray)
        p.setColor(self.foregroundRole(), Qt.white)
        self.setPalette(p)

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
        # self.treeview = ctree_view()
        self.pfview  = pf_view(pc=self._model.pc)
        self.traceview = trace_view(data=self._model.mua.data, fs=self.prb.fs)
        
        self.splitter1.addWidget(self.traceview.native)
        self.splitter1.addWidget(self.splitter_fet)
        self.splitter_fet.addWidget(self.fetview0.native)
        self.splitter_fet.addWidget(self.fetview1.native)
        self.splitter1.addWidget(self.spkview.native)
        self.splitter1.setSizes([30,50,80])  # trace_view, fet_view, spk_view

        self.splitter2.addWidget(self.corview.native) 
        self.splitter2.addWidget(self.pfview.native)
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
        # self.treeview.set_data(clu) 
        self.traceview.set_data(chs, clu, mua.spk_times[group_id])
        try:
            self.corview.set_data(clu, mua.spk_times[group_id])
        except Exception as e:
            pass

        self.pfview.set_data(clu, mua.spk_times[group_id]/self.prb.fs)

        self.traceview.locate_buffer = 1500

        # everytime when view.set_data
        # the corresponding clu will go through its registration process
        # then won'g go through it again
        if self.clu._event_reg_enable is True:
            self.register_event()


    def register_event(self):
        self.spkview.register_event()
        self.fetview0.register_event()
        self.fetview1.register_event()
        self.ampview.register_event()
        self.traceview.register_event()
        self.corview.register_event()
        self.pfview.register_event()  # make sure self.pfview already get pc
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
        # self.treeview.set_data(model.clu[group_id]) 
        self.traceview.set_data(self.prb[group_id], model.clu[group_id], model.mua.spk_times[group_id])
        # try:
        #     self.corview.set_data(model.clu[group_id], model.mua.spk_times[group_id])
        # except Exception as e:
        #     pass

        self.traceview.locate_buffer = 2000