import time
import sys
import socket
import numpy as np
from spiketag.utils import Timer
from spiketag.view import scatter_3d_view
from PyQt5 import QtCore
from PyQt5.QtCore import Qt, QThread, QEventLoop
from PyQt5.QtWidgets import QMainWindow, QAction, QFileDialog, QWidget, QSplitter, QComboBox, QTextBrowser, QSlider, QPushButton, QTableWidget, QVBoxLayout, QHBoxLayout, QLineEdit, QLabel, QGridLayout
from PyQt5.QtWidgets import QApplication
from spiketag.view.grid_scatter3d import grid_scatter3d
from spiketag.base import probe
from spiketag.realtime import BMI
from spiketag.analysis import *
from spiketag.analysis.decoder import NaiveBayes


class BMI_GUI(QWidget):
    def __init__(self, fet_file, prb=None, show=True):
        QWidget.__init__(self)
        self.fet_view_timer = QtCore.QTimer(self)
        self.fet_view_timer.timeout.connect(self.update_fet_views)   # fet_view_update
        self.current_group = 0
        self.init_BMI(fet_file, prb)
        self.fresh_interval = 30
        self._fet = {}
        self._clu = {}
        for i in range(40):
            self._fet[i] = np.zeros((40, 4))
            self._clu[i] = np.zeros((40,), dtype=np.int64)
        if show:
            self.init_UI()


    def init_BMI(self, fet_file, prb=None):
        print('---init FPGA BMI---')
        print('---download probe file into FPGA---')
        try:
            self.bmi = BMI(fetfile=fet_file, prb=prb)
        except Exception as e:
            raise
        

    def init_UI(self, keys='interactive'):
        self.setAutoFillBackground(True)
        p = self.palette()
        p.setColor(self.backgroundRole(), Qt.darkGray)
        p.setColor(self.foregroundRole(), Qt.white)
        self.setPalette(p)

        self.bmiBtn = QPushButton("BMI Stream Off",self)
        self.bmiBtn.setCheckable(True)
        self.bmiBtn.setStyleSheet("background-color: darkgrey")
        self.bmiBtn.toggled.connect(self.bmi_process_toggle)            

        rows,cols=6,7
        self.fet_grid = grid_scatter3d(rows, cols) 
        self.N = 2000
        for i in range(rows*cols):
            if i in self.bmi.fpga.configured_groups:
                self.fet_grid.fet_view[i].info = i
            self.fet_grid.fet_view[i].set_data(np.zeros((self.N,4), dtype=np.float32)) 

        layout = QVBoxLayout()
        layout.addWidget(self.bmiBtn)
        layout.addWidget(self.fet_grid)

        self.setLayout(layout)


    def bmi_process_toggle(self, checked):
        if checked:
            self.bmi_process_start()
        else:
            self.bmi_process_stop()


    def bmi_process_start(self, gui_queue=True):
        self.bmiBtn.setText('BMI Stream ON')
        self.bmiBtn.setStyleSheet("background-color: green")
        self.bmi.start(gui_queue)
        if gui_queue:
            self.fet_view_timer.start(self.fresh_interval)


    def bmi_process_stop(self):
        self.bmiBtn.setText('BMI Stream Off')
        self.bmiBtn.setStyleSheet("background-color: darkgrey")
        self.bmi.stop()
        self.fet_view_timer.stop()


    def update_fet_views(self):
        group_need_update = []
        _nspk, limit = 0, 40
        while not self.bmi.gui_queue.empty():
            with Timer('receive', verbose=False):
                _timestamp, _grp_id, _fet0, _fet1, _fet2, _fet3, _spk_id = self.bmi.gui_queue.get()
                # print(_grp_id, _spk_id)
                if _grp_id in self.bmi.fpga.configured_groups and _spk_id>0:
                    group_need_update.append(_grp_id)
                    incoming_fet = np.array([_fet0, _fet1, _fet2, _fet3])/float(2**16)
                    incoming_clu = int(_spk_id)
                    self._fet[_grp_id] = np.roll(self._fet[_grp_id], 1, axis=0)
                    self._fet[_grp_id][0] = incoming_fet
                    self._clu[_grp_id] = np.roll(self._clu[_grp_id], 1)
                    self._clu[_grp_id][0] = incoming_clu

                    _nspk += 1
                    if _nspk>limit:
                        while not self.bmi.gui_queue.empty():
                            self.bmi.gui_queue.get()
                        break

        with Timer('update', verbose=False):
            for grp_id in group_need_update:
                with Timer('render', verbose=False):
                    self.fet_grid.fet_view[grp_id].stream_in(self._fet[grp_id][:_nspk], 
                                                             self._clu[grp_id][:_nspk],
                                                             rho=1, highlight_no=10)



if __name__ == '__main__':
    # 1. prb, gui and bmi and binner
    app = QApplication(sys.argv) 
    prb = probe(prbfile='./dusty.json')
    gui = BMI_GUI(prb=prb, fet_file='./fet.bin')


    # 2. binner
    # bin_size, B_bins = 25e-3, 10
    # gui.bmi.set_binner(bin_size=bin_size, B_bins=B_bins)

    # 3. decoder
    # pos = np.fromfile('../sorting/dusty_pos.bin').reshape(-1,2)
    # pc = place_field(pos=pos, t_step=33.333e-3)
    # replay_offset = 2.004
    # start = 320
    # end   = 2500
    # pc.align_with_recording(start, end, replay_offset)
    # pc.initialize(bin_size=4, v_cutoff=25)
    # pc.load_spktag('../sorting/spktag/test_allspikes', show=True)
    # dec = NaiveBayes(t_step=bin_size, t_window=B_bins*bin_size)
    # dec.connect_to(pc)
    # gui.bmi.set_decoder(dec, dec_result_file='./decoded_pos.bin')

    gui.show()
    sys.exit(app.exec_())
