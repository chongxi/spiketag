import time
import sys
import socket
import numpy as np
from spiketag.utils import Timer
from spiketag.view import scatter_3d_view
from PyQt5 import QtCore
from PyQt5.QtCore import Qt, QThread, QEventLoop
from PyQt5.QtWidgets import QMainWindow, QAction, QFileDialog, QWidget, QSplitter, QComboBox, QTextBrowser, QSlider, QPushButton, QTableWidget, QVBoxLayout, QHBoxLayout, QLineEdit, QLabel, QGridLayout
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QApplication
from spiketag.view.grid_scatter3d import grid_scatter3d
from spiketag.fpga import xike_config
from spiketag.base import probe
from spiketag.realtime import BMI
from spiketag.fpga import read_mem_16



class BMI_GUI(QWidget):
    def __init__(self, prb, fet_file, show=True):
        QWidget.__init__(self)
        self.fet_view_timer = QtCore.QTimer(self)
        self.fet_view_timer.timeout.connect(self.update_fet_views)   # fet_view_update
        self.current_group = 0
        self.init_FPGA(prb, fet_file)
        self.track_groups()
        self.fresh_interval = 30
        self._fet = {}
        self._clu = {}
        for i in range(40):
            self._fet[i] = np.zeros((40, 4))
            self._clu[i] = np.zeros((40,), dtype=np.int64)
        if show:
            self.init_UI()


    def init_FPGA(self, prb, fet_file):
        print('---init FPGA BMI---')
        print('---download probe file into FPGA---')
        try:
            self.fpga = BMI(prb, fet_file)
            self.fpga_setting = xike_config(prb)
            print('---success---')
        except Exception as e:
            raise
        

    def init_UI(self, keys='interactive'):
        self.setAutoFillBackground(True)
        p = self.palette()
        p.setColor(self.backgroundRole(), Qt.darkGray)
        p.setColor(self.foregroundRole(), Qt.white)
        self.setPalette(p)

        self.fpgaBtn = QPushButton("FPGA Stream Off",self)
        self.fpgaBtn.setCheckable(True)
        self.fpgaBtn.setStyleSheet("background-color: darkgrey")
        self.fpgaBtn.toggled.connect(self.fpga_process_toggle)            

        rows,cols=6,7
        self.fet_grid = grid_scatter3d(rows, cols) 
        self.N = 2000
        for i in range(rows*cols):
            if i in self._group_on_track:
                self.fet_grid.fet_view[i].info = i
            self.fet_grid.fet_view[i].set_data(np.zeros((self.N,4), dtype=np.float32)) 

        layout = QVBoxLayout()
        layout.addWidget(self.fpgaBtn)
        layout.addWidget(self.fet_grid)

        self.setLayout(layout)


    def fpga_process_toggle(self, checked):
        if checked:
            self.fpga_process_start()
        else:
            self.fpga_process_stop()


    def fpga_process_start(self):
        self.fpgaBtn.setText('FPGA Stream ON')
        self.fpgaBtn.setStyleSheet("background-color: green")
        self.fpga.start()
        self.fet_view_timer.start(self.fresh_interval)


    def fpga_process_stop(self):
        self.fpgaBtn.setText('FPGA Stream Off')
        self.fpgaBtn.setStyleSheet("background-color: darkgrey")
        self.fpga.stop()
        self.fet_view_timer.stop()


    @property
    def group_on_track(self):
        return self._group_on_track


    def track_groups(self, grp_ids=None):
        if grp_ids is None:
            self._group_on_track = np.where(self.fpga_setting.transformer_status)[0]
        else:
            self._group_on_track = grp_ids
        print('{} groups on track:{}'.format(len(self.group_on_track), self.group_on_track))


    def update_fet_views(self):
        group_need_update = []
        _nspk, limit = 0, 40
        while not self.fpga.gui_queue.empty():
            with Timer('receive', verbose=False):
                _timestamp, _grp_id, _fet0, _fet1, _fet2, _fet3, _spk_id = self.fpga.gui_queue.get()
                if _grp_id in self._group_on_track:
                    group_need_update.append(_grp_id)
                    incoming_fet = np.array([_fet0, _fet1, _fet2, _fet3])/float(2**16)
                    incoming_clu = int(_spk_id)
                    self._fet[_grp_id] = np.roll(self._fet[_grp_id], 1, axis=0)
                    self._fet[_grp_id][0] = incoming_fet
                    self._clu[_grp_id] = np.roll(self._clu[_grp_id], 1)
                    self._clu[_grp_id][0] = incoming_clu

                    _nspk += 1
                    if _nspk>limit:
                        while not self.fpga.gui_queue.empty():
                            self.fpga.gui_queue.get()
                        break

        with Timer('update', verbose=False):
            for grp_id in group_need_update:
                with Timer('render', verbose=False):
                    self.fet_grid.fet_view[grp_id].stream_in(self._fet[grp_id][:_nspk], 
                                                             self._clu[grp_id][:_nspk],
                                                             rho=1, highlight_no=10)



if __name__ == '__main__':
    app = QApplication(sys.argv) 
    prb = probe(prbfile='./dusty.json')
    gui = BMI_GUI(prb=prb, fet_file='./fet.bin')
    gui.show()
    n_id = read_mem_16(0) + 1
    print(n_id)
    gui.fpga.set_binner(bin_size=33.33, n_id=n_id, n_bin=1)
    # fpga.thres[:] = -300.
    # fpga.thres[:] = -150.
    # fpga.thres[80:84] = -590.
    # print(fpga.thres)
    sys.exit(app.exec_())
