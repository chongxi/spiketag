import io
import os
import time
import sys
import struct
import socket
import numpy as np
import torch as torch
from torch.multiprocessing import Process, Pipe, SimpleQueue 
from spiketag.utils import Timer
from spiketag.utils import EventEmitter
from spiketag.view import scatter_3d_view
from PyQt5 import QtCore
from PyQt5.QtCore import Qt, QThread, QEventLoop
from PyQt5.QtWidgets import QMainWindow, QAction, QFileDialog, QWidget, QSplitter, QComboBox, QTextBrowser, QSlider, QPushButton, QTableWidget, QVBoxLayout, QHBoxLayout, QLineEdit, QLabel, QGridLayout
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QApplication
from spiketag.view.grid_scatter3d import grid_scatter3d
from spiketag.fpga import xike_config
from spiketag.utils import Timer


class bmi_recv(object):
    """docstring for FPGA"""
    def __init__(self, prb=None):
        self.prb = prb
        # self.group_idx = np.array(self.prb.grp_dict.keys())
        self.reset()
        # self.load_vq()

    def close(self):
        self.r32.close()

    def reset(self):
        self.r32 = io.open('/dev/xillybus_fet_clf_32', 'rb')
        # self.r32_buf = io.BufferedReader(r32)
        self.fd = os.open("./fet.bin", os.O_CREAT | os.O_WRONLY | os.O_NONBLOCK)
        self._size = 7*4  # 6 samples, 4 bytes/sample

    def shared_mem_init(self):
        n_spike_count_vector = len(self.prb.grp_dict.keys())
        # trigger task using frame counter
        self.spike_count_vector = torch.zeros(n_spike_count_vector,)
        self.spike_count_vector.share_memory_()

    def _fpga_process(self, queue):
        '''
        A daemon process dedicated on reading data from PCIE and update
        the shared memory with other processors: shared_arr 
        '''
        while True:
            buf = self.r32.read(self._size)
            # ----- real-time processing the BMI output ------
            # ----- This section should cost < 100us -----
            # tic = time.time()
            bmi_output = struct.unpack('<7i', buf)
            timestamp, grp_id, fet0, fet1, fet2, fet3, spk_id = bmi_output

            ##### real-time decoder

            ##### queue for visualization
            queue.put(bmi_output)


            ##### file for visualization
            # os.write(self.fd, buf)
            # toc = time.time()
            # print(toc-tic)
            # ----- This section should cost < 100us -----


    def start(self):
        self.queue = SimpleQueue()
        self.fpga_process = Process(target=self._fpga_process, name='fpga', args=(self.queue,)) #, args=(self.pipe_jovian_side,)
        self.fpga_process.daemon = True
        self.fpga_process.start()  


    def stop(self):
        self.fpga_process.terminate()
        self.fpga_process.join()





class BMI_GUI(QWidget):
    def __init__(self):
        QWidget.__init__(self)
        self.init_UI()
        self.fet_view_timer = QtCore.QTimer(self)
        self.fet_view_timer.timeout.connect(self.test_update)   # fet_view_update
        self.current_group = 0
        self.fpga = bmi_recv()
        self.fresh_interval = 60
        self._fet = {}
        self._clu = {}
        for i in range(40):
            self._fet[i] = []
            self._clu[i] = []


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

        self.fet_grid = grid_scatter3d(8, 5) 
        self.N = 300
        for i in range(40):
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


    def test_update(self):
        group_need_update = []
        while not self.fpga.queue.empty():
            with Timer('receive'):
                _timestamp, _grp_id, _fet0, _fet1, _fet2, _fet3, _spk_id = self.fpga.queue.get()
                group_need_update.append(_grp_id)
                self._fet[_grp_id].append(np.array([_fet0, _fet1, _fet2, _fet3])/float(2**16))
                self._clu[_grp_id].append(_spk_id)

        with Timer('update'):
            for grp_id in group_need_update:
                fet = np.array(self._fet[grp_id])
                clu = np.array(self._clu[grp_id])
                if fet.shape[0]>0 and fet.shape[0]<=self.N:
                    self.fet_grid.fet_view[grp_id].stream_in(fet, clu, highlight_no=30)
                elif fet.shape[0]>self.N:
                    self.fet_grid.fet_view[grp_id].stream_in(fet[-self.N:], clu[-self.N:], highlight_no=30)
        # print(self.fpga.queue.get())


    def fet_view_update(self):
        # with Timer('update fet', verbose=False):
        # 
        # try:
            N = 5000
            fet = np.fromfile('./fet.bin', dtype=np.int32)
            # try:
                # fet = np.memmap('./fet.bin', dtype='int32', mode='r')
            if fet.shape[0] > 0:
                try:
                    fet = fet.reshape(-1, 7)
                    fet_info = fet[:,:2]
                    fet_val = fet[:,2:6]/float(2**16)
                    labels  = fet[:, -1]
                    # get idx of fet from current selected group
                    idx = np.where(fet_info[:,1]==self.current_group)[0]
                    if len(idx)>N:
                        idx = idx[-N:]
                        fet = fet_val[idx, :]
                        clu = np.zeros((fet.shape[0],), dtype=np.int32)
                        clu[-30:] = 1 
                    # if self.current_group in self.fpga.vq_grp_idx:
                    #     fet = fet_val[idx, :]
                    #     clu = self.fpga.vq['labels'][self.current_group][labels[idx]]
                    #     # # self.log.info(clu)

                    else:
                        fet = fet_val[idx, :]
                        # if len(idx)>N:
                        #     fet = fet[-N:, :]
                        clu = np.zeros((fet.shape[0],), dtype=np.int32)
                        clu[-30:] = 1

                    # self.log.info('get_fet{}'.format(idx.shape))
                    if len(fet)>0:
                        try:
                            self.fet_view.stream_in(fet, clu, highlight_no=30)
                        except:
                            pass
                except:
                    pass 


if __name__ == '__main__':
    app = QApplication(sys.argv) 
    gui = BMI_GUI()
    gui.show()
    sys.exit(app.exec_())
