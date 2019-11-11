import sys
from ...base import probe
from ...view import raster_view, scatter_3d_view
import numpy as np
from PyQt5 import QtCore
from PyQt5.QtCore import Qt, QThread, QEventLoop
from PyQt5.QtWidgets import QMainWindow, QAction, QFileDialog, QWidget, QSplitter, QComboBox, QTextBrowser, QSlider, QPushButton, QTableWidget, QVBoxLayout, QHBoxLayout, QLineEdit, QLabel, QGridLayout
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QApplication
from ...realtime import BMI


class BMI_RASTER_GUI(QWidget):
    def __init__(self, prb=None, fet_file='./fet.bin', t_window=5e-3):
        QWidget.__init__(self)
        self.view_timer = QtCore.QTimer(self)
        self.view_timer.timeout.connect(self.view_update)
        self.update_interval = 60
        self.bmi = BMI(prb, fet_file)

        self.setAutoFillBackground(True)
        p = self.palette()
        p.setColor(self.backgroundRole(), Qt.darkGray)
        p.setColor(self.foregroundRole(), Qt.white)
        self.setPalette(p)

        self.bmiBtn = QPushButton("BMI Stream Off",self)
        self.bmiBtn.setCheckable(True)
        self.bmiBtn.setStyleSheet("background-color: darkgrey")
        self.bmiBtn.toggled.connect(self.bmi_process_toggle)     

        self.rsview = raster_view(n_units=self.bmi.fpga.n_units+1, t_window=t_window)

        layout = QVBoxLayout()
        layout.addWidget(self.bmiBtn)
        layout.addWidget(self.rsview.native)

        self.setLayout(layout)


    def bmi_process_toggle(self, checked):
        if checked:
            self.bmi_process_start()
        else:
            self.bmi_process_stop()


    def bmi_process_start(self, gui_queue=False):
        self.bmiBtn.setText('BMI Stream ON')
        self.bmiBtn.setStyleSheet("background-color: green")
        self.bmi.start(gui_queue)
        self.view_timer.start(self.update_interval)


    def bmi_process_stop(self):
        self.bmiBtn.setText('BMI Stream Off')
        self.bmiBtn.setStyleSheet("background-color: darkgrey")
        self.bmi.stop()
        self.view_timer.stop()

    def view_update(self):
        self.rsview.update_fromfile('./fet.bin', last_N=8000, view_window=10)

    # def keyPressEvent(self, e):
    #     print("event",e)
        # if e.key() == Qt.Key_F5:
        #     if self.view_timer.isActive():
        #         self.view_timer.stop()
        #     else:
        #         self.view_timer.start(self.update_interval)
        #     self.close()
        # if e.key()==Qt.Key_Space:
        #     self.close()


if __name__ == '__main__':
    # 1. prb, gui and bmi and binner
    app = QApplication(sys.argv) 
    prb = probe(prbfile='./dusty/dusty.json')
    gui = BMI_RASTER_GUI(prb=prb, fet_file='./fet.bin')
    bin_size, B_bins = 25e-3, 10
    gui.bmi.set_binner(bin_size=bin_size, B_bins=B_bins)

    # 2. decoder
    # pos = np.fromfile('./sorting/dusty_pos.bin').reshape(-1,2)
    # pc = place_field(pos=pos, t_step=33.333e-3)
    # replay_offset = 2.004
    # start = 320
    # end   = 2500
    # pc.align_with_recording(start, end, replay_offset)
    # pc.initialize(bin_size=4, v_cutoff=25)
    # pc.load_spktag('./sorting/spktag/test_allspikes', show=True)
    # dec = NaiveBayes(t_step=bin_size, t_window=B_bins*bin_size)
    # dec.connect_to(pc)
    # gui.bmi.set_decoder(dec, dec_result_file='./decoded_pos.bin')

    gui.show()
    sys.exit(app.exec_())
