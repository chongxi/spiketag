from spiketag.base import *
from spiketag.view import *
from spiketag.mvc import MainModel, MainView
from spiketag.mvc.Control import controller
import numpy as np
from PyQt5.QtWidgets import QApplication
import sys

tritrode = probe(shank_no=1)
tritrode[0] = np.array([0,1,2])
tritrode.mapping[0] = np.array([-90,0])
tritrode.mapping[1] = np.array([90,0])
tritrode.mapping[2] = np.array([0,10])
tritrode.fs = 20000.
tritrode.n_ch = 3
tritrode.reorder_by_chip=False

def model_view():
    model = MainModel('./cell_0109.bin', './cell_0109.spk.bin', tritrode, binary_radix=11)
    view = MainView(tritrode)
    view.set_data(0, model.mua, model.spk[0], model.fet[0], model.clu[0])
    view.show()

def sorter():
    ctrl = controller('./cell_0109.bin', './cell_0109.spk.bin', tritrode, binary_radix=11)
    ctrl.show()

if __name__ == '__main__':
    app  = QApplication(sys.argv)
    sorter()
    sys.exit(app.exec_())
