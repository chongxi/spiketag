from spiketag.base import *
from spiketag.view import *
from spiketag.mvc import MainModel, MainView
import numpy as np
from PyQt5.QtWidgets import QApplication
import sys

tritrode = probe(shank_no=1)
tritrode[0] = np.array([0,1,2])
tritrode.mapping[0] = np.array([-10,0])
tritrode.mapping[1] = np.array([10,0])
tritrode.mapping[2] = np.array([0,10])
tritrode.fs = 20000.
tritrode.n_ch = 3
tritrode.reorder_by_chip=False

model = MainModel('./cell_0109.bin', './cell_0109.spk.bin', tritrode, binary_radix=11)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    view = MainView(tritrode)
    view.set_data(0, model.mua, model.spk[0], model.fet[0], model.clu[0])
    view.show()
    sys.exit(app.exec_())
