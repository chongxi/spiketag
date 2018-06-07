from spiketag.base import *
from spiketag.view import *
from spiketag.mvc import MainModel, MainView
import numpy as np
from PyQt5.QtWidgets import QApplication
import sys


prb = probe(shank_no=1)
prb[0] = np.array([0,1,2])
prb.fs = 20000.
prb.n_ch = 3
prb.reorder_by_chip=False

model = MainModel('./cell_0109.bin', './cell_0109.spk.bin', prb, binary_radix=13)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    view = MainView(prb)
    view.set_data(0, model.mua, model.spk[0], model.fet[0], model.clu[0])
    view.show()
    sys.exit(app.exec_())

