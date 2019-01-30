from spiketag.base import *
from spiketag.view import *
from spiketag.mvc import MainModel, MainView
from spiketag.mvc.Control import controller
import numpy as np
from PyQt5.QtWidgets import QApplication
import sys


tritrode = probe(shank_no=1, grp_No=1)
tritrode[0] = np.array([0,1,2])
tritrode.mapping[0] = np.array([-90,0])
tritrode.mapping[1] = np.array([90,0])
tritrode.mapping[2] = np.array([0,10])
tritrode.fs = 20000.
tritrode.n_ch = 3


# def model_view():
#     model = MainModel('./cell_0109.bin', './cell_0109.spk.bin', tritrode, binary_radix=11)
#     view = MainView(tritrode)
#     view.set_data(0, model.mua, model.spk[0], model.fet[0], model.clu[0])
#     view.show()


def sort():
    ctrl = controller(
                      probe = tritrode,
                      mua_filename='./cell_0109.bin', 
                      spk_filename='./cell_0109.spk.bin', 
                      binary_radix=11, 
                      # cutoff=[-100, 100],
                      # time_segs=[[0,320]],
                      fall_off_size=15
                     )
    ctrl.sort(clu_method='no_clustering')
    ctrl.show()


if __name__ == '__main__':
    app  = QApplication(sys.argv)
    sort()
    sys.exit(app.exec_())

