"""
Example demonstrating the use of two GLCanvases in one QtApp.
"""

try:
    from PyQt5 import QtCore
    from PyQt5 import QtWidgets as QtGui
except:
    from PyQt4 import QtCore
    from PyQt4 import QtGui as QtGui

import sys
import numpy as np
from ..view import scatter_3d_view
from ..utils import Timer

class grid_scatter3d(QtGui.QWidget):
    '''
    Show all feature spaces using fet.bin
    
    Example:
    from spiketag.view import grid_scatter3d
    gd = grid_scatter3d()
    gd.from_file('./fet.bin')
    gd.show()
    '''
    def __init__(self, rows=5, cols=8):
        # QtGui.QMainWindow.__init__(self)
        super(grid_scatter3d, self).__init__()
        self.resize(1000, 500)
        self.setWindowTitle('Feature Spaces')

        self.grid = QtGui.QGridLayout()
        self.setLayout(self.grid)
        self.grid.setContentsMargins(0,0,0,0)
        self.grid.setSpacing(1)
        # self.grid.setMargin(0)

        self.rows = rows
        self.cols = cols
        positions = [(i,j) for i in range(self.rows) for j in range(self.cols)]
        self.fet_view = {}
        for idx, position in enumerate(positions):
            self.fet_view[idx] = scatter_3d_view()
            self.fet_view[idx].info = idx
            self.grid.addWidget(self.fet_view[idx].native, *position)
            # self.fet_view[idx].create_native()
            # self.fet_view[idx].native.setParent(self)
            # self.grid.addWidget(self.fet_view[idx].native, *position)


    def load_units(self, fet, scale_factor=float(2**16)):
        '''
        unit packet:
        (timestamps, #group, fet0, fet1, fet2, fet3, unit_id)
        (timestamps, #group, fet0, fet1, fet2, fet3, unit_id)
        (timestamps, #group, fet0, fet1, fet2, fet3, unit_id)
        (timestamps, #group, fet0, fet1, fet2, fet3, unit_id)
        ...
        '''
        grps = np.unique(fet[:,1])
        # n_units = np.unique(fet[:,-1])
        for i_grp in grps:
            _fet = fet[fet[:,1]==i_grp][:, 2:6]/scale_factor
            _clu = fet[fet[:,1]==i_grp][:, -1].astype(np.int32)
            self.fet_view[i_grp].set_data(_fet, _clu)
            self.fet_view[i_grp].register_event()

    def from_file(self, unit_packet_binfile):
        '''
        The fet.bin
        '''
        self.fet = np.fromfile(unit_packet_binfile, dtype=np.int32).reshape(-1,7)
        self.load_units(self.fet)



    # def set_data(self, groupNo, fet, clu):
    #     self.fet_view[groupNo].set_data(fet, clu)
    #     self.fet_view[groupNo].show()

# if __name__ == '__main__':
#     appQt = QtGui.QApplication(sys.argv)
#     rows, cols = 6, 7
#     win = FeatureSpaces(rows, cols)
#     win.show()

#     N = 10000
#     fet = np.random.randn(N, 60)
#     clu = np.zeros((N,)).astype(np.int)
#     for idx in range(rows*cols):
#         with Timer('fet_view[{0}].set_data'.format(idx)):
#             win.fet_view[idx].set_data(fet, clu)
#             win.fet_view[idx]._timer.start()
#     # win.fet_view[0].highlight(np.arange(1000))
#     appQt.exec_()
