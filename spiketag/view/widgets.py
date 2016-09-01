import numpy as np
from PyQt4 import QtGui, QtCore
from phy.plot import View
from phy.io.array import _accumulate


def _representsInt(s):
    try: 
        int(s)
        return True
    except ValueError:
        return False


class param_widget(QtGui.QWidget):
    """
    Widget for editing OBJECT parameters
    """
    signal_objet_changed = QtCore.pyqtSignal(name='objectChanged')
    signal_ch_changed    = QtCore.pyqtSignal(name='chChanged')
    # clu_param_changed    = QtCore.pyqtSignal(name='cluparamChanged')
    signal_recluster     = QtCore.pyqtSignal(name='recluster')
    signal_refine        = QtCore.pyqtSignal(name='refine')


    def __init__(self, parent=None):
        super(param_widget, self).__init__(parent)

        l_fet_method = QtGui.QLabel("feature")
        self.fet_method = list(['pca', 'peak'])
        self.fet_combo = QtGui.QComboBox(self)
        self.fet_combo.addItems(self.fet_method)
        self.fet_combo.currentIndexChanged.connect(self.update_param)

        l_clu_method = QtGui.QLabel("clutering")
        self.clu_method = list(['hdbscan', 'dpc', 'kmeans', 'gmm'])
        self.clu_combo = QtGui.QComboBox(self)
        self.clu_combo.addItems(self.clu_method)
        self.clu_combo.currentIndexChanged.connect(self.update_param)

        self.clu_param_text = QtGui.QLabel("fall-off-size: 18")
        self.clu_param  = QtGui.QSlider(1) # 1: horizontal, 2: Vertical
        self.clu_param.setMinimum(3)
        self.clu_param.setMaximum(200)
        self.clu_param.setValue(25)
        self.clu_param.setTickPosition(QtGui.QSlider.TicksBelow)
        self.clu_param.setTickInterval(3)
        self.clu_param.valueChanged.connect(self.update_clu_param)


        self.recluster_btn = QtGui.QPushButton('re-cluster')
        self.recluster_btn.clicked.connect(self.recluster)
        self.refine_btn = QtGui.QPushButton("refine")
        self.refine_btn.clicked.connect(self.refine)


        l_ch = QtGui.QLabel("Channel")
        self.ch = QtGui.QSpinBox()
        self.ch.setMinimum(0)
        self.ch.setMaximum(31)
        self.ch.setValue(26)
        self.ch.valueChanged.connect(self.update_ch)

        gbox = QtGui.QGridLayout()
        # addWidget (QWidget, int row, int column, int rowSpan, int columnSpan, Qt.Alignment alignment = 0)
        gbox.addWidget(l_fet_method, 0, 0)
        gbox.addWidget(self.fet_combo, 0, 1)
        gbox.addWidget(l_clu_method, 1, 0)
        gbox.addWidget(self.clu_combo, 1, 1)
        gbox.addWidget(l_ch, 2, 0)
        gbox.addWidget(self.ch, 2, 1)
        gbox.addWidget(self.clu_param_text, 3, 0)
        gbox.addWidget(self.recluster_btn, 4, 0)
        gbox.addWidget(self.refine_btn, 4, 1)
        gbox.addWidget(self.clu_param, 5, 0, 1, 2)

        vbox = QtGui.QVBoxLayout()
        vbox.addLayout(gbox)
        vbox.addStretch(1.0)

        self.setLayout(vbox)

    def recluster(self, option):
        self.signal_recluster.emit()

    def refine(self, option):
        self.signal_refine.emit()

    def update_clu_param(self, option):
        # self.clu_param_changed.emit()
        text = "fall-off-size: " + str(self.clu_param.value())
        self.clu_param_text.setText(text)

    def update_ch(self, option):
        self.signal_ch_changed.emit()

    def update_param(self, option):
        self.signal_objet_changed.emit()
