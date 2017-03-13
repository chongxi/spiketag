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
    signal_objet_changed  = QtCore.pyqtSignal(name='objectChanged')
    signal_ch_changed     = QtCore.pyqtSignal(name='chChanged')
    signal_get_fet        = QtCore.pyqtSignal(name='getfet')
    signal_recluster      = QtCore.pyqtSignal(name='recluster')
    signal_refine         = QtCore.pyqtSignal(name='refine')
    signal_build_vq       = QtCore.pyqtSignal(name='vq')
    signal_apply_to_all   = QtCore.pyqtSignal(name='apply2all')
    signal_wave_view_zoom = QtCore.pyqtSignal(name='zoom') 
    
    def __init__(self, parent=None):
        super(param_widget, self).__init__(parent)

        l_fet_method = QtGui.QLabel("feature")
        self.fet_method = list(['pca', 'weighted-pca', 'ica', 'weighted-ica', 'peak'])
        self.fet_combo = QtGui.QComboBox(self)
        self.fet_combo.addItems(self.fet_method)
        self.fet_combo.setCurrentIndex(1)
        self.fet_combo.currentIndexChanged.connect(self.get_fet)

        l_fet_No = QtGui.QLabel("fetNo")
        self.fet_No = QtGui.QSpinBox()
        self.fet_No.setMinimum(1)
        self.fet_No.setMaximum(12)
        self.fet_No.setValue(3)
        self.fet_No.valueChanged.connect(self.get_fet)

        l_clu_method = QtGui.QLabel("clutering")
        self.clu_method = list(['hdbscan', 'dpc', 'kmeans', 'gmm'])
        self.clu_combo = QtGui.QComboBox(self)
        self.clu_combo.addItems(self.clu_method)
        self.clu_combo.currentIndexChanged.connect(self.update_param)

        l_ch = QtGui.QLabel("Channel")
        self.ch = QtGui.QSpinBox()
        self.ch.setMinimum(0)
        self.ch.setMaximum(31)
        self.ch.setValue(26)
        self.ch.valueChanged.connect(self.update_ch)

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
        self.vq_btn = QtGui.QPushButton("build vq")
        # self.vq_btn.setCheckable(True)
        self.vq_btn.clicked.connect(self.build_vq)

        self.apply_to_all = QtGui.QCheckBox('Apply to all channels')
        self.apply_to_all.stateChanged.connect(self.apply_to_all_changed)

        self.wave_view_zoom_text = QtGui.QLabel("wave_view_zoom:")
        self.wave_view_zoom = QtGui.QSlider(1)
        self.wave_view_zoom.setMinimum(200)
        self.wave_view_zoom.setMaximum(800)
        self.wave_view_zoom.setValue(200)
        self.wave_view_zoom.setTickPosition(QtGui.QSlider.TicksBelow)
        self.wave_view_zoom.setTickInterval(3)
        self.wave_view_zoom.valueChanged.connect(self.zoom) 

        gbox = QtGui.QGridLayout()
        # addWidget (QWidget, int row, int column, int rowSpan, int columnSpan, Qt.Alignment alignment = 0)
        gbox.addWidget(l_fet_method, 0, 0)
        gbox.addWidget(self.fet_combo, 0, 1)
        gbox.addWidget(l_fet_No, 1, 0)
        gbox.addWidget(self.fet_No, 1, 1)
        gbox.addWidget(l_clu_method, 2, 0)
        gbox.addWidget(self.clu_combo, 2, 1)
        gbox.addWidget(l_ch, 3, 0)
        gbox.addWidget(self.ch, 3, 1)
        gbox.addWidget(self.clu_param_text, 4, 0)
        gbox.addWidget(self.recluster_btn, 5, 0)
        gbox.addWidget(self.refine_btn, 5, 1)
        gbox.addWidget(self.clu_param, 6, 0, 1, 2)
        gbox.addWidget(self.vq_btn, 7, 0, 1, 1)
        gbox.addWidget(self.apply_to_all, 8, 0, 1, 2)
        gbox.addWidget(self.wave_view_zoom_text, 9, 0)
        gbox.addWidget(self.wave_view_zoom, 10, 0, 1, 2)

        vbox = QtGui.QVBoxLayout()
        vbox.addLayout(gbox)
        vbox.addStretch(1.0)

        self.setLayout(vbox)


    def update_ch(self, option):
        self.signal_ch_changed.emit()

    def get_fet(self, option):
        self.signal_get_fet.emit()

    def recluster(self, option):
        self.signal_recluster.emit()

    def refine(self, option):
        self.signal_refine.emit()

    def update_clu_param(self, option):
        # self.clu_param_changed.emit()
        text = "fall-off-size: " + str(self.clu_param.value())
        self.clu_param_text.setText(text)

    def build_vq(self, option):
        self.signal_build_vq.emit()

    def apply_to_all_changed(self, state):
        if state == QtCore.Qt.Checked:
            self._apply_to_all = True
        else:
            self._apply_to_all = False
        self.signal_apply_to_all.emit()

    def update_param(self, option):
        self.signal_objet_changed.emit()

    def zoom(self, option):
        self.signal_wave_view_zoom.emit()
