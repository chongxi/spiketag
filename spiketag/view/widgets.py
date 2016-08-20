import numpy as np
import seaborn as sns
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

        l_clu_param = QtGui.QLabel("fall-off-size")
        self.clu_param  = QtGui.QSlider(1) # 1: horizontal, 2: Vertical
        self.clu_param.setMinimum(3)
        self.clu_param.setMaximum(100)
        self.clu_param.setValue(5)
        self.clu_param.setTickPosition(QtGui.QSlider.TicksBelow)
        self.clu_param.setTickInterval(1)
        self.clu_param.valueChanged.connect(self.update_param)

        l_ch = QtGui.QLabel("Channel")
        self.ch = QtGui.QSpinBox()
        self.ch.setMinimum(0)
        self.ch.setMaximum(31)
        self.ch.setValue(26)
        self.ch.valueChanged.connect(self.update_param)

        gbox = QtGui.QGridLayout()
        # addWidget (QWidget, int row, int column, int rowSpan, int columnSpan, Qt.Alignment alignment = 0)
        gbox.addWidget(l_fet_method, 0, 0)
        gbox.addWidget(self.fet_combo, 0, 1)
        gbox.addWidget(l_clu_method, 1, 0)
        gbox.addWidget(self.clu_combo, 1, 1)
        gbox.addWidget(l_ch, 2, 0)
        gbox.addWidget(self.ch, 2, 1)
        gbox.addWidget(l_clu_param, 3, 0)
        gbox.addWidget(self.clu_param, 4, 0, 1, 2)

        vbox = QtGui.QVBoxLayout()
        vbox.addLayout(gbox)
        vbox.addStretch(1.0)

        self.setLayout(vbox)

    def update_param(self, option):
        self.signal_objet_changed.emit()



class spike_view(View):
    def __init__(self, interactive=False):
        super(spike_view, self).__init__('grid')
        self.palette = sns.color_palette()
        self._transparency = 0.1
        self.interactive = interactive
        self._selected_cluster = 0
        self._view_lock = False
        self._spkNolist = set()
        

    def attach(self, gui):
        gui.add_view(self)
        

    def set_data(self, spk, ch_base, span_ch=1, clu=None):
        self.span_ch = span_ch
        self.ch_base = ch_base
        if clu is None:
            clu = np.zeros((spk[ch_base].shape[0],)).astype('int32')
        nclu = len(np.unique(clu))
        num_ch = span_ch*2+1
        self.clu = clu
        self._clu_cumsum = self._get_clu_cumsum(clu)
        self._clu_nspks  = self._get_clu_nspks(clu)
        self.clear()
        self.grid.shape = (num_ch, nclu)
        for chNo in range(num_ch):
            for cluNo in np.unique(clu):
                color = self.palette[cluNo] if cluNo>0 else np.array([1,1,1])
                self[chNo,cluNo].plot( y     = spk[ch_base][clu==cluNo,:,chNo].squeeze(),
                                       color = np.hstack((color, self._transparency)),
                                       data_bounds=(-2, -2000, 2, 1000))

        self.spk = spk
        self.ch_base = ch_base
        self.clu = clu
        self.n_signals = spk[ch_base].shape[0]
        self.n_samples = spk[ch_base].shape[1]
        self.n_ch      = spk[ch_base].shape[2]

        self._xsig = np.linspace(-0.5, 0.5, self.n_samples)
        self.build_engine()
        

    def build_engine(self):
        for cls, data_list in self._items.items():
            # Some variables are not concatenated. They are specified
            # in `allow_list`.
            data = _accumulate(data_list, cls.allow_list)
            box_index = data.pop('box_index')
            visual = cls()
            self.add_visual(visual)
            # important!, this map the spk to view space through a affine transformation: y = ax + b
            # self.y = self._affine_transform(self.spk[self.ch_base].transpose(2,0,1).ravel())
            self.y = self._affine_transform(np.asarray(data['y']).ravel())
            self.x = np.tile(self._xsig, len(self.y)/self.n_samples)
            self.color = np.repeat(data['color'], self.n_samples, axis=0)

            visual.set_data(**data)
            if 'a_box_index' in visual.program._code_variables:
                visual.program['a_box_index'] = box_index.astype(np.float32)

    @property
    def transparency(self):
        return self._transparency

    @transparency.setter
    def transparency(self, v):
        self._transparency = v
        if self._transparency >= 0.9:
            self._transparency = 0.9
        elif self._transparency <= 0.001:
            self._transparency = 0.001
        self.color[:,-1] = self._transparency
        self.visuals[0].program['a_color'] = self.color
        self.update()

    @staticmethod
    def _idx_subclu(clu, cluNo, fulidx_list):
        subclu_idx = np.where(clu==cluNo)[0]
        return np.searchsorted(subclu_idx, fulidx_list)

    @staticmethod
    def _idx_fulclu(clu, cluNo, subidx_list):
        return np.where(clu==cluNo)[0][subidx_list]

    @staticmethod
    def _get_clu_cumsum(clu):
        spks_per_clu = [0,]
        for cluid in np.unique(clu):
            spks_per_clu.append(sum(clu==cluid))
        csum = np.cumsum(np.asarray(spks_per_clu))
        return csum

    @staticmethod
    def _get_clu_nspks(clu):
        spks_per_clu = []
        for cluid in np.unique(clu):
            spks_per_clu.append(sum(clu==cluid))
        spks_per_clu = np.asarray(spks_per_clu)
        return spks_per_clu

    @staticmethod
    def _affine_transform(x):
        #TODO: automatically calculate from data_bounds (-2000,1000) -> (-1,1) to (a,b)
        # that gives the affine transformation y = ax + b
        a = 1/1500.0
        b = 1/3.0
        y = a*x + b
        return y


    def _spkNo2maskNo(self, spkNolist, cluNo):
        mask = []
        n_signals=self.n_signals
        n_samples=self.n_samples
        n_ch     =self.n_ch
        clu_offset = self._clu_cumsum
        for spkNo in spkNolist:
            for ch in range(n_ch):
                # start = n_samples*(spkNo + ch*n_signals)
                start = n_samples*(spkNo + ch*n_signals + clu_offset[cluNo])
                # end   = n_samples*(spkNo + ch*n_signals +1)
                end   = n_samples*(spkNo + ch*n_signals + clu_offset[cluNo] + 1)
                mask.append(np.arange(start, end))
        return mask


    def highlight(self, spkNolist, cluNo, highlight_color=[1,0,0,0.7]):
        """
        highlight the nth spike at near electrodes

        """
        view_mask = self._spkNo2maskNo(spkNolist=spkNolist, cluNo=cluNo)
        color = self.color.copy()
        depth = np.zeros(*self.x.shape)
        for _mask_ in view_mask:
            color[_mask_] = np.asarray(highlight_color).astype(np.float32)
            depth[_mask_] = -1
        pos = np.c_[self.x, self.y, depth]
        self.visuals[0].program['a_color'] = color
        self.visuals[0].program['a_position'] = pos.astype(np.float32)  # pos include depth
        self.update()


    def _highlight_method2(self, spk, ch, spkNo, color=[1,0,0,0.7]):
        """
        highlight the nth spike, for checking highlight is correct
        , while method 1 is high performance

        """
        color = np.asarray(color)
        num_ch = self.span_ch*2+1
        for chNo in range(num_ch):
            self[chNo,0].plot(y = spk[ch][spkNo][:,chNo],
                              color = color,
                              data_bounds=(-2, -2000, 2, 1000))
        self.build()


    def _unhighligh_method2(self):
        self.visuals.pop()
        self.update()


    def _data_in_box(self, box):
        ch_No = box[0]
        clu_No = box[1]
        data_in_box = self.spk[self.ch_base][self.clu==clu_No, :, ch_No].squeeze()
        data_in_box = self._affine_transform(data_in_box)
        return data_in_box


    def _get_closest_spk(self, box, pos):
        data = self._data_in_box(box)
        nearest_x = abs(pos[0] - self._xsig).argmin()
        nearest_spkNo = abs(pos[1] - data[:, nearest_x]).argmin()
        return nearest_spkNo

    def _get_close_spks(self, box, pos):
        data = self._data_in_box(box)
        nearest_x = abs(pos[0] - self._xsig).argmin()
        close_spkNolist = np.where(abs(pos[1] - data[:, nearest_x]) < 0.01)[0]
        return close_spkNolist


    @property
    def selected_clu(self):
        return self._selected_cluster

    @property
    def selected_spk(self):
        # this is the spkNolist in the whole clu, not the sub cluster,
        # incase you are selecting in the subclu, the index in subclu is mapped to full clu
        # selected_spks = np.where(self.clu==self._selected_cluster)[0][list(self._spkNolist)]
        # self._spkNolist is the local idx in the subcluster
        # use _idx_fulclu to map the local idx to full cluster idx
        selected_spk = self._idx_fulclu(self.clu, self._selected_cluster, list(self._spkNolist))
        return selected_spk


    def on_mouse_move(self, e):
        # if e.button == 3:
        if self.interactive is True:
            ndc = self.panzoom.get_mouse_pos(e.pos)
            box = self.interact.get_closest_box(ndc)
            tpos = self.get_pos_from_mouse(e.pos, box)[0]
            
            modifiers = e.modifiers
            if modifiers is () and not isinstance(e.button, int):
                if self._view_lock is False:
                    spkNo = self._get_closest_spk(box, tpos)
                    self.highlight(spkNolist=(spkNo,), cluNo=box[1])

            if modifiers is not ():
                self._selected_cluster = box[1]
                if len(modifiers) ==2 and modifiers[0].name == 'Shift' and modifiers[1].name == 'Control':
                    # spkNo = self._get_closest_spk(box, tpos)
                    # self._spkNolist.add(spkNo)
                    close_spkNolist = self._get_close_spks(box, tpos)
                    intersect_spks  = np.intersect1d(self._spkNolist, list(close_spkNolist))
                    self._spkNolist = self._spkNolist.union(close_spkNolist)
                    self.highlight(spkNolist=self._spkNolist, cluNo=self._selected_cluster) 

                elif len(modifiers) ==1 and modifiers[0].name == 'Control':
                    # self._spkNolist = []
                    # spkNo = self._get_closest_spk(box, tpos)
                    # self.highlight(spkNolist=(spkNo,), cluNo=self._selected_cluster) 
                    self._spkNolist = set()
                    close_spkNolist = self._get_close_spks(box, tpos)
                    self.highlight(spkNolist=close_spkNolist, cluNo=self._selected_cluster) 

                elif len(modifiers) ==1 and modifiers[0].name == 'Shift':
                    close_spkNolist = self._get_close_spks(box, tpos)
                    intersect_spks  = np.intersect1d(list(self._spkNolist), list(close_spkNolist))
                    if len(intersect_spks)>0:
                        for spkNo in intersect_spks:
                            self._spkNolist.remove(spkNo)
                    self.highlight(spkNolist=self._spkNolist, cluNo=self._selected_cluster) 


    def on_mouse_wheel(self, e):
        modifiers = e.modifiers
        if modifiers is not ():
            if modifiers[0].name == 'Shift':
                self.transparency *= np.exp(e.delta[1]/4)


    def on_key_press(self, e):
        if e.text == 'c':
            self._view_lock = not self._view_lock
            self.highlight(spkNolist=self._spkNolist, cluNo=self._selected_cluster) 
        if e.text == 's':
            ### split ###
            # print self.selected_clu, self.selected_spk
            if len(self.selected_spk) > 0:
                # assign selected spks to new cluster, number is maxclu + 1
                newcluNo = max(np.unique(self.clu)) + 1
                self.clu[self.selected_spk] = newcluNo
                self._spkNolist = set(np.arange(len(self._spkNolist)))
                self._selected_cluster = newcluNo
                self.set_data(self.spk, self.ch_base, self.span_ch, self.clu)

        if _representsInt(e.text):
            target_clu_No = int(e.text)
            if len(self.selected_spk) > 0:
                
                #TODO: self._spkNolist = #spks in target_clu -> #spks + #spks_tobe_appended
                _spkNolist_in_targetclu = set(self._idx_subclu(self.clu, 
                                target_clu_No, list(self.selected_spk)))
                print _spkNolist_in_targetclu
                self.clu[self.selected_spk] = target_clu_No
                self._spkNolist = _spkNolist_in_targetclu
                # assign selected spks to cluster with known No
                self._selected_cluster = target_clu_No
                self.set_data(self.spk, self.ch_base, self.span_ch, self.clu)
