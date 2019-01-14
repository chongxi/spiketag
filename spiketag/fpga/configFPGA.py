import numpy as np
from .memory_api import read_mem_16, write_mem_16
from .bram_xike  import pca_hash, scale_hash, shift_hash, vq_hash
from . import bram_thres


def calculate_threshold(x, beta=4.0):
    thr = -beta*np.median(abs(x)/0.6745,axis=0)
    return thr

class xike_config(object):

    def __init__(self, probe, offset_value=32, thres_value=-500):
        """TODO: to be defined1.
        :probe: spiketag::probe
        :offset_value: TODO
        :thres_value: TODO
        :ch_hash: TODO

        """
        self._n_ch = probe.n_ch
        self.probe = probe
        self._offset_value = offset_value
        # self._thres_value = thres_value
        self.set_channel_group()
        self.init_FPGA_detector()
        self.transfomer_constructed = {}
        for i in range(self.probe.n_group):
            self.transfomer_constructed[i]=0
        '''
        y = a(xP+b)
        x: (spklen*ch_span,)
        P: (spklen*ch_span, 4)  : pca[grpNo]
        b: (4, )                : shift[grpNo]
        a: ()                   : scale[grpNo]
        '''
        ngrp    = self.probe.n_group     # 40 for tetrodes
        ch_span = self.probe.group_len   # 4  for tetrodes ;  40*4=160 chs
        spklen  = 19                     # 19
        p_dim   = 4
        self.scale = scale_hash(nCh=ngrp, base_address=0)
        self.shift = shift_hash(nCh=ngrp, base_address=ngrp * 1)
        self.pca   =   pca_hash(nCh=ngrp, base_address=ngrp * (p_dim + 1))
        self.vq    =    vq_hash(nCh=ngrp, base_address=ngrp * (p_dim + 1 + spklen*ch_span))


    def set_channel_group(self):
        # channel group hashing
        # the channel hashing for spike grouping protocol in FPGA
        self.ch_hash = bram_thres.channel_hash(nCh=self._n_ch, base_address=256)
        # the channel groupNo for transformer to report in FPGA
        self.ch_grpNo = bram_thres.chgpNo(nCh=self._n_ch)
        self.ch_ref = bram_thres.ch_ref(self._n_ch)
        for ch in range(self.probe.n_ch):
            self.ch_hash[ch] = self.probe.ch_hash(ch)
            try:
                self.ch_grpNo[ch] = self.probe.ch2g[ch]
            except:
                self.ch_grpNo[ch] = 100            

    def set_channel_ref(self, ch_ref):
        self.ch_ref[:] = ch_ref

    def set_threshold(self, threshold):
        self.thres[:] = threshold

    def init_FPGA_detector(self):
        # dc offset
        write_mem_16(0, 1)
        self.dc = bram_thres.offset(nCh=self._n_ch)
        self.dc[:] = np.ones((self._n_ch,)) * self._offset_value
        # threshold
        self.thres = bram_thres.threshold(nCh=self._n_ch)
        # self.thres.enable(True)
        # self.thres[:] = self._thres_value


    def _config_FPGA_transformer(self, grpNo, P, b, a):
        self.transfomer_constructed[grpNo] = 1
        self.scale[grpNo] = a
        self.shift[grpNo] = b
        self.pca[grpNo]   = P

    def _config_FPGA_vq_knn(self, grpNo, vq):
        self.vq[grpNo] = vq


# if __name__ == "__main__":
   # config(offset_value = 32, thres_value = -500)

