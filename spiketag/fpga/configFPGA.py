import numpy as np
from .memory_api import read_mem_16, write_mem_16
from .bram_xike  import pca_hash, scale_hash, shift_hash, vq_hash
from . import bram_thres


def calculate_threshold(x, beta=4.0):
    thr = -beta*np.median(abs(x)/0.6745,axis=0)
    return thr

class dummyobj(object):
    pass

class xike_config(object):

    def __init__(self, probe=None, offset_value=32, thres_value=-500):
        """TODO: to be defined1.
        :offset_value: TODO
        :thres_value: TODO
        :ch_hash: TODO

        """
        if probe is None:
            probe = dummyobj()
            probe.n_ch      = 160
            probe.fs        = 25000.
            probe.n_group   = 40
            probe.group_len = 4
            _set_channel_params_to_fpga = False
        else:
            _set_channel_params_to_fpga = True

        self._n_ch = probe.n_ch
        self.probe = probe

        '''
        1. ch_hash, ch_grpNo and ch_ref
        '''
        # the channel hashing for spike grouping protocol in FPGA
        self.ch_hash = bram_thres.channel_hash(nCh=self._n_ch, base_address=256)
        # the channel groupNo for transformer to report in FPGA
        self.ch_grpNo = bram_thres.chgpNo(nCh=self._n_ch)
        self.ch_ref = bram_thres.ch_ref(self._n_ch)

        '''
        1. if probe is selected then write to FPGA
        '''
        if _set_channel_params_to_fpga:
            self.set_channel_params_to_fpga()

        '''
        2. dc_offset and threshold
        '''
        self.dc = bram_thres.offset(nCh=self._n_ch)
        self.dc[:] = np.ones((self._n_ch,)) * offset_value
        self.thres = bram_thres.threshold(nCh=self._n_ch)

        '''
        3. Transformer and VQ
        y = a(xP+b)
        x: (spklen*ch_span,)
        P: (spklen*ch_span, 4)  : pca[grpNo]
        b: (4, )                : shift[grpNo]
        a: ()                   : scale[grpNo]
        '''
        self._transformer_status = np.zeros((self.probe.n_group,))

        ngrp    = self.probe.n_group     # 40 for tetrodes
        ch_span = self.probe.group_len   # 4  for tetrodes ;  40*4=160 chs
        spklen  = 19                     # 19
        p_dim   = 4
        self.scale = scale_hash(nCh=ngrp, base_address=0)
        self.shift = shift_hash(nCh=ngrp, base_address=ngrp * 1)
        self.pca   =   pca_hash(nCh=ngrp, base_address=ngrp * (p_dim + 1))
        self.vq    =    vq_hash(nCh=ngrp, base_address=ngrp * (p_dim + 1 + spklen*ch_span))

    def set_channel_params_to_fpga(self):
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
        write_mem_16(0, 1)


    def _config_FPGA_probe(self, prb):
        self.probe = prb
        self.set_channel_params_to_fpga()

    def _config_FPGA_ch_ref(self, grpNo, ch_ref):
        for ch in self.probe[grpNo]:
            self.ch_ref[ch] = ch_ref  

    def _config_FPGA_thres(self, grpNo, threshold):
        for ch in self.probe[grpNo]:
            self.thres[ch] = threshold

    def _config_FPGA_transformer(self, grpNo, P, b, a):
        self.scale[grpNo] = a
        self.shift[grpNo] = b
        self.pca[grpNo]   = P

    def _config_FPGA_vq_knn(self, grpNo, vq):
        self.vq[grpNo] = vq

    @property
    def transformer_status(self):
        for i in range(self.probe.n_group):
            self._transformer_status[i] = self.scale[i] != 0
        return self._transformer_status
    


# if __name__ == "__main__":
   # config(offset_value = 32, thres_value = -500)

