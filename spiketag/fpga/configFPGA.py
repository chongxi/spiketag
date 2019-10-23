import numpy as np
from .memory_api import read_mem_16, write_mem_16
from .bram_xike  import pca_hash, scale_hash, shift_hash, vq_hash, label_hash
from . import bram_thres


def calculate_threshold(x, beta=4.0):
    thr = -beta*np.median(abs(x)/0.6745,axis=0)
    return thr

class dummyobj(object):
    pass

class xike_config(object):

    def __init__(self, probe=None):
        """TODO: to be defined1.
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

        self.n_ch = 160 # probe.n_ch
        self.probe = probe

        '''
        1. ch_hash, ch_grpNo and ch_ref
        '''
        # the channel hashing for spike grouping protocol in FPGA
        self.ch_hash = bram_thres.channel_hash(nCh=self.n_ch, base_address=256)
        # the channel groupNo for transformer to report in FPGA
        self.ch_grpNo = bram_thres.chgpNo(nCh=self.n_ch)
        self.ch_ref = bram_thres.ch_ref(self.n_ch)

        '''
        1. if probe is selected then write to FPGA
        '''
        if _set_channel_params_to_fpga:
            self.set_channel_params_to_fpga()

        '''
        2. dc_offset and threshold
        '''
        self.dc = bram_thres.offset(nCh=self.n_ch)
        self.dc[:] = np.ones((self.n_ch,)) * 32 # emprical, this need some invesigation why the offset exists
        self.thres = bram_thres.threshold(nCh=self.n_ch)

        '''
        3. Transformer and VQ
        y = a(xP+b)
        x: (spklen*ch_span,)
        P: (spklen*ch_span, 4)  : pca[grpNo]
        b: (4, )                : shift[grpNo]
        a: ()                   : scale[grpNo]
        transformer_status track a
        '''
        # ngrp    = self.probe.n_group     # 40 for tetrodes
        self.ngrp    = 40 # now fixed in current version of FPGA
        self.ch_span = 4  # 4  for tetrodes ;  40*4=160 chs (fixed for current version of FPGA)
        self.spklen  = 19                     # 19
        self.p_dim   = 4
        self.n_vq    = 500
        self._transformer_status = np.zeros((self.ngrp,))

        self.scale = scale_hash(nCh=self.ngrp,  base_address=0)
        self.shift = shift_hash(nCh=self.ngrp,  base_address=self.ngrp * 1)
        self.pca   =   pca_hash(nCh=self.ngrp,  base_address=self.ngrp * (self.p_dim + 1))
        self.vq    =    vq_hash(nCh=self.ngrp,  base_address=self.ngrp * (self.p_dim + 1 + self.spklen*self.ch_span))
        self.label = label_hash(nCh=self.ngrp,  base_address=self.ngrp * (self.p_dim + 1 + self.spklen*self.ch_span + self.n_vq))

    def set_channel_params_to_fpga(self):
        assert(self.n_ch == self.probe.n_ch)  # very important!
        for ch in range(self.probe.n_ch):
            ## it is possible that the probe.ch_hash is less than 40 groups (e.g. only 128 channels used)
            self.ch_hash[ch] = self.probe.ch_hash(ch)
            try:
                self.ch_grpNo[ch] = self.probe.ch2g[ch]
            except:
                self.ch_grpNo[ch] = 100            

    def set_channel_ref(self, ch_ref):
        self.ch_ref[:] = ch_ref

    def set_threshold(self, threshold):
        self.thres[:] = threshold

    '''
    ------------------------------------------------------------------------------------
    property interface with mem_reg_16 in the FPGA ([4:0] address for 32 slots)
    ------------------------------------------------------------------------------------
    '''

    @property
    def n_units(self):
        self._n_units = read_mem_16(0)
        return self._n_units

    @n_units.setter
    def n_units(self, val):
        write_mem_16(0, val)

    @property
    def target_unit(self):
        return read_mem_16(8)
    
    @target_unit.setter
    def target_unit(self, target_unit_id):
        if target_unit_id > self._n_units:
            print('cannot be bigger than {} configured units'.format(self.n_units))
        else:
            write_mem_16(8, target_unit_id)

    '''
    ------------------------------------------------------------------------------------
    ------------------------------------------------------------------------------------
    '''

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

    def _config_FPGA_vq_knn(self, grpNo, vq, label):
        self.vq[grpNo]    = vq
        self.label[grpNo] = label
    
    def reset_transformer(self):
        for grpNo in range(self.ngrp):
            self.scale[grpNo] = 0

    '''
    ------------------------------------------------------------------------------------
    property interface with `MODEL RAM` [15:0] address in the FPGA 
    5 sections: (scale, shift, pca, vq and label)
    ------------------------------------------------------------------------------------
    '''

    @property
    def transformer_status(self):
        for i in range(self.probe.n_group):
            self._transformer_status[i] = self.scale[i] != 0
        return self._transformer_status

    @property
    def configured_groups(self):
        return np.where(self.transformer_status)[0]


    '''
    ------------------------------------------------------------------------------------
    ------------------------------------------------------------------------------------
    '''