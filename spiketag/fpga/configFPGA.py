import numpy as np
from spiketag.fpga.memory_api import read_mem_16, write_mem_16
from spiketag.fpga import bram_thres
from spiketag.fpga import channel_hash

def calculate_threshold(x, beta=4.0):
    thr = -beta*np.median(abs(x)/0.6745,axis=0)
    return thr

class xike_config(object):

    """Docstring for MyClass. """

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
        self._thres_value = thres_value
        self.config()
        
    def config(self):
        # dc offset
        write_mem_16(0, 1)
        self.dc = bram_thres.offset(nCh=self._n_ch)
        self.dc[:] = np.ones((self._n_ch,)) * self._offset_value
        # threshold
        self.thres = bram_thres.threshold(nCh=self._n_ch)
        self.thres.enable(True)
        self.thres[:] = self._thres_value
        # channel group hashing
        self.ch_ugp = channel_hash(nCh=self._n_ch, base_address=256)
        for ch in range(self._n_ch):
            self.ch_ugp[ch] = self.probe.ch_hash(ch)

# if __name__ == "__main__":
   # config(offset_value = 32, thres_value = -500)

