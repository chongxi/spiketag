import numpy as np
from .base.MUA import MUA
from .base.CLU import CLU

class Model(object):
    """
    Model contains four sub-objects:
       -- mua: MUA; mua.tospk
       -> spk: SPK; spk.tofet
       -> fet: FET; fet.toclu, fet.get
       -> clu: dict, every item is a CLU on that channel; clu.merge, clu.split, clu.move  etc..
    """
    def __init__(self, filename, nCh=32, ch_span=1, fs=25000, numbytes=4, fet_method='pca', clu_method='hdbscan',clu_njobs=1):
        self.filename   = filename
        self.nCh        = nCh
        self.ch_span    = ch_span
        self.fs         = fs
        self.numbytes   = numbytes
        self.fet_method = fet_method
        self.clu_method = clu_method
        self.clu_njobs  = clu_njobs
        self._model_init_()

    def _model_init_(self):
        self.mua = MUA(self.filename, self.nCh, self.fs, self.numbytes)
        self.spk = self.mua.tospk(ch_span = self.ch_span)
        self.fet = self.spk.tofet(method  = self.fet_method)
        self.clu = self.fet.toclu(method  = self.clu_method, njobs = self.clu_njobs)

    def save(self, file):
        np.savez(file, filename= self.filename,
                       spkdict = self.spk.spk,
                       fetdict = self.fet.fet)
