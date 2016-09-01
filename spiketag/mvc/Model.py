import numpy as np
import logging
import multiprocessing
from sklearn.neighbors import KDTree
from ..base import *


def log_start(level=logging.INFO):
    logger = multiprocessing.log_to_stderr()
    logger.setLevel(level)  # DEBUG, INFO, WARNING, ERROR, CRITICAL
    logger.propagate = False

log_start(level=logging.INFO)
info = multiprocessing.get_logger().info


class MainModel(object):
    """
    filename is the mua binary file
    spktag_filename is the spiketag file, if None, Model will do clustering
    other parameters are metadata of the binary file

    Model contains four sub-objects:
       -- mua: MUA; mua.tospk
       -> spk: SPK; spk.tofet
       -> fet: FET; fet.toclu, fet.get
       -> clu: dict, every item is a CLU on that channel; clu.merge, clu.split, clu.move  etc..
    """

    def __init__(self, filename, spktag_filename=None, 
                 nCh=32, fs=25000, numbytes=4, binary_radix=14,
                 ch_span=1, spklen=25,
                 corr_cutoff=0.9,
                 fet_method='weighted-pca', fetlen=6, fet_whiten=False,
                 clu_method='hdbscan', fall_off_size=18, n_jobs=24):

        # raw recording param
        self.filename = filename
        self.spktag_filename = spktag_filename
        self.nCh = nCh
        self.fs = fs
        self.numbytes = numbytes
        self.binpoint = binary_radix

        # mua param
        self._corr_cutoff = corr_cutoff
        self._ch_span = ch_span
        self._spklen = 25

        # fet param
        self.fet_method = fet_method
        self._fet_whiten = fet_whiten
        self._fetlen = fetlen

        # clu param
        self.clu_method = clu_method
        self._fall_off_size = fall_off_size
        self._n_jobs = n_jobs

        self._model_init_(self.spktag_filename)


    def _model_init_(self, spktag_filename=None):
        '''
        If spktag_filename is given, Model will generate spk,fet,clu from
        loading spktag array, rather than compute it from mua. 
        Otherwise, it would assume that there is no stored infomation and 
        everything needs to be calculated from mua
        '''
        info('load mua data')
        self.mua = MUA(self.filename, self.nCh, self.fs,
                       self.numbytes, self.binpoint)

        # The first time
        if spktag_filename is None:
            info('removing high corr noise from spikes pool')
            self.mua.remove_high_corr_noise(corr_cutoff=self._corr_cutoff)

            info('extract spikes from pivital meta data')
            self.spk = self.mua.tospk(ch_span=self._ch_span)

            info('extrat features with {}'.format(self.fet_method))
            self.fet = self.spk.tofet(method=self.fet_method, 
                                      whiten=self._fet_whiten,
                                      ncomp=self._fetlen)

            info('clustering with {}'.format(self.clu_method))
            self.clu = self.fet.toclu(method=self.clu_method, 
                                      fall_off_size=self._fall_off_size,
                                      njobs=self._n_jobs)

            self.spktag = SPKTAG(self.nCh, 
                                 self._ch_span,
                                 self.mua.pivotal_pos, 
                                 self.spk, 
                                 self.fet, 
                                 self.clu)
            info('Model.spktag is generated, ch_span:{}, nspk:{}'.format(self._ch_span, self.spktag.nspk))

        # After first time
        else:
            self.spktag = SPKTAG()
            info('load spktag file')
            self.spktag.fromfile(spktag_filename)
            self.spk = self.spktag.tospk()
            self.fet = self.spktag.tofet()
            self.clu = self.spktag.toclu()


    def cluster(self, method='hdbscan', *args, **kwargs):
        chNo = kwargs['chNo'] if 'chNo' in kwargs.keys() else None
        if chNo is not None:
            self.clu[chNo] = self.fet.toclu(method=method, *args, **kwargs)
        else:
            pass


    def construct_kdtree(self, chNo):
        self.kd = []
        for clu_id, value in self.clu[chNo].index.items():
            fet = self.fet[chNo][value]
            self.kd.append(KDTree(fet))


    def predict(self, chNo, X, method='knn', k=10):
        if X.ndim==1: X=X.reshape(1,-1)
        if method == 'knn':
            self.construct_kdtree(chNo)
            d = []
            k = 10
            for _kd in self.kd:
                tmp = _kd.query(X, k)[0]
                d.append(tmp.mean(axis=1))
            d = np.vstack(np.asarray(d))
            # np.argmin(d[0:,:],axis=0)
            labels = np.argmin(d[1:,:],axis=0)+1
        return labels


    def tofile(self, filename=None):
        '''
        This should automatically update the spktag array and save
        So that next time it can be loaded and avoid re-clustering
        '''
        self.spktag.update(self.spk, self.fet, self.clu)
        if filename is not None:
            self.spktag.tofile(filename)
        elif self.spktag_filename is not None:
            self.spktag.tofile(self.spktag_filename)
        else:
            barename = self.filename.split('.')[0]
            self.spktag_filename = barename + '_spktag.bin'
            self.spktag.tofile(self.spktag_filename)

