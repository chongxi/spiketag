import numpy as np
from sklearn.neighbors import KDTree
from ..base import *
from ..utils.conf import info 
from ..utils import conf
from ..utils.utils import Timer


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

    def __init__(self, filename, probe=None, spktag_filename=None, 
                 numbytes=4, binary_radix=13, spklen=19, corr_cutoff=0.9,
                 fet_method='weighted-pca', fetlen=6, fet_whiten=False,
                 clu_method='hdbscan', fall_off_size=18, n_jobs=24):

        # raw recording param
        self.filename = filename
        self.spktag_filename = spktag_filename
        self.probe = probe
        self.numbytes = numbytes
        self.binpoint = binary_radix

        # mua param
        self._corr_cutoff = corr_cutoff
        self._spklen = spklen 

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
         # The first time
        if spktag_filename is None:

            info('load mua data')
            self.mua = MUA(self.filename, self.probe, self.numbytes, self.binpoint)

            # info('removing high corr noise from spikes pool')
            # self.mua.remove_high_corr_noise(corr_cutoff=self._corr_cutoff)

            info('removing all spks on group which len(spks) less then fetlen')
            self.mua.remove_groups_under_fetlen(self._fetlen)

            info('extract spikes from pivital meta data')
            self.spk = self.mua.tospk()

            info('extrat features with {}'.format(self.fet_method))
            self.fet = self.spk.tofet(method=self.fet_method, 
                                      whiten=self._fet_whiten,
                                      ncomp=self._fetlen)

            info('clustering with {}'.format(self.clu_method))
            self.clu = self.fet.toclu(method=self.clu_method, 
                                      fall_off_size=self._fall_off_size,
                                      njobs=self._n_jobs)

            self.spktag = SPKTAG(self.probe,
                                 self.mua.pivotal_pos, 
                                 self.spk, 
                                 self.fet, 
                                 self.clu)
            info('Model.spktag is generated, nspk:{}'.format(self.spktag.nspk))

        # After first time
        else:
            self.spktag = SPKTAG()
            info('load spktag file')
            self.spktag.fromfile(spktag_filename)
            self.spk = self.spktag.tospk()
            self.fet = self.spktag.tofet()
            self.clu = self.spktag.toclu()

            info('load mua data for wave view')
            self.mua = MUA(self.filename, self.spktag.probe, self.numbytes, self.binpoint)


    def cluster(self, method='hdbscan', *args, **kwargs):
        groupNo = kwargs['groupNo'] if 'groupNo' in kwargs.keys() else None
        if groupNo is not None:
            self.clu[groupNo] = self.fet.toclu(method=method, *args, **kwargs)
        else:
            pass


    def construct_kdtree(self, groupNo):
        self.kd = []
        for clu_id, value in self.clu[groupNo].index.items():
            fet = self.fet[groupNo][value]
            self.kd.append(KDTree(fet))


    def predict(self, groupNo, X, method='knn', k=10):
        if X.ndim==1: X=X.reshape(1,-1)
        if method == 'knn':
            self.construct_kdtree(groupNo)
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

    def remove_spk(self, group, global_ids):
        '''
        Delete spks using global_ids, spks includes SPK, FET, CLU, SPKTAG. 
        '''
        info("received model modified event, removed spikes[group={}, global_ids={}]".format(group, global_ids))
       
        with Timer("[MODEL] Model -- remove spk from SPK.", verbose=conf.ENABLE_PROFILER):
            self.spk.remove(group, global_ids)
        with Timer("[MODEL] Model -- spk to FET.", verbose=conf.ENABLE_PROFILER):
            self.fet[group] = self.spk._tofet(group, method=self.fet_method)
        with Timer("[MODEL] Model -- fet to  CLU.", verbose=conf.ENABLE_PROFILER):
            self.clu[group] = CLU(self.fet._toclu(group, method='reset'))
        with Timer("[MODEL] Model --  remove spk from SPKTAG.", verbose=conf.ENABLE_PROFILER):
            self.spktag.remove(group, global_ids)

    def mask_spk(self, group, global_ids):
        '''
        Delete spks using global_ids, spks includes SPK, FET, CLU, SPKTAG. 
        '''
        info("received model modified event, mask spikes[group={}, global_ids={}]".format(group, global_ids))
        
        with Timer("mask spk from SPK.", verbose=conf.ENABLE_PROFILER):
            self.spk.mask(group, global_ids)
        with Timer("spk to FET.", verbose=conf.ENABLE_PROFILER):
            self.fet[group] = self.spk._tofet(group, method=self.fet_method)
        with Timer("reset clu", verbose=conf.ENABLE_PROFILER):
            self.clu[group] = CLU(self.fet._toclu(group, method='reset'))
        with Timer("remove spk from SPKTAG.", verbose=conf.ENABLE_PROFILER):
            self.spktag.mask(group, global_ids)

