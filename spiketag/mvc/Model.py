import numpy as np
from sklearn.neighbors import KDTree
from ..base.SPK import _construct_transformer
from ..base import *
from ..utils.conf import info 
from ..utils import conf
from ..utils.utils import Timer
from ..analysis.place_field import place_field


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

    def __init__(self, mua_filename='./mua.bin', spk_filename='./spk.bin', probe=None, spktag_filename=None, 
                 numbytes=4, binary_radix=13, scale=False, spklen=19, corr_cutoff=0.9, amp_cutoff=[-15000, 1000],
                 fet_method='pca', fetlen=4, fet_whiten=False,
                 clu_method='hdbscan', fall_off_size=18, n_jobs=24,
                 time_segs=None,
                 playground_log=None, session_id=0, 
                 pc=None, bin_size=4, v_cutoff=5, replay_offset=0.,
                 sort_movment_only=False):

        # raw recording param
        self.mua_filename = mua_filename
        self.spk_filename = spk_filename
        self.spktag_filename = spktag_filename
        self.probe = probe
        self.numbytes = numbytes
        self.binpoint = binary_radix
        self.scale = scale

        # mua param
        self._corr_cutoff = corr_cutoff
        self._spklen = spklen 
        self._amp_cutoff = amp_cutoff
        self._time_segs = time_segs

        # fet param
        self.fet_method = fet_method
        self._fet_whiten = fet_whiten
        self._fetlen = fetlen

        # clu param
        self.clu_method = clu_method
        self._fall_off_size = fall_off_size
        self._n_jobs = n_jobs

        # playground log
        self.time_still = None
        # TODO1: fix this
        if playground_log is not None:
            self.pc = place_field(logfile=playground_log, session_id=session_id, bin_size=bin_size, v_cutoff=v_cutoff)
            start, end = self._time_segs
            self.pc.align_with_recording(start, end, replay_offset)
            self.pc.initialize()
            if sort_movment_only:
                self.time_still = self.ts[self.low_speed_idx] 

        # Load pc 
        elif pc is not None:
            self.pc = pc
            start, end = self._time_segs
            self.pc.align_with_recording(start, end, replay_offset)
            self.pc.initialize()
            if sort_movment_only:
                self.time_still = self.pc.ts[self.pc.low_speed_idx]

        else:
            self.pc = None

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
            self.mua = MUA(probe        = self.probe,
                           mua_filename = self.mua_filename, 
                           spk_filename = self.spk_filename, 
                           numbytes     = self.numbytes, 
                           binary_radix = self.binpoint,
                           scale        = self.scale,
                           cutoff       = self._amp_cutoff,         # for amp   cut_off
                           time_segs    = self._time_segs,      # for time  cut_off
                           time_still   = self.time_still,      # for speed cut_off
                           lfp          = False)

            self.get_spk(amp_cutoff=True, speed_cutoff=True, time_cutoff=True)
            self.get_fet()
            self.get_clu()
            self.init_clu_manager()

        # After first time
        else:
            self.spktag = SPKTAG(probe=self.probe)
            self.spktag.load(spktag_filename)
            self.gtimes = self.spktag.gtimes
            self.spk = self.spktag.spk
            self.fet = self.spktag.fet
            self.clu = self.spktag.clu
            self.clu_manager = self.spktag.clu_manager
            self.spk_time_dict = self.spktag.spk_time_dict
            self.spk_time_array = self.spktag.spk_time_array
            self.spkid_matrix = self.spktag.spkid_matrix
            # info('load spktag file')
            # self.spktag.fromfile(spktag_filename)
            # self.gtimes = self.spktag.to_gtimes()
            # self.spk = self.spktag.tospk()
            # self.fet = self.spktag.tofet()
            # self.clu = self.spktag.toclu()
            # self.clu_manager = status_manager()
            # for _clu in self.clu.values():
            #     self.clu_manager.append(_clu)
            # self.spktag.clu_manager = self.clu_manager

            info('load mua data for wave view')
            self.mua = MUA(probe        = self.probe,
                           mua_filename = self.mua_filename, 
                           spk_filename = self.spk_filename, 
                           numbytes     = self.numbytes, 
                           binary_radix = self.binpoint,
                           scale        = self.scale,
                           cutoff       = self._amp_cutoff, 
                           time_segs    = self._time_segs, 
                           time_still   = self.time_still,
                           lfp          = False)
            self.mua.spk_times = self.gtimes
            self.spk_times = self.mua.spk_times
            info('Model.spktag is generated, nspk:{}'.format(self.spktag.nspk))

        self.groups = self.probe.grp_dict.keys()
        self.ngrp   = len(self.groups)


    def get_spk(self, amp_cutoff=True, speed_cutoff=True, time_cutoff=True):
        info('extract spikes from pivital meta data')
        # self.spk = self.mua.tospk(amp_cutoff=amp_cutoff,
        #                           speed_cutoff=speed_cutoff,
        #                           time_cutoff=time_cutoff)
        self.spk = SPK()
        self.spk.load_spkwav('./spk_wav.bin')
        self.mua.spkdict = {} #self.spk.spk_dict
        self.mua.spk_times = {} #self.spk.spk_time_dict
        for g in self.probe.grp_dict.keys():
            if g in self.spk.spk_dict.keys():
                if time_cutoff and self._time_segs is not None:
                    time_in_seg = (self.spk.spk_time_dict[g] > self._time_segs[0] * self.probe.fs) & (self.spk.spk_time_dict[g] <= self._time_segs[1] * self.probe.fs)
                    self.spk.spk_dict[g] = self.spk.spk_dict[g][time_in_seg]
                    ## spk.spk_time_dict and spk.electrode_group are used in spk.to_spikedf()
                    self.spk.spk_time_dict[g] = self.spk.spk_time_dict[g][time_in_seg]
                    self.spk.spk_group_dict[g] = self.spk.spk_group_dict[g][time_in_seg]
                if speed_cutoff:
                    pass # ! TODO
                if amp_cutoff:
                    pass # ! TODO
                self.mua.spkdict[g] = self.spk.spk_dict[g]
                self.mua.spk_times[g] = self.spk.spk_time_dict[g]
            else:
                self.mua.spkdict[g] = np.empty((0,0))
                self.mua.spk_times[g] = np.empty((0,0))

            if self.mua.spk_times[g].shape[0] == 0: 
                self.mua.spkdict[g] = np.random.randn(1, self.mua.spklen, len(self.probe[g]))
                self.mua.spk_times[g] = np.array([0])
                self.spk.spk_dict[g] = self.mua.spkdict[g]
                self.spk.spk_time_dict[g] = np.array([0])
                self.spk.spk_group_dict[g] = np.array([-1])
        
        self.gtimes = self.mua.spk_times

    def get_fet(self):
        info('extract features with {}'.format(self.fet_method))
        self.fet = self.spk.tofet(method=self.fet_method, 
                                  whiten=self._fet_whiten,
                                  ncomp=self._fetlen)
        # all clu are zeroes when fets are initialized

    def get_clu(self):
        self.clu = {}
        for g in self.probe.grp_dict.keys():
            if g in self.spk.spk_dict.keys():
                self.clu[g] = self.fet.clu[g]
            elif g not in self.spk.spk_dict.keys():
                _dummy_clu = CLU(np.array([0]))
                _dummy_clu._id = g
                self.clu[g] = _dummy_clu

    def init_clu_manager(self):
        '''
        assume run get_clu() first
        '''
        self.clu_manager = status_manager()
        for g in self.clu.keys():
            self.clu_manager.append(self.clu[g])

    def sort(self, clu_method, group_id='all', **kwargs):
        # info('removing high corr noise from spikes pool')
        # self.mua.remove_high_corr_noise(corr_cutoff=self._corr_cutoff)

        # info('removing all spks on group which len(spks) less then fetlen')
        # self.mua.remove_groups_under_fetlen(self._fetlen)

        info('clustering with {}'.format(clu_method))
        self.fet.toclu(method=clu_method, group_id=group_id, **kwargs)

        self.spktag = SPKTAG(self.probe,
                             self.spk, 
                             self.fet, 
                             self.clu,
                             self.clu_manager,
                             self.gtimes)
        info('Model.spktag is generated, nspk:{}'.format(self.spktag.nspk))

    
    def update_spktag(self):
        self.spktag = SPKTAG(self.probe,
                             self.spk, 
                             self.fet, 
                             self.clu,
                             self.clu_manager,
                             self.gtimes)
        info('Model.spktag is generated, nspk:{}'.format(self.spktag.nspk))


    def cluster(self, group_id, method, **params):
        self.sort(clu_method=method, group_id=group_id, **params)


    def construct_transformer(self, group_id, ndim=4):
        '''
        construct transformer parameters for a specific group
        y = a(xP+b)
        P: _pca_comp
        b: _shift
        a: _scale
        '''
        # concateated spike waveforms from one channel group in such an order: [spkch0, spkch1, ...]
        r = self.spk[group_id]
        x = r.transpose(0,2,1).ravel().reshape(-1, r.shape[1]*r.shape[2])  # (nspk, 76) important to transpose first to concateate waveforms without interleaving
        # construct transfomer params
        _pca_comp, _shift, _scale = _construct_transformer(x, ncomp=ndim)
        y = _scale * (x @ _pca_comp + _shift)
        return _pca_comp, _shift, _scale, y


    def construct_kdtree(self, group_id, global_ids=None, n_dim=4):
        self.kd = {} 
        for clu_id, value in self.clu[group_id].index.items():
            diff_ids = np.setdiff1d(value, global_ids, assume_unique=True)
            if len(diff_ids) > 0:
                fet = self.fet[group_id][diff_ids][:, :n_dim]
                self.kd[KDTree(fet)] = clu_id


    def predict(self, group_id, global_ids, method='knn', k=10, n_dim=4):
        X = self.fet[group_id][global_ids][:,:n_dim]
        if X.ndim==1: X=X.reshape(1,-1)

        if method == 'knn':
            self.construct_kdtree(group_id, global_ids)
            d = []
            for _kd, _ in self.kd.items():
                tmp = _kd.query(X, k)[0]
                d.append(tmp.mean(axis=1))
            d = np.vstack(np.asarray(d))
            labels = np.asarray(list(self.kd.values()))[np.argmin(d, axis=0)]
        return labels


    def tofile(self, filename=None, including_noise=False):
        '''
        This should automatically update the spktag array and save
        So that next time it can be loaded and avoid re-clustering
        '''
        if filename is not None:
            self.spktag.tofile(filename, including_noise=including_noise)
        elif self.spktag_filename is not None:
            self.spktag.tofile(self.spktag_filename, including_noise=including_noise)
        else:
            barename = self.filename.split('.')[0]
            self.spktag_filename = barename + '_spktag.bin'
            self.spktag.tofile(self.spktag_filename, including_noise=including_noise)

    def refine(self, group, global_ids):
        info("received model modified event, refine spikes[group={}, global_ids={}]".format(group, global_ids))
        labels = self.predict(group, global_ids) 
        info("the result of refine: {}".format(labels))
        self.clu[group].refill(global_ids, labels)
    
    def remove_spk(self, group, global_ids):
        '''
        Delete spks using global_ids, spks includes SPK, FET, CLU, SPKTAG. 
        '''
        info("received model modified event, removed spikes[group={}, global_ids={}]".format(group, global_ids))
        with Timer("[MODEL] Model -- remove spk from times", verbose=conf.ENABLE_PROFILER):
            self.gtimes[group] = np.delete(self.gtimes[group], global_ids)
        with Timer("[MODEL] Model -- remove spk from SPK.", verbose=conf.ENABLE_PROFILER):
            self.spk.remove(group, global_ids)
        with Timer("[MODEL] Model -- SPK to FET.", verbose=conf.ENABLE_PROFILER):
            self.fet[group] = self.spk._tofet(group, method=self.fet_method)
        with Timer("[MODEL] Model -- FET to CLU.", verbose=conf.ENABLE_PROFILER):
            self.clu[group] = self.fet._toclu(group)
            
    def mask_spk(self, group, global_ids):
        '''
        Mask spks using global_ids, spks includes SPK, FET, CLU, SPKTAG. 
        '''
        info("received model modified event, mask spikes[group={}, global_ids={}]".format(group, global_ids))
        
        with Timer("[MODEL] Model -- mask spk from SPK.", verbose=conf.ENABLE_PROFILER):
            self.spk.mask(group, global_ids)
        with Timer("[MODEL] Model -- SPK to FET.", verbose=conf.ENABLE_PROFILER):
            self.fet[group] = self.spk._tofet(group, method=self.fet_method)
        with Timer("[MODEL] Model -- FET to CLU", verbose=conf.ENABLE_PROFILER):
            self.clu[group] = self.fet._toclu(group)

    @property
    def nspk_per_grp(self):
        self._nspk = []
        for grp_id in range(self.ngrp):
            self._nspk.append(self.gtimes[grp_id].shape[0])    
        return np.array(self._nspk)

    @property
    def nspk_per_clu(self):
        self._nspk_per_clu = []
        for grp_id in range(self.ngrp):
            self._nspk_per_clu.append(np.array(list(self.clu[grp_id].index_count.values())))    
        return self._nspk_per_clu    
    
    @property
    def nclus(self):
        self._nclus = []
        for i in range(self.ngrp):
            n = self.clu[i].nclu
            self._nclus.append(n)
        self._nclus = np.array(self._nclus) - 1  # delete the noisy clu 0
        return self._nclus
