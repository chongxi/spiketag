from .MUA import MUA
from .SPK import SPK
from .FET import FET
from .CLU import CLU
from .CLU import status_manager
import numpy as np
import json
import pickle
import pandas as pd
from numba import njit


@njit(cache=True)
def to_global_labels(grp_clu_matrix, cumsum_nclu):
    '''
    grp_clu_matrix: (N, 2) matrix, each row is a (grp_id, clu_id) pair
    cumsum_nclu: (40,) vector, cumsum of model.nclus or spktag.nclus
    '''
    grp_clu_matrix = grp_clu_matrix.astype(np.int32)
    N = grp_clu_matrix.shape[0]
    labels = np.zeros((N,))
    for i in range(N):
        grp_id, clu_id = grp_clu_matrix[i]
        if grp_id > 0:  # for each grp_id there is a base value
            base = cumsum_nclu[grp_id - 1]
        else:
            base = 0
        if clu_id > 0:  # for each label, we need to add the base
            labels[i] = base + clu_id
    return labels



class SPKTAG(object):
    def __init__(self, probe=None, spk=None, fet=None, clu=None, clu_manager=None, gtimes=None, filename=None):
        '''
        spk         : spk object
        fet         : fet object
        clu         : dictionary of clu object (each item is a channel based clu object)
        clu_manager : clu manager
        gtimes      : dictionary of group with spike times
        '''
        self.probe = probe
        if filename is not None: # load from file
            self.fromfile(filename)
        elif gtimes is not None : # construct
            self.gtimes  = gtimes
            self.spk     = spk 
            self.fet     = fet 
            self.clu     = clu 
            self.spklen  = spk.spklen
            self.fetlen  = fet.fetlen
            self.grplen  = self.probe.group_len
            self.ngrp    = len(self.probe.grp_dict.keys())
            self.clu_manager = clu_manager

            self.dtype   = [('t', 'int32'),
                            ('group','int32'),  
                            ('spk', 'f4', (self.spklen, self.grplen)), 
                            ('fet','f4',(self.fetlen,)),
                            ('clu','int32')]
        else:
            pass

    @property
    def nspk(self):
        return sum([len(v) for v in self.gtimes.values()])

    def build_meta(self):
        meta = {}
        meta["fs"] = self.probe.fs
        meta["ngrp"] = self.ngrp
        meta["grplen"] = self.probe.group_len
        meta["fetlen"] = self.fetlen
        meta["spklen"] = self.spklen
        meta["clu_statelist"] = self.clu_manager.state_list
        return meta

    def build_hdbscan_tree(self):
        treeinfo = {}
        for i in range(self.ngrp):
            try:
                treeinfo[i] = self.clu[i]._extra_info
            except:
                treeinfo[i] = None
        return treeinfo

    def build_spktag(self):
        spktag = np.zeros(self.nspk, dtype=self.dtype)
        start_index = 0
        for g, times in self.gtimes.items():
            if times.shape[0] > 0:
                end_index = start_index + len(times)
                spktag['t'][start_index:end_index] = times
                spktag['group'][start_index:end_index] = np.full((len(times)), g, dtype=np.int)
                spktag['spk'][start_index:end_index] = self.spk[g]
                spktag['fet'][start_index:end_index] = self.fet[g]        
                spktag['clu'][start_index:end_index] = self.clu[g].membership
                start_index = end_index
        return spktag


    def build_spkid_matrix(self, including_noise=False):
        spkid_matrix = np.hstack((self.spktag['t'].reshape(-1,1), 
                                  self.spktag['group'].reshape(-1,1), 
                                  self.spktag['fet'], 
                                  self.spktag['clu'].reshape(-1,1)))
        if including_noise is False:
            spkid_matrix = spkid_matrix[spkid_matrix[:,-1]!=0]
        grp_clu_matrix = spkid_matrix[:, [1,-1]]
        global_labels = to_global_labels(grp_clu_matrix, self.nclus.cumsum())
        spkid_matrix[:, -1] = global_labels
        spkid_matrix = pd.DataFrame(spkid_matrix).sort_values(0, ascending=True)
        spkid_matrix.columns = ['frame_id','group_id','fet0','fet1','fet2','fet3','spike_id']
        spkid_matrix.index = np.arange(global_labels.shape[0])
        return spkid_matrix


    def update(self, spk, fet, clu, gtimes):
        self.spk = spk
        self.fet = fet
        self.clu = clu
        self.gtimes = gtimes
        self.build_meta()
        self.build_spktag()	
        self.build_spkid_matrix()


    def tofile(self, filename, including_noise=False):
        self.meta = self.build_meta()
        self.treeinfo = self.build_hdbscan_tree()
        self.spktag = self.build_spktag()
        self.spkid_matrix = self.build_spkid_matrix(including_noise=including_noise)
        with open(filename+'.meta', 'w') as metafile:
                json.dump(self.meta, metafile, indent=4)
        np.save(filename+'.npy', self.treeinfo)
        self.spktag.tofile(filename)   # numpy to file
        self.spkid_matrix.to_pickle(filename+'.pd')  # pandas data frame


    def fromfile(self, filename):
        # meta file
        with open(filename+'.meta', 'r') as metafile:
            self.meta = json.load(metafile)
        self.fs     = self.meta['fs']
        self.ngrp   = self.meta['ngrp']
        self.grplen = self.meta['grplen']
        self.spklen = self.meta['spklen']
        self.fetlen = self.meta['fetlen']
        self.clu_statelist = self.meta['clu_statelist']

        # condensed tree info
        self.treeinfo = np.load(filename+'.npy', allow_pickle=True).item()

        # spiketag
        self.dtype = [('t', 'int32'), 
                      ('group', 'int32'),  
                      ('spk', 'f4', (self.spklen, self.grplen)), 
                      ('fet', 'f4', (self.fetlen,)),
                      ('clu', 'int32')]
        self.spktag = np.fromfile(filename, dtype=self.dtype)
        try:
            self.spkid_matrix = pd.read_pickle(filename+'.pd')
        except:
            pass


    def tospk(self):
        spkdict = {}
        for g in self.gtimes.keys():
            spkdict[g] = self.spktag['spk'][self.spktag['group']==g]
        self.spk = SPK(spkdict)
        return self.spk		


    def tofet(self):
        fetdict = {}
        for g in self.gtimes.keys():
            fetdict[g] = self.spktag['fet'][self.spktag['group']==g]
        self.fet = FET(fetdict)
        return self.fet		


    def toclu(self):
        cludict = {}
        for g in self.gtimes.keys():
            cludict[g] = CLU(self.spktag['clu'][self.spktag['group']==g], treeinfo=self.treeinfo[g])
            cludict[g]._id    = g
            cludict[g]._state = cludict[g].s[self.clu_statelist[g]]
        self.clu = cludict
        return self.clu


    def to_gtimes(self):
        times = self.spktag['t']
        groups = self.spktag['group']
        gtimes = {}
        for g in np.unique(groups):
            gtimes[g] = times[np.where(groups == g)[0]]
        self.gtimes = gtimes
        return self.gtimes


    @property
    def done_groups(self):
        return np.where(np.array(self.clu_manager.state_list) == 3)[0]

    @property
    def nclus(self):
        self._nclus = []
        for i in range(self.ngrp):
            n = self.clu[i].nclu
            self._nclus.append(n)
        self._nclus = np.array(self._nclus) - 1
        return self._nclus

    def _get_label(self, grp_id, clu_id):
        assert(clu_id<=self.nclus[grp_id]), "group {} contains only {} clusters".format(grp_id, self.nclus[grp_id])
        if clu_id == 0:
            return 0
        else:
            clu_offset = self.nclus.cumsum()[grp_id-1]
            return clu_offset+clu_id

    def get_spk_times(self, group_id, cluster_id):
        '''
        get spike times from a specific group with a specific cluster number
        '''
        idx = self.clu[group_id][cluster_id]
        spk_times = self.gtimes[group_id][idx]/self.fs
        return spk_times

    def get_spk_time_dict(self):
        '''
        callable after clu_manager is initiated
        '''
        k = 0
        spk_time_dict = {}
        for grp_No, grp_state in enumerate(self.clu_manager.state_list):
            if grp_state == 3: # done state
                for clu_No in range(1, self.clu[grp_No].nclu):
                    spk_time_dict[k] = self.get_spk_times(grp_No, clu_No)
                    k+=1
        return spk_time_dict

    def load(self, filename):
        self.fromfile(filename)
        self.gtimes = self.to_gtimes()
        self.spk = self.tospk()
        self.fet = self.tofet()
        self.clu = self.toclu()
        self.clu_manager = status_manager()
        for _clu in self.clu.values():
            self.clu_manager.append(_clu)
        self.spk_time_dict = self.get_spk_time_dict()
        self.spk_time_array = np.array(list(self.spk_time_dict.values()))
        self.n_units = len(self.spk_time_dict)
        print('loading from spktag {}: {} neurons extracted'.format(filename, self.n_units))
