import numpy as np
import numexpr as ne
import pandas as pd
# from spiketag.view import spike_view
from .FET import FET
from ..utils.conf import info


def _transform(X, P, shift, scale):
    '''
    y = scale*((PX)+shift) 
    check the range of X, it has to be float32, if not, use X /= float(2**13)
    '''
    y = (np.dot(X,P) + shift)/scale 
    return y


def _construct_transformer(x, ncomp=6):
    from sklearn.decomposition import PCA
    pca = PCA(n_components=ncomp, whiten=False)
    # step 1
    temp_fet = pca.fit(x)
    # pca_comp[i] = pca.components_.T
    # 8 bit PCA: #1.#7
    pca_comp = np.floor(pca.components_.T*(2**7))/(2**7)
    temp_fet = np.dot(x, pca_comp)
    # pca_comp[i] = pca.components_.T
    # step 2
    shift = -np.dot(x.mean(axis=0), pca.components_.T)
    temp_fet += shift
    # step 3
    scale = temp_fet.max()-temp_fet.min()
    temp_fet /= scale
    # quantization for FPGA
    fet = temp_fet
    return pca_comp, shift, scale


def _to_fet(_spk_array, _weight_vector, method='weighted-pca', ncomp=6, whiten=False):

    X = _spk_array.transpose(0,2,1).ravel().reshape(-1, _spk_array.shape[1]*_spk_array.shape[2])
    W = _weight_vector

    if isinstance(method, int):
        fet = _spk_array[:, method, :]

    elif method == 'peak':
        # TODO: 9:13?
        temp_fet = _spk_array[:,4:7,:].min(axis=1).squeeze()  
        temp_fet = temp_fet - np.mean(temp_fet, axis=0)
        fet = temp_fet/(temp_fet.max()-temp_fet.min())

    elif method == 'tsne':
        from sklearn.manifold import TSNE
        tsne = TSNE(n_components=ncomp, random_state=0)
        if _spk_array.shape[0] >= ncomp:
            temp_fet = tsne.fit_transform(X)
            fet = temp_fet/(temp_fet.max() - temp_fet.min())
        else:
            fet = np.empty((0, ncomp), dtype=np.float32)

    elif method == 'pca':
        from sklearn.decomposition import PCA
        pca = PCA(n_components=ncomp, whiten=whiten)
        if _spk_array.shape[0] >= ncomp:
            temp_fet = pca.fit_transform(X)
            fet = temp_fet/(temp_fet.max()-temp_fet.min()) 
        else:
            fet = np.empty((0, ncomp), dtype=np.float32)

    elif method == 'weighted-pca':
        ne.set_num_threads(32)
        from sklearn.decomposition import PCA
        pca = PCA(n_components=ncomp, whiten=whiten)
        if _spk_array.shape[0] >= ncomp:
            # step 0
            X = ne.evaluate('X*W')
            # step 1
            temp_fet = pca.fit(X)
            # pca_comp[i] = pca.components_.T
            pca_comp = np.floor(pca.components_.T*(2**7))/(2**7)
            temp_fet = np.dot(X, pca_comp)
            # pca_comp[i] = pca.components_.T
            # step 2
            shift = -np.dot(X.mean(axis=0), pca.components_.T)
            temp_fet += shift
            # step 3
            scale = temp_fet.max()-temp_fet.min()
            temp_fet /= scale
            # quantization for FPGA
            fet = temp_fet
            # fet[i] = np.floor(temp_fet*2**8)/(2**8)
        else:
            # keep same shape even no feature value, for future
            # convinience.
            fet = np.empty((0, ncomp), dtype=np.float32)

    elif method == 'ica':
        from sklearn.decomposition import FastICA
        ica = FastICA(n_components=3, whiten=True)  # ICA must be whitened
        temp_fet = ica.fit_transform(X)
        fet = temp_fet/(temp_fet.max()-temp_fet.min()) 

    elif method == 'weighted-ica':
        ne.set_num_threads(32)
        from sklearn.decomposition import FastICA
        ica = FastICA(n_components=3, whiten=True)  # ICA must be whitened
        X = ne.evaluate('X*W')
        temp_fet = ica.fit_transform(X)
        fet = temp_fet/(temp_fet.max()-temp_fet.min()) 

    else:
        print('method has to be {peak, pca, weighted-pca, ica, weighted-ica}')

    return fet


class SPK():
    def __init__(self, spkdict=None):
        '''
        Example:
            from spiketag.base import SPK, FET, CLU
            spk = SPK()
            spk.load_spkwav('./spk_wav.bin')   # load spk_wav.bin
            spk_df = spk.sort(method='dpgmm')  # sort spk_wav.bin by dpgmm

        Variables:
            spk.ch, 
            spk.spk_time, 
            spk.electrode_group, 
            spk.spk_dict, 
            spk_spk_time_dict
        '''
        if spkdict is not None:
            self.spkdict = spkdict
            self(self.spkdict)
    
    def __call__(self, spkdict):
        self.__spk = spkdict.copy() 
        self.spk = spkdict
        self.n_group = len(spkdict)
        self.ch_span = list(self.spk.values())[0].shape[-1]
        self.spklen = 19
        weight_vector = np.array([0.2871761 , 0.2871761 , 0.3571761 , 0.45907732, 0.45485107, 
                                  0.664169  , 0.85485229, 0.91183021, 0.83639082, 0.83206653, 
                                  0.79556892, 0.55092225, 0.57119953, 0.67515538, 0.68811997,  
                                  0.62243462, 0.34097719, 0.38911416, 0.33874702], dtype=np.float32)
        weight_channel = self.weight_channel_saw(np.arange(self.ch_span))
        W = weight_channel * weight_vector.reshape(-1,1)
        self.W = W.T.ravel()
        self.W = np.ones((self.spklen*self.ch_span,)).astype(np.float32) # for tetrode

    @property
    def nspk(self):
        nspk = 0
        for i in range(self.n_group):
            nspk += self.spk[i].shape[0]
        return nspk

    def weight_channel_saw(self, chlist, a=None, p=None):
        n = len(chlist)
        if a is None: # a is max value of saw
            a = float(n)/2 
        if p is None:
            p = n/2   # p is the half period of entire saw
        return (a/p) * (p - abs(chlist % (2*p) - p) ) + 1

    def __getitem__(self,i):
        return self.spk[i]

    def __setitem__(self, i,  _spk_array):
        self.spk[i] = _spk_array

    def mask(self, group, ids):
        self.spk[group] = np.delete(self.__spk[group], ids, axis=0)
       
    def remove(self, group, ids):
        self.spk[group] = np.delete(self.spk[group], ids, axis=0)

    def _tofet(self, group, method='pca', ncomp=6, whiten=False):
        spk = self.spk[group]
        if spk.shape[0] > ncomp:
            fet = _to_fet(spk, self.W, method, ncomp, whiten)
        else:
            fet = np.zeros((spk.shape[0], ncomp), dtype=np.float32)
        return fet

    def load_spkwav(self, file='./spk_wav.bin'):
        '''
        spk = SPK()
        spk.load_spkwav('./spk_wav.bin')        
        '''
        spk = np.fromfile(file, dtype=np.int32).reshape(-1, 20, 4)
        self.ch, self.spk_time, self.electrode_group = spk[..., 0, 1], spk[..., 0, 2], spk[..., 0, 3]
        group_list = np.sort(np.unique(self.electrode_group))
        self.spk_dict = {}
        self.spk_time_dict = {}
        for group in group_list:
            self.spk_dict[group] = spk[self.electrode_group == group][:, 1:, :]/(2**14) # grouped spike waveforms
            self.spk_time_dict[group] = spk[self.electrode_group == group][:, 0, 2]
        self(self.spk_dict)

    def tofet(self, group_id=None, method='pca', ncomp=4, whiten=False):
        fet = {}
        # pca_comp = {}
        # shift = {}
        # scale = {}
        if group_id is not None:
            return self._tofet(group_id, method, ncomp, whiten)
        else:
            for group in self.spk.keys():
                fet[group] = self._tofet(group, method, ncomp, whiten)
            #     info('group[{}]:{} spikes'.format(group, fet[group].shape[0]))
            #     info('spk._tofet(group_id={}, method={}, ncomp={}, whiten={})'.format(group, method, ncomp, whiten))
            # info('----------------success------------------')
            # info(' ')
            self.fet = FET(fet)
            return self.fet
    
    def sort(self, minimum_spks=50, cluster_method='dpgmm', file=None):
        self.tofet(method='pca', ncomp=4, whiten=False);
        ### todo: 1. add blocking mode    2. kmeans for clusterless sort
        self.fet.to_clu(method=cluster_method, mode='blocking')
        ### todo: 3. merge all low spike number cluster ( < minimum_spks(e.g., 50) spikes ) to 0 cluster (noise)
        self.fet.assign_clu_global_labels()
        self.to_spikedf(file)

    def to_spikedf(self, file=None):
        '''
        spike_df is a dataframe that each row is a spike packet (frame_id, group_id, fet0, fet1, fet2, fet3, spike_id)
        sorted by timestamps
        '''
        spk_matrix = np.array([]).reshape(-1, 7)
        for g in self.fet.group:
            h = np.hstack((self.spk_time_dict[g].reshape(-1,1),                         # spike frame_id (time stamps in #samples)
                           self.electrode_group[self.electrode_group==g].reshape(-1,1), # spike group_id (electrode group)
                           self.fet[g][:,:4],                                           # spike features (multichannel waveform 4d feature)
                           self.fet.clu[g].membership_global.reshape(-1,1)))            # spike spike_id (assigned unit id)
            spk_matrix = np.append(spk_matrix, h, axis=0)

        self.spike_df = pd.DataFrame(spk_matrix)
        self.spike_df.columns = ['frame_id', 'group_id', 'fet0', 'fet1', 'fet2', 'fet3', 'spike_id']
        self.spike_df = self.spike_df.sort_values('frame_id')

        if file is not None:
            self.spike_df.to_pickle(file, index=False)

        return self.spike_df

    # def show(self, group_id):
    #     self.spk_view = spike_view()
    #     self.spk_view.set_data(self.spk[group_id])
    #     self.spk_view.show()
