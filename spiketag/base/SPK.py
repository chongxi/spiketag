import numpy as np
import numexpr as ne
from .FET import FET


class SPK():
    def __init__(self, spkdict):
        self.spk = spkdict
        self.nCh = len(spkdict)
        self.ch_span = self.spk[0].shape[-1]
        self.spklen = 19
        weight_vector = np.array([0.2871761 , 0.2871761 , 0.3571761 , 0.45907732, 0.45485107, 
                                  0.664169  , 0.85485229, 0.91183021, 0.83639082, 0.83206653, 
                                  0.79556892, 0.55092225, 0.57119953, 0.67515538, 0.68811997,  
                                  0.62243462, 0.34097719, 0.38911416, 0.33874702], dtype=np.float32)
        weight_channel = self.weight_channel_saw(np.arange(self.ch_span))
        W = weight_channel * weight_vector.reshape(-1,1)
        self.W = W.T.ravel()

    @property
    def nspk(self):
        nspk = 0
        for i in range(self.nCh):
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
    
    def tofet(self, method='weighted-pca', ncomp=6, whiten=False):
        fet = {}
        pca_comp = {}
        shift = {}
        scale = {}

        if isinstance(method, int):
            for i in range(len(self.spk)):
                spk = self.spk[i]
                if spk.shape[0] > 0:
                    fet[i] = spk[:,method,:]
                else:
                    fet[i] = np.array([])

        elif method == 'peak':
            for i in range(len(self.spk)):
                spk = self.spk[i]
                if spk.shape[0] > 0:
                    # TODO: 9:13?
                    temp_fet = spk[:,4:7,:].min(axis=1).squeeze()  
                    temp_fet = temp_fet - np.mean(temp_fet, axis=0)
                    fet[i] = temp_fet/(temp_fet.max()-temp_fet.min())
                else:
                    fet[i] = np.array([])
            self.fet = fet

        elif method == 'pca':
            from sklearn.decomposition import PCA
            for i, spk in self.spk.items():
                # TODO: 6?
                pca = PCA(n_components=ncomp, whiten=whiten)
                if spk.shape[0] > 0:
                    # X = np.concatenate((spk[:,:,:].transpose(2,1,0)),axis=0).T   #
                    X = spk.transpose(0,2,1).ravel().reshape(-1, spk.shape[1]*spk.shape[2])
                    temp_fet = pca.fit_transform(X)
                    fet[i] = temp_fet/(temp_fet.max()-temp_fet.min()) 
                else:
                    fet[i] = np.array([])
            self.fet = fet

        elif method == 'weighted-pca':
            ne.set_num_threads(32)
            from sklearn.decomposition import PCA
            for i in range(len(self.spk)):
                # TODO: 6?
                pca = PCA(n_components=ncomp, whiten=whiten)
                spk = self.spk[i]
                # Because when len(datas) less than ncomp, the len(fet) will
                # less then ncomp too. So len(datas) smaller then ncomp is
                # meaningless.
                if spk.shape[0] >= ncomp:
                    # step 0
                    X = spk.transpose(0,2,1).ravel().reshape(-1, spk.shape[1]*spk.shape[2])
                    W = self.W
                    X = ne.evaluate('X*W')
                    # step 1
                    temp_fet = pca.fit(X)
                    # pca_comp[i] = pca.components_.T
                    pca_comp[i] = np.floor(pca.components_.T*(2**7))/(2**7)
                    temp_fet = np.dot(X, pca_comp[i])
                    # pca_comp[i] = pca.components_.T
                    # step 2
                    shift[i] = -np.dot(X.mean(axis=0), pca.components_.T)
                    temp_fet += shift[i]
                    # step 3
                    scale[i] = temp_fet.max()-temp_fet.min()
                    temp_fet /= scale[i]
                    # quantization for FPGA
                    fet[i] = temp_fet
                    # fet[i] = np.floor(temp_fet*2**8)/(2**8)
                else:
                    # keep same shape even no feature value, for future
                    # convinience.
                    fet[i] = np.empty((0, ncomp), dtype=np.float32)
            self.fet = fet
            self.pca_comp = pca_comp
            self.shift = shift
            self.scale = scale

        elif method == 'ica':
            from sklearn.decomposition import FastICA
            for i, spk in self.spk.items():
                # TODO: 6?
                ica = FastICA(n_components=3, whiten=True)  # ICA must be whitened
                if spk.shape[0] > 0:
                    # X = np.concatenate((spk[:,:,:].transpose(2,1,0)),axis=0).T   #
                    X = spk.transpose(0,2,1).ravel().reshape(-1, spk.shape[1]*spk.shape[2])
                    temp_fet = ica.fit_transform(X)
                    fet[i] = temp_fet/(temp_fet.max()-temp_fet.min()) 
                else:
                    fet[i] = np.array([])
            self.fet = fet

        elif method == 'weighted-ica':
            ne.set_num_threads(32)
            from sklearn.decomposition import FastICA
            for i in range(len(self.spk)):
                # TODO: 6?
                ica = FastICA(n_components=3, whiten=True)  # ICA must be whitened
                spk = self.spk[i]
                if spk.shape[0] > 0:
                    # X = np.concatenate((spk[:,:,:].transpose(2,1,0)),axis=0).T   #
                    X = spk.transpose(0,2,1).ravel().reshape(-1, spk.shape[1]*spk.shape[2])
                    W = self.W
                    X = ne.evaluate('X*W')
                    temp_fet = ica.fit_transform(X)
                    fet[i] = temp_fet/(temp_fet.max()-temp_fet.min()) 
                else:
                    fet[i] = np.array([])
            self.fet = fet

        else:
            print 'method = {peak, pca, weighted-pca, ica, weighted-ica}'
        return FET(fet)
