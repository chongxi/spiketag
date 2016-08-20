import numpy as np
from .FET import FET
from .Binload import bload
from numba import jit
import numexpr as ne



@jit(cache=True)
def _to_spk(data, pos, chlist, spklen=25, ch_span=1):
    n = len(pos)
    spk = np.empty((n, spklen, 2*ch_span+1), dtype=np.float32)
    for i in range(n):
        # i spike in chlist
        spk[i, ...]  = data[pos[i]-10:pos[i]+15, chlist]
    _nan = np.where(chlist==-1)[0]
    spk[..., _nan] = 0
    return spk


class MUA():
    def __init__(self, filename, nCh=32, fs=25000, numbytes=4, binary_radix=14):

        self.numbytes = numbytes
        self.dtype = 'i'+str(self.numbytes)
        bf = bload(nCh, fs)
        bf.load(filename, dtype=self.dtype)
        self.filename = filename
        self.nCh = nCh
        self.fs  = fs*1.0
        self.data = bf.asarray(binpoint=binary_radix)
        self.t    = bf.t

        self.npts = bf._npts
        self.pre = int(4e-4 * fs)
        self.post = int(4e-4 * fs)
        spk_meta = np.fromfile(filename+'.spk', dtype='<i4')
        self.pivotal_pos = spk_meta.reshape(-1,2).T

    def get_near_ch(self, ch, ch_span=1):
        chmax = self.nCh - 1
        start = ch-ch_span # if ch-span>=0 else 0
        end   = ch+ch_span # if ch+span<chmax else chmax
        near_ch = np.arange(start, end+1, 1)
        near_ch[near_ch>chmax] = -1
        near_ch[near_ch<0] = -1
        return near_ch

    def tospk(self, ch_span=1):
        self.ch_hash = np.asarray([self.get_near_ch(ch, ch_span) 
                                                for ch in range(self.nCh)])
        spkdict = {}
        for ch in range(self.nCh):
            pos = self.pivotal_pos[0, self.pivotal_pos[1]==ch]
            spkdict[ch] = _to_spk(data   =self.data, 
                                  pos    = pos, 
                                  chlist =self.ch_hash[ch], 
                                  spklen =25,
                                  ch_span=ch_span)
        return SPK(spkdict)



class SPK():
    def __init__(self, spkdict):
        self.spk = spkdict
        self.nCh = len(spkdict)
        self.ch_span = self.spk[0].shape[-1]
        weight_vector = np.array([ 0.02989419,  0.0426025 ,  0.07831115,  0.07639907,  0.0971761 ,
                                   0.10907732,  0.23485107,  0.414169  ,  0.55485229,  0.71183021,
                                   0.80639082,  0.83206653,  0.79556892,  0.65092225,  0.47119953,
                                   0.23515538,  0.08119973,  0.25243462,  0.44097719,  0.43911416,
                                   0.48874702,  0.48230024,  0.38475716,  0.37505245,  0.23355913 ],
                                   dtype=np.float32)
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
    
    def tofet(self, method='weighted-pca', ncomp=6):
        fet = {}
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
                    fet[i] = spk[:,9:13,:].min(axis=1).squeeze()  
                else:
                    fet[i] = np.array([])
            self.fet = fet
        elif method == 'pca':
            from sklearn.decomposition import PCA
            for i in range(len(self.spk)):
                # TODO: 6?
                pca = PCA(n_components=ncomp, whiten=True)
                spk = self.spk[i]
                if spk.shape[0] > 0:
                    X = np.concatenate((spk[:,:,:].transpose(2,1,0)),axis=0).T   #
                    temp_fet = pca.fit_transform(X)
                    fet[i] = temp_fet/(temp_fet.max()-temp_fet.min()) # scale down to (-1,1)
                else:
                    fet[i] = np.array([])
            self.fet = fet
        elif method == 'weighted-pca':
            ne.set_num_threads(32)
            from sklearn.decomposition import PCA
            for i in range(len(self.spk)):
                # TODO: 6?
                pca = PCA(n_components=ncomp, whiten=True)
                spk = self.spk[i]
                if spk.shape[0] > 0:
                    X = np.concatenate((spk[:,:,:].transpose(2,1,0)),axis=0).T   #
                    W = self.W
                    X = ne.evaluate('X*W')
                    temp_fet = pca.fit_transform(X)
                    fet[i] = temp_fet/(temp_fet.max()-temp_fet.min()) # scale down to (-1,1)
                else:
                    fet[i] = np.array([])
            self.fet = fet            

        else:
            print 'method = {peak, pca or some integer}'
        return FET(fet)
