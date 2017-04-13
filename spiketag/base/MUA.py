import os
import numpy as np
from numba import jit
from multiprocessing import Pool
from .SPK import SPK
from .Binload import bload

@jit(cache=True)
def _to_spk(data, pos, chlist, spklen=19, prelen=8):
    n = len(pos)
    spk = np.empty((n, spklen, len(chlist)), dtype=np.float32)
    for i in range(n):
        # i spike in chlist
        spk[i, ...]  = data[pos[i]-prelen:pos[i]-prelen+spklen, chlist]
    _nan = np.where(chlist==-1)[0]
    spk[..., _nan] = 0
    return spk

@jit(cache=True, nopython=True)
def peakdet(v, delta, x = None):
    
    # initiate with .1 because numba doesn't support empty list
    maxtab = [(.1,.1)]
    mintab = [(.1,.1)]
    if x is None:
        x = np.arange(len(v))

    mn, mx = np.Inf, -np.Inf
    mnpos, mxpos = np.NaN, np.NaN
    lookformax = True
    for i in np.arange(len(v)):
        this = v[i]
        if this > mx:
            mx = this
            mxpos = x[i]
        if this < mn:
            mn = this
            mnpos = x[i]
        if lookformax:
            if this < mx-delta:
                maxtab.append((mxpos, mx))
                mn = this
                mnpos = x[i]
                lookformax = False
        else:
            if this > mn+delta:
                mintab.append((mnpos, mn))
                mx = this
                mxpos = x[i]
                lookformax = True
    return np.array(maxtab[1:]), np.array(mintab[1:])	

#  @jit(nopython=True)
#  def detect_spk(data, threshholds):
    
    #  t = np.array([1], dtype=np.int64)
    #  ch = np.array([1],dtype=np.int32)
  
    #  for i in range(data.shape[1]):
        #  _, mintab = peakdet(data[:,i],.3)    
        #  spks = mintab[mintab[:,1] < threshholds[i]].astype(np.int64)
        #  t = np.hstack((t,spks[:,0]))
        #  ch = np.hstack((ch,np.full(spks[:,0].shape,i,dtype=np.int32)))
        #  print 'deteck_spk channel {} finished at {}'.format(i, time.ctime())
    #  print 'detect_spk ends at {}'.format(time.ctime())

    #  return np.array([t[1:], ch[1:]])

class MUA():
    def __init__(self, filename, probe, numbytes=4, binary_radix=14):
        
        self.nCh = probe.n_ch
        self.ch  = range(self.nCh)
        self.fs  = probe.fs*1.0
        self.probe = probe
        self.numbytes = numbytes
        self.dtype = 'i'+str(self.numbytes)
        self.bf = bload(self.nCh, self.fs)
        self.bf.load(filename, dtype=self.dtype)
        self.filename = filename
        self.data = self.bf.asarray(binpoint=binary_radix)
        self.t    = self.bf.t

        self.npts = self.bf._npts
        self.spklen = 19
        self.prelen = 8
        spk_meta = np.fromfile(filename+'.spk', dtype='<i4')
        self.pivotal_pos = spk_meta.reshape(-1,2).T

    def tospk(self):
        self.ch_hash = np.asarray([self.probe.get_group_ch(ch) 
                                                for ch in range(self.nCh)])
        spkdict = {}
        for ch in range(self.nCh):
            pos = self.pivotal_pos[0, self.pivotal_pos[1]==ch]
            spkdict[ch] = _to_spk(data   = self.data, 
                                  pos    = pos, 
                                  chlist = self.ch_hash[ch], 
                                  spklen = self.spklen,
                                  prelen = self.prelen)
                                 
        return SPK(spkdict)

    def get_nid(self, corr_cutoff=0.95):  # get noisy spk id
        # 1. dump spikes file (binary)
        piv = self.pivotal_pos.T
        nspk = self.pivotal_pos.shape[1]
        rows = np.arange(-10,15).reshape(1,-1) + piv[:,0].reshape(-1,1)
        cols = piv[:,1].reshape(-1,1)
        full_spk = self.data[rows, :]
        filename = os.path.dirname(self.filename)+'/.'+os.path.basename(self.filename)+'.spkfull'
        full_spk.tofile(filename)

        # 2. parallel screen the noise id out, then gather from CPUs
        from ipyparallel import Client
        from ipyparallel.util import interactive
        rc = Client()
        cpu = rc[:]
        cpu.block = True

        @cpu.remote(block=True)      # to be executed by cpu
        @interactive                 # to be on the global()
        def get_noise_ids(filename, corr_cutoff):
            spk_data = np.memmap(filename, dtype='f4').reshape(-1, 25, 32)
            noise_id = []
            # corr_cutoff = 0.98
            # ind is index assign to each cpu
            # corr_cutoff is threshold of corr_coef
            for i in ind:
                spikes = spk_data[i]
                seq = np.abs(np.corrcoef(spikes[5:15, 16:].T).ravel())
                seq[seq>corr_cutoff] = 1
                seq[seq<corr_cutoff] = 0
                if np.median(seq) == 1.0:
                    noise_id.append(i)
            return noise_id

        # f = interactive(get_noise_ids)
        cpu.execute('import numpy as np')
        cpu.scatter('ind', range(nspk))
        noise_id = get_noise_ids(filename, corr_cutoff)
        # cpu.execute("%reset")
        try:
            os.remove(filename)
        except OSError:
            pass
        return np.hstack(np.asarray(noise_id))


    def remove_high_corr_noise(self, corr_cutoff=0.95):
        nid = self.get_nid(corr_cutoff)
        self.pivotal_pos = np.delete(self.pivotal_pos, nid, axis=1)


    def detect_spks(self):
        '''
            detect spikes by peakdet and threshhold. (channel by channel)
            
            return 
            -------
            [0],t  : array-like
                the time of spikes
            [1],ch : array-like
                the channel number of each spikes. so len(t) == len(ch)
        '''
        threshholds = self.bf.to_threshold(self.data)
        t = np.array([], dtype=np.int64)
        ch = np.array([], dtype=np.int32) 
  
        for i in range(self.data.shape[1]):
            _, mintab = peakdet(self.data[:,i], .3)    
            spks = mintab[mintab[:,1] < threshholds[i]].astype(np.int64)
            t = np.hstack((t, spks[:,0]))
            ch = np.hstack((ch, np.full(spks[:,0].shape, i)))
        
        return np.array([t,ch]) 
