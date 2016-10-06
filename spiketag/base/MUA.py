import os
import numpy as np
from numba import jit
from multiprocessing import Pool
from .SPK import SPK
from .Binload import bload


@jit(cache=True)
def _to_spk(data, pos, chlist, spklen=19, prelen=8, ch_span=1):
    n = len(pos)
    spk = np.empty((n, spklen, 2*ch_span+1), dtype=np.float32)
    for i in range(n):
        # i spike in chlist
        spk[i, ...]  = data[pos[i]-prelen:pos[i]-prelen+spklen, chlist]
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
        self.ch  = range(nCh)
        self.fs  = fs*1.0
        self.data = bf.asarray(binpoint=binary_radix)
        self.t    = bf.t

        self.npts = bf._npts
        self.spklen = 19
        self.prelen = 8
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
            spkdict[ch] = _to_spk(data   = self.data, 
                                  pos    = pos, 
                                  chlist = self.ch_hash[ch], 
                                  spklen = self.spklen,
                                  prelen = self.prelen,
                                  ch_span= ch_span)
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
