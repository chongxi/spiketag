import os
import numpy as np
from numba import njit
from multiprocessing import Pool
from ..view import wave_view 
from .SPK import SPK
from .Binload import bload
from ..utils.conf import info
from ..utils import Timer

def _calculate_threshold(x, beta):
    thr = -beta*np.median(abs(x)/0.6745,axis=0)
    return thr


def find_spk_in_time_seg(spk_times, time_segs):
    spk_times_in_range = []
    for time_seg in time_segs:
        spk_in_time_seg = spk_times[np.logical_and(spk_times<time_seg[1], spk_times>time_seg[0])]
        spk_times_in_range.append(spk_in_time_seg)
    return np.hstack(np.array(spk_times_in_range))


@njit(cache=True)
def _to_spk(data, pos, chlist, spklen=19, prelen=7, cutoff_neg=-5000, cutoff_pos=1000):
    nspks = len(pos)
    spk = np.empty((nspks, spklen, len(chlist)), dtype=np.float32)
    noise_idx = []
    for i in range(nspks):
        spike_wav = data[pos[i]-prelen:pos[i]-prelen+spklen, chlist]
        spk[i, ...] = spike_wav
        peaks  = spike_wav.reshape(1,-1)
        if peaks.min()<cutoff_neg or peaks.max()>cutoff_pos: 
            noise_idx.append(i)
    _nan = np.where(chlist==-1)[0]
    spk[..., _nan] = 0
    return spk, np.array(noise_idx)


def idx_still_spike(time_spike, time_still, dt):
    idx = np.searchsorted(time_still, time_spike) - 1
    dd = time_spike - time_still[idx]
    idx_still = np.where(dd < dt)[0]
    assert(np.max(time_spike[idx_still] - time_still[idx[idx_still]])<dt)
    return idx[idx_still], idx_still


class MUA(object):
    def __init__(self, mua_filename, probe, numbytes=4, binary_radix=13, scale=True, 
                 spk_filename=None, 
                 cutoff=[-1500, 1000], time_segs=None, time_still=None, lfp=False):
        '''
        mua_filename:
        spk_filename:
        probe:
        numbytes:
        binary_radix:
        cutoff: [a, b] is voltage range for the pivotal peak. Value < a or Value > b will be filtered
        time_segs: [[a,b],[c,d]] The time segments for extracting spikes for sorting (unit in seconds) 
        ''' 
        self.nCh = probe.n_ch
        self.fs  = probe.fs*1.0
        self.probe = probe
        self.numbytes = numbytes
        self.dtype = 'i'+str(self.numbytes)
        self.bf = bload(self.nCh, self.fs)
        self.bf.load(mua_filename, dtype=self.dtype)
        self.mua_file = mua_filename
        self.binary_radix = binary_radix
        self.scale = scale
        self._scale_factor = 1.0 if self.scale else np.float32(2**self.binary_radix)
        if probe.reorder_by_chip is True:
            self.bf.reorder_by_chip(probe._nchips)
        if scale is True:
            with Timer('scale the data: convert data from memmap to numpy with radix {}'.format(self.binary_radix), verbose=True):
                self.data = self.bf.asarray(binpoint=self.binary_radix)
        else:
            self.data = self.bf.data.numpy().reshape(-1, self.nCh)

        self.t    = self.bf.t
        self.npts = self.bf._npts
        self.spklen = 19
        self.prelen = 9 
        self.cutoff_neg, self.cutoff_pos = cutoff[0], cutoff[1]
        if time_segs is None:
            self.time_segs = np.array([[self.t[0], self.t[-1]]])
        else:
            self.time_segs = np.array(time_segs)

        self.time_still = time_still


        # acquire pivotal_pos from spk.bin under same folder
        foldername = '/'.join(self.mua_file.split('/')[:-1])+'/'
        info('processing folder: {}'.format(foldername))
        # self.spk_file = self.mua_file[:-4] + '.spk.bin'
        if spk_filename is not None:
            self.spk_file = spk_filename
            spk_meta = np.fromfile(self.spk_file, dtype='<i4')
            self.pivotal_pos = spk_meta.reshape(-1,2).T

            # check spike is extracable
            # delete begin AND end
            self.pivotal_pos = np.delete(self.pivotal_pos, 
                               np.where((self.pivotal_pos[0] + self.spklen) > self.data.shape[0])[0], axis=1)

            self.pivotal_pos = np.delete(self.pivotal_pos, 
                               np.where((self.pivotal_pos[0] - self.prelen) < 0)[0], axis=1)        

            if lfp:
                self.pivotal_pos[0] -= 20

            info('raw data have {} spks'.format(self.pivotal_pos.shape[1]))
            info('----------------success------------------')
            info(' ')
        else:
            self.pivotal_pos = None
            info('no spike file provided')


    def get_threshold(self, beta=4.0):
        return _calculate_threshold(self.data[::100], beta)

    def tofile(self, file_name, nchs, dtype=np.int32):
        data = self.data[:, nchs].astype(dtype)
        data.tofile(file_name)

    def _get_spk_times(self, group_id, time_segs, method='spk_info'):
        if method == 'spk_info':
            pivotal_pos = self.pivotal_pos
            pivotal_chs = self.probe[group_id]
            spk_times = pivotal_pos[0][np.in1d(pivotal_pos[1], pivotal_chs)]
            spk_times = find_spk_in_time_seg(spk_times, time_segs*self.fs)
        return spk_times

    def _tospk(self, group_id, time_segs, method='spk_info'):
        pivotal_chs = self.probe[group_id]
        spk_times   = self._get_spk_times(group_id, time_segs, method)
        if spk_times.shape[0] > 0:
            spks, noise_idx = _to_spk(data   = self.data, 
                                      pos    = spk_times, 
                                      chlist = pivotal_chs, 
                                      spklen = self.spklen,
                                      prelen = self.prelen,
                                      cutoff_neg = self.cutoff_neg * self._scale_factor,
                                      cutoff_pos = self.cutoff_pos * self._scale_factor)
            if self.scale is True: # already scaled
                return spks, spk_times, noise_idx
            else:                  # haven't scaled so need to be scaled here
                return spks/self._scale_factor, spk_times, noise_idx
        else:
            return None, None, None

    def _delete_spks(self, spks, spk_times, noise_idx):
        spks = np.delete(spks, noise_idx, axis=0)
        spk_times = np.delete(spk_times, noise_idx, axis=0)
        return spks, spk_times

    def tospk(self, amp_cutoff=True, speed_cutoff=False, time_cutoff=True):
        info('mua.tospk() with time_cutoff={}, amp_cutoff={}, speed_cutoff={}'.format(
                               time_cutoff,    amp_cutoff,    speed_cutoff))
        self.spkdict = {}
        self.spk_times = {}
        for g in self.probe.grp_dict.keys():
            spks, spk_times, noise_idx = self._tospk(group_id=g,  time_segs=self.time_segs, method='spk_info')
            ### remove noise from spike
            if amp_cutoff is True and spks is not None:
                n_noise = float(noise_idx.shape[0])
                n_spk   = float(spks.shape[0])
                info('group {} delete {}%({}/{}) spks via cutoff'.format(g, n_noise/n_spk*100, n_noise, n_spk))
                self.spkdict[g], self.spk_times[g] = self._delete_spks(spks, spk_times, noise_idx)
            else:
                self.spkdict[g], self.spk_times[g] = spks, spk_times

            ### remove spike during v_smoothed < 5cm/sec
            if speed_cutoff is True and self.time_still is not None and spks is not None:
                _, idx_still = idx_still_spike(self.spk_times[g]/self.fs, self.time_still, 1/60.)
                n_idx_still = float(idx_still.shape[0])
                n_spk       = float(self.spk_times[g].shape[0])
                info('group {} delete {}%({}/{}) spks via speed'.format(g, n_idx_still/n_spk*100, n_idx_still, n_spk))
                self.spkdict[g]   = np.delete(self.spkdict[g],   idx_still, axis=0)
                self.spk_times[g] = np.delete(self.spk_times[g], idx_still, axis=0)

        # check 0 spks case, fill in some random noise
        for g in self.probe.grp_dict.keys():
            if self.spkdict[g] is None:
                self.spkdict[g] = np.random.randn(1, self.spklen, len(self.probe[g]))
                self.spk_times[g] = np.array([0]) 
        info('----------------success------------------')     
        info(' ')               
        return SPK(self.spkdict)

    def get_nid(self, corr_cutoff=0.95):  # get noisy spk id
        # 1. dump spikes file (binary)
        piv = self.pivotal_pos.T
        nspk = self.pivotal_pos.shape[1]
        # the reason adding mod operation here is if the spike is in the very end ,i.e: within 15 offset to end 
        # point, this will make self.data[rows, :] out of bound.
        rows = (np.arange(-10,15).reshape(1,-1) + piv[:,0].reshape(-1,1)) % self.data.shape[0]
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
        def get_noise_ids(filename, corr_cutoff, n_group):
            spk_data = np.memmap(filename, dtype='f4').reshape(-1, 25, n_group)
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
        noise_id = get_noise_ids(filename, corr_cutoff, self.probe.n_group)
        # cpu.execute("%reset")
        try:
            os.remove(filename)
        except OSError:
            pass
        return np.hstack(np.asarray(noise_id))


    def remove_high_corr_noise(self, corr_cutoff=0.95):
        nid = self.get_nid(corr_cutoff)
        self.pivotal_pos = np.delete(self.pivotal_pos, nid, axis=1)
        info('removed noise ids: {} '.format(nid)) 

    # def remove_groups_under_fetlen(self, fetlen):
    #     ids = []
    #     groups = {}
    #     for g in self.probe.keys():
    #         pivotal_chs = self.probe.fetch_pivotal_chs(g)
    #         _ids = np.where(np.in1d(self.pivotal_pos[1], pivotal_chs))[0]
    #         if len(_ids) < fetlen:
    #             ids.extend(_ids)
    #             groups[g] = len(_ids)
    #     self.pivotal_pos = np.delete(self.pivotal_pos, ids, axis=1)
    #     info('removed all spks on these groups: {}'.format(groups)) 

    def group_spk_times(self):
        group_with_times = {}
        for g in range(self.probe.n_group):
            pos   = self.pivotal_pos[0][np.in1d(self.pivotal_pos[1], pivotal_chs)]
            times = self.pivotal_pos[0][np.where(np.in1d(self.pivotal_pos[1],self.probe[g]))[0]]
            if len(times) > 0: group_with_times[g] = times
        return group_with_times


    def show(self, chs, span=None, time=0):
        '''
        if self.pivotal_pos exsists, it is the spks
        spks: (t,ch) encodes pivital
        array([[  37074,   37155,   37192, ..., 1602920, 1602943, 1602947],
               [     58,      49,      58, ...,      58,      75,      77]], dtype=int32)
        '''
        if span is None:
            if self.pivotal_pos is not None:
                self.wview = wave_view(self.data, chs=chs, spks=self.pivotal_pos)
            else:
                self.wview = wave_view(self.data, chs=chs)
            self.wview.slideto(time * self.fs)
        else:
            start = int((time-span)*self.fs) if time>span else 0
            stop  = int((time+span)*self.fs) if (time+span)*self.fs<self.data.shape[0] else self.data.shape[0]
            if self.pivotal_pos is not None:
                self.wview = wave_view(self.data[start:stop], chs=chs, spks=self.pivotal_pos)
            else:
                self.wview = wave_view(self.data[start:stop], chs=chs)
            self.wview.slideto(span * self.fs)
        self.wview.show()
