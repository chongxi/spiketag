#--------------------------------------------------------------
# Here is for all correlote method
#--------------------------------------------------------------
import numpy as np


def CCG(spk_time, spk_id, window_bins=50, bin_size=1, fs=25000.0):
    '''
        calculate the CCG from spike times of multiple neurons

        Parameter
        ---------
        spk_time: a numpy array of spike times of N neurons (spk_time in #samples)
        spk_id  : a numpy array of spike id    of N neurons
        window_bins: #bins in the window for calculating the CCG
        bin_size:    #ms of a single bin
        fs:          sampling rate of the spike train

        Return
        ---------
        ccg: a CCG matrix (N, N, #bins)
        ccg[i,j] is the cross-cologram of neuron pair (i,j)
    '''
    ccg = correlate(spk_time, 
                    spk_id, np.unique(spk_id), 
                    fs=fs,
                    window_bins=window_bins, 
                    bin_size=bin_size)
    return ccg


def correlate(spike_time, membership, cluster_ids, fs=25e3, window_bins=50, bin_size=1):
    '''
        Compute cross-correlate for every pair of clusters. Learn this algorithm from phy, but make it accurate and faster.

        Parameter
        ---------
        spike_time : array-like
            the time of spikes
        membership : array-like
            the cluster of spikes respectly, the len should be equal to spike time
        cluster_ids : array-like
            all cluster ids in membership
        fs : float
            sample rate
        window_bins : int
            the number of bins within window
        bin_size : int 
            the time range(ms) of a bin
    '''

    assert bin_size != 0
    assert window_bins % 2 == 0
    assert fs > 0
    assert spike_time.size == membership.size

    # the offset within the array
    bin_offset = int(bin_size * fs / 1e3)
    radius = window_bins // 2

    n_clu = len(cluster_ids)
    half_ccg = np.zeros([n_clu, n_clu, int(window_bins // 2 + 1)],dtype='int64')
    
    # shift = 1 mean calcute the time between current spike to the 1 before
    # spike, and etc.
    shift = 1
    mask = np.ones_like(spike_time, dtype=np.bool)

    # if the time interval is beyond the window, break the loop.
    while mask[:-shift].any():
        
        spike_offsets = _offset_after_shifted(spike_time, shift)
        # add (bin_offset-1) can make sure bin_offs are right
        spike_bin_offs = (spike_offsets + (bin_offset - 1)) // bin_offset
        
        # mark the offset beyond the window size
        mask[:-shift][spike_bin_offs > radius] = False
        m = mask[:-shift].copy()
        
        # get all time offs within window size        
        time_offs = spike_bin_offs[m]
        # locate the index within ccg matrix
        idx = np.ravel_multi_index((membership[:-shift][m],membership[shift:][m], time_offs), half_ccg.shape)
        # increment the number of spikes by index 
        _increment(half_ccg.ravel(), idx)

        shift += 1
    
    half_ccg[np.arange(n_clu), np.arange(n_clu), 0] = 0

    # symmetrize each correlate by the first column
    return _symmetrize_ccg(half_ccg)



def _offset_after_shifted(spike_time, shift):
    return spike_time[shift:] - spike_time[:-shift]

def _increment(array, idx):
    counts = np.bincount(idx)
    array[:len(counts)] += counts 
    return array

def _symmetrize_ccg(half_ccg):
    n_clu, _ , n_bin = half_ccg.shape

    half_ccg[..., 0] = np.maximum(half_ccg[..., 0], half_ccg[..., 0].T)
    sym = half_ccg[..., 1:][..., ::-1]
    sym = np.transpose(sym, (1, 0, 2))

    return np.dstack((sym, half_ccg[...,1:]))
