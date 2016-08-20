from numba import int32, int64, float32, jit
from phy.plot.utils import _get_array
from phy.io.array import _accumulate


# @jit('(int64, int64, int64, int32[:], int64, int32[:], int64[:])', cache=True)
@jit(cache=True)
def _spkNo2maskNo_numba(n_signals, n_samples, n_ch, clu_offset, cluNo, spkNolist, mask):
    '''
    turn the spkNolist(local) and cluNo into spike_view mask
    for highlight
    '''
    i = 0
    offset = clu_offset[cluNo]
    for spkNo in spkNolist:
        for ch in xrange(n_ch):
            start = n_samples*(spkNo + ch*n_signals + offset)
            end   = n_samples*(spkNo + ch*n_signals + offset + 1)
            for j in xrange(start, end):
                mask[i] = j
                i += 1

@jit(cache=True)
def _cache_out(_cache_mask, _cache, target):
    N = len(_cache_mask)
    M = target.shape[1]
    for i in xrange(N):
        k = _cache_mask[i]
        for j in xrange(M):
            target[k,j] = _cache[k,j]


@jit(cache=True)
def _cache_in_matrix(mask, source, target):
    N = len(mask)
    M = target.shape[1]
    for i in xrange(N):
        k = mask[i]
        for j in xrange(M):
            target[k,j] = source[i,j]


@jit(cache=True)
def _cache_in_vector(mask, source, target):
    N = len(mask)
    M = target.shape[1]
    for i in xrange(N):
        k = mask[i]
        for j in xrange(M):
            target[k,j] = source[j]


@jit(cache=True)
def _cache_in_scalar(mask, source, target):
    N = len(mask)
    M = target.shape[1]
    for i in xrange(N):
        k = mask[i]
        for j in xrange(M):
            target[k,j] = source


def _representsInt(s):
    try: 
        int(s)
        return True
    except ValueError:
        return False
