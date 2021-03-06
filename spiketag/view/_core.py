from numba import int32, int64, float32, jit
from phy.plot.utils import _get_array
from phy.io.array import _accumulate


# @jit('(int64, int64, int64, int32[:], int64, int32[:], int64[:])', cache=True)
@jit(cache=True, nopython=True)
def _spkNo2maskNo_numba(n_signals, n_samples, n_ch, clu_offset, cluNo, spkNolist, mask):
    '''
    turn the spkNolist(local) and cluNo into spike_view mask
    for highlight
    '''
    i = 0
    offset = clu_offset[cluNo]
    for spkNo in spkNolist:
        for ch in range(n_ch):
            start = n_samples*(spkNo + ch*n_signals + offset)
            end   = n_samples*(spkNo + ch*n_signals + offset + 1)
            for j in range(start, end):
                mask[i] = j
                i += 1

@jit(cache=True, nopython=True)
def _cache_out(_cache_mask, _cache, target):
    N = len(_cache_mask)
    M = target.shape[1]
    for i in range(N):
        k = _cache_mask[i]
        for j in range(M):
            target[k,j] = _cache[k,j]


@jit(cache=True, nopython=True)
def _cache_in_matrix(mask, source, target):
    N = len(mask)
    M = target.shape[1]
    for i in range(N):
        k = mask[i]
        for j in range(M):
            target[k,j] = source[i,j]


@jit(cache=True, nopython=True)
def _cache_in_vector(mask, source, target):
    N = len(mask)
    M = target.shape[1]
    for i in range(N):
        k = mask[i]
        for j in range(M):
            target[k,j] = source[j]


@jit(cache=True, nopython=True)
def _cache_in_scalar(mask, source, target):
    N = len(mask)
    M = target.shape[1]
    for i in range(N):
        k = mask[i]
        for j in range(M):
            target[k,j] = source


# TODO: Using Numba to replace spkview._get_data() and spkview._build() might make the `cluster` event response much faster
@jit(cache=True, nopython=True)
def _get_box_index(box_index, n_ch, n_clu, spk_len, clu_list):
    i = 0
    for chNo in range(n_ch):
        for cluNo in range(n_clu):
            box_len = clu_list[cluNo].shape[0]*spk_len
#             print(box_len)
            box_index[i:i+box_len, 0] = chNo
            box_index[i:i+box_len, 1] = cluNo
            i+=box_len


def _representsInt(s):
    try: 
        int(s)
        return True
    except ValueError:
        return False
