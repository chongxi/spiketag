import os
import mmap
from numba import jit
import numexpr as ne
import numpy as np
from ..utils import Timer 
from ..utils.conf import info

def memory_map(filename, access=mmap.ACCESS_WRITE):
    size = os.path.getsize(filename)
    fd = os.open(filename, os.O_RDWR)
    return mmap.mmap(fd, size, access=access)

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


class bload(object):
    '''
    load bin file data with file format
    init the nCh and fs when create the instance

    bf = Binload(nCh=16,fs=30000)
    bf.load('./137_36_shankD_116.dat')

    1. bf.t is time; bf.asarray(binpoint) return data
    2. bf.asarray(binpoint=14) to convert int to fix-point
    3. bf.to_threshold(k=4.5) to export median-based threshold
    4. (bf._npts, bf._nCh, bf._nbytes) are metadata
    '''
    
    def __init__(self, nCh=16, fs=30000):
        self._nCh = nCh
        self.fs = float(fs)

    def load(self, file_name, dtype='int32'):
        '''
        bin.load('filename','int16')
        bin.load('filename','float32')
        '''
        self.mm   = memory_map(file_name)
        self.npmm = np.memmap(file_name, dtype=dtype, mode='r')
        self.dtype = dtype
        self._npts = len(self.npmm)/self._nCh #full #pts/ch
        self._nbytes = self.npmm.nbytes
        info("#############  load data  ###################")
        info('{0} loaded, it contains: '.format(file_name))
        info('{0} * {1} points ({2} bytes) '.format(self._npts, self._nCh, self._nbytes))
        info('{0} channels with sampling rate of {1:.4f} '.format(self._nCh, self.fs))
        info('{0:.3f} secs ({1:.3f} mins) of data'.format(self._npts/self.fs, self._npts/self.fs/60))
        info("#############################################")

        dt = 1/self.fs
        self.t = np.linspace(0,self._npts*dt,self._npts,endpoint='false')

    def __repr__(self):
        return self.info0 + self.info1 + self.info2 + self.info3

    def asarray(self, binpoint=14):
        ne.set_num_threads(32)
        data = np.asarray(self.npmm).reshape(-1, self._nCh)
        _scale = np.float32(2**binpoint)
        return ne.evaluate('data/_scale')

    def to_threshold(self, data, k=4.5):
        # QQ threshold for spike detection
        data = data[::20,:]
        thres_arr = -np.median(ne.evaluate('abs(data)*k/0.675'), axis=0)
        return thres_arr

    def detect_spks(self, delta=.3):
        '''
            detect spikes by peakdet and threshhold. (channel by channel)
            
            return 
            -------
            [0],t  : array-like
                the time of spikes
            [1],ch : array-like
                the channel number of each spikes. so len(t) == len(ch)
        '''
        with Timer('[MODEL] Binload -- covert to data'):
            data = self.asarray()
        
        with Timer('[MODEL] Binload -- threshholds'):
            threshholds = self.to_threshold(data)
        
        t = np.array([], dtype=np.int64)
        ch = np.array([], dtype=np.int32) 
  
        for i in range(data.shape[1]):
            with Timer('[MODEL] Binload -- cal peakdet for ' + str(i)):
                _, mintab = peakdet(data[:,i],delta)    
            spks = mintab[mintab[:,1] < threshholds[i]].astype(np.int64)
            t = np.hstack((t, spks[:,0]))
            ch = np.hstack((ch, np.full(spks[:,0].shape, i, dtype=np.int64)))

        return np.array([t, ch])
