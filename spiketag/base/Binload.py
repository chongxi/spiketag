import numpy as np
import numexpr as ne
import os
import mmap

def memory_map(filename, access=mmap.ACCESS_WRITE):
    size = os.path.getsize(filename)
    fd = os.open(filename, os.O_RDWR)
    return mmap.mmap(fd, size, access=access)

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
        self.info0 = '{0} loaded, it contains: \n'.format(file_name)
        self.info1 = '{0} * {1} points ({2} bytes) \n'.format(self._npts, self._nCh, self._nbytes)
        self.info2 = '{0} channels with sampling rate of {1:.4f} \n'.format(self._nCh, self.fs)
        self.info3 = '{0:.3f} secs ({1:.3f} mins) of data'.format(self._npts/self.fs, self._npts/self.fs/60)
        print "#############  load data  ###################"
        print self.info0 + self.info1 + self.info2 + self.info3
        print "#############################################"

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

