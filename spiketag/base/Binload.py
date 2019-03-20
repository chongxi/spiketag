import os
import mmap
from numba import jit
import numexpr as ne
import numpy as np
import torch
from ..utils import Timer, interpNd
from ..utils.conf import info
from ..view import wave_view
import torch


def fs2t(N, fs):
    dt = 1./fs
    t = np.arange(0, N*dt, dt)
    return t

def fft(x):
    GPU = torch.cuda.is_available()
    if GPU:
        x = torch.from_numpy(x).cuda()
        fx = torch.rfft(x, 1, onesided=False)
        complex_fx = fx[:,0].cpu().numpy() + fx[:,1].cpu().numpy()*1j
        torch.cuda.empty_cache()
    else:
        x = torch.from_numpy(x)
        fx = torch.rfft(x, 1, onesided=False)
        complex_fx = fx[:,0].numpy() + fx[:,1].numpy()*1j
    return complex_fx

def ifft(complex_x):
    x = np.vstack((complex_x.real, complex_x.imag)).T
    # ifx = irfft(torch.from_numpy(x), 1, onesided=False)
    # if torch.cuda.is_available():
    x = torch.from_numpy(x)
    ifx = torch.ifft(x, 1)
    reconstruct_x = ifx[:,0]
    # torch.cuda.empty_cache()
    return reconstruct_x 

def _deconvolve(signal, kernel):
    _h = fft(kernel)
    length = len(signal) - len(kernel) + 1
    kernel = np.hstack((kernel, np.zeros(len(signal) - len(kernel), dtype=np.float32))) # zero pad the kernel to same length
    H = fft(kernel)
    deconvolved = ifft(fft(signal.astype(np.float32))*np.conj(H)/(H*np.conj(H)))
    return deconvolved[:length]

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
    
    def __init__(self, nCh=160, fs=25000):
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
        self.t = fs2t(self._npts, self.fs)
        self.data = torch.from_numpy(self.npmm)
        info("#############  load data  ###################")
        info('{0} loaded, it contains: '.format(file_name))
        info('{0} * {1} points ({2} bytes) '.format(self._npts, self._nCh, self._nbytes))
        info('{0} channels with sampling rate of {1:.4f} '.format(self._nCh, self.fs))
        info('{0:.3f} secs ({1:.3f} mins) of data'.format(self._npts/self.fs, self._npts/self.fs/60))
        info("#############################################")

        # dt = 1/self.fs
        # self.t = np.linspace(0,self._npts*dt,self._npts,endpoint='false')

    def __repr__(self):
        return self.info0 + self.info1 + self.info2 + self.info3


    def asarray(self, binpoint=13):
        ne.set_num_threads(32)
        try:
            data = self.data.reshape(-1, self._nCh).numpy()
        except:
            data = self.data.reshape(-1, self._nCh)
        _scale = np.float32(2**binpoint)
        self.data = torch.from_numpy(ne.evaluate('data/_scale'))
        return self.data.numpy()


    def reorder_by_chip(self, nchips=5):
        nch_perchip = self._nCh/nchips
        # self.npmm = self.npmm.reshape(-1, nch_perchip, nchips)
        # self.npmm = self.npmm.swapaxes(1,2)
        # self.npmm = self.npmm.reshape(-1, self._nCh)
        self.data = self.data.reshape(-1,nch_perchip, nchips).transpose(1,2).reshape(-1, self._nCh)
        info('reordered with nchips={0} and nch_perchip={1}'.format(nchips,nch_perchip))


    def resample(self, new_fs):
        self.data = torch.from_numpy(interpNd(self.data.numpy().reshape(-1, self._nCh), self.fs, new_fs, method='quadratic'))
        self.fs = new_fs


    def show(self, chs=None):
        if type(self.data) != np.ndarray:
            if chs is not None:
                self.wview = wave_view(data=self.data.numpy().reshape(-1, self._nCh), fs=self.fs, chs=chs)
            else:
                self.wview = wave_view(data=self.data.numpy().reshape(-1, self._nCh), fs=self.fs)
        else:
            if chs is not None:
                self.wview = wave_view(data=self.data.reshape(-1, self._nCh), fs=self.fs, chs=chs)
            else:
                self.wview = wave_view(data=self.data.reshape(-1, self._nCh), fs=self.fs)
        self.wview.show()


    def convolve(self, kernel):
        data = self.data.numpy().reshape(-1, self._nCh)
        new_data = []
        for datum in data.T:
            new_data.append(np.convolve(datum, kernel))
        self.data = np.vstack((new_data)).T


    def deconvolve(self, kernel):
        if type(self.data) != np.ndarray:
            self.data = self.data.numpy().reshape(-1, self._nCh)
        length = self.data.shape[0] - len(kernel) + 1
        new_data = np.zeros((length, self.data.shape[1]), dtype=np.float32)
        # print(new_data.shape)
        for i in range(self.data.shape[1]):
            print('deconvolve {}th channel'.format(i))
            new_data[:, i] = _deconvolve(self.data[:,i], kernel)
        self.data = new_data


    def normalize_columns(self, absmax=15000, dtype='int16'):
        if type(self.data) != np.ndarray:
            self.data = self.data.numpy().reshape(-1, self._nCh)
        rows, cols = self.data.shape
        new_data = np.zeros((rows, cols), dtype=dtype)
        for col in xrange(cols):
            new_data[:,col] = np.floor(self.data[:,col])/abs(self.data[:,col]).max()*absmax
            new_data[:,col] = new_data[:,col].astype(dtype)
        self.data = new_data


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
