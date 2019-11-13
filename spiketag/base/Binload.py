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
from scipy import signal
import os.path as op
from tqdm import tqdm


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


def lp_filter(fs, fstop, x):
    nyquist_fs = fs/2
    wstop = fstop/nyquist_fs
    b, a = signal.butter(5, wstop, analog=False)
    fx = torch.zeros_like(x)
    print(fx.shape)
    for i, _x in enumerate(x.transpose(0,1)):
        print('filter {}th channel'.format(i))
        _x = _x.numpy()
        fx[:, i] = torch.from_numpy(signal.filtfilt(b, a, _x).astype(_x.dtype)) 
    return fx

def get_clock_spk():
    import spiketag
    res_folder = op.join(spiketag.__path__[0], 'res')
    clk_spkwav = np.fromfile(res_folder+'/'+'clock_spk.bin', dtype=np.int16).reshape(-1, 4)
    return clk_spkwav

def shift(spk, n):
    return np.roll(spk, shift=n, axis=1)

def add_spikes(x, t, spk):
    x[t:t+spk.shape[0], :] += torch.from_numpy(spk).type(x.dtype)


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
        self.nyquist_fs = fs/2

    def load(self, file_name, dtype='int32'):
        '''
        bin.load('filename','int16')
        bin.load('filename','float32')
        '''
        self.mm   = memory_map(file_name)
        self.npmm = np.memmap(file_name, dtype=dtype, mode='readwrite')
        self.dtype = dtype
        self._npts = len(self.npmm)/self._nCh #full #pts/ch
        self._nbytes = self.npmm.nbytes
        self.t = fs2t(self._npts, self.fs)
        self.data = torch.from_numpy(self.npmm.reshape(-1, self._nCh))
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


    def _filter(self, chs, sample_seg=None, fstart=0, fstop=300, ftype='low-pass', noise_level=0):
        # base_line = np.zeros((bf.data.shape[0], 4))
        if ftype == 'low-pass':
            print('filter:{} -- fstart:{}, fstop:{}'.format(ftype, fstart, fstop))
            if sample_seg is None:
                x = self.data.reshape(-1, self._nCh)[:, chs]
            else:
                x = self.data.reshape(-1, self._nCh)[sample_seg[0]:sample_seg[1], chs]
            base_line = lp_filter(fs=self.fs, fstop=fstop, x=x)
            # base_line += (noise_level*torch.randn(*base_line.shape)).type(base_line.dtype)
            return base_line   


    def _generate_clock_chs(self, chs, clk_spkwav=None, start=3000, interval=1000):
        '''
        Important: The first spikes shows up at 120.5 ms at the second channel in chs (when start=3000)
        '''
        ## 1. get the baseline through low-pass filter
        base_line = self._filter(chs=chs)
        ## 2. add clock spikes, with shifted channels every 1000 samples
        if clk_spkwav is None:
            clk_spkwav = get_clock_spk()
        k = 0
        for i in range(start, base_line.shape[0], interval):
            add_spikes(x=base_line, t=i, spk=shift(clk_spkwav, k))
            k += 1
        return base_line 
        



    # def filter(self, band, ftype='low-pass', noise_level=0):
    #     self.data = self.data.reshape(-1, self._nCh)
    #     data = torch.zeros_like(self.data)
    #     for ch in range(self._nCh):
    #         data[:,ch] = self._filter(ch, band, ftype=ftype, noise_level=noise_level)
    #     self.data = data


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


    def convolve(self, kernel, scale=1, device='gpu'):
        from ..core import convolve
        data = self.data.numpy().reshape(-1, self._nCh)
        new_data = []
        for datum in tqdm(data.T):
            new_data.append(convolve(datum, kernel, device=device, scale=scale, mode='same'))
            # new_data.append(np.convolve(datum, kernel))
        self.data = np.vstack((new_data)).T


    def deconvolve(self, kernel):
        if torch.get_num_threads() == 1:
            import os
            torch.set_num_threads(os.cpu_count())
        if type(self.data) != np.ndarray:
            self.data = self.data.numpy().reshape(-1, self._nCh)
        length = self.data.shape[0] - len(kernel) + 1
        new_data = np.zeros((length, self.data.shape[1]), dtype=np.float32)
        # print(new_data.shape)
        for i in tqdm(range(self.data.shape[1])):
            # print('deconvolve {}th channel'.format(i))
            new_data[:, i] = _deconvolve(self.data[:,i], kernel)
        self.data = new_data
        torch.set_num_threads(1)


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
