import numpy as np
import torch
import torch.nn as nn
from scipy.stats import zscore
from scipy.signal import find_peaks

# define a time series class
class TimeSeries(object):
    '''
    Time series with different sampling rate, different start offset, and different length can cause headache when 
    analyzing and visualizing them. This class is designed to extract and align time series without resampling. 

    sig = TS(t, data, name='sig')
    sig_ROI = sig.between(0.5, 1.5) # return another TS object with t and data between 0.5 and 1.5
    sig_ROI.plot(ax=ax) # plot the ROI

    Examples:
    ---------
    # 1. load data from different sources (with different sampling rate)
    lfp = TS(t = lfp_ch18['t'], data = lfp_ch18['lfp'], name='lfp')
    unit = UNIT(bin_len=0.1, nbins=50) # 100ms bin, 50 bins
    unit.load_unitpacket('./fet.bin')
    bmi_time = unit.bin_index/10
    hdv = np.fromfile('./animal_hdv.bin', np.float32).reshape(-1,2)
    hd, v = hdv[:,0] - 90, hdv[:,1]

    # 2. load lfp,spk,vel as a TimeSeries object
    lfp = TS(t = lfp_18['t'], data = lfp_18['lfp'], name='lfp')
    spk = TS(t = unit.spike_time[unit.spike_id!=0], data = unit.spike_id[unit.spike_id!=0], name='spike_timing')
    vel = TS(t = bmi_time, data=v, name='ball_velocity')

    # 3. extract common ROI
    t_start, t_end = 737.6, 738.6
    _lfp = lfp.between(t_start, t_end)
    _spk = spk.between(t_start, t_end)
    _bv  = vel.between(t_start, t_end)

    ### check the ROI time points, they can be very different: _lfp.t.shape, _bv.t.shape, _spk.t.shape

    # 4. plot together although they have different length in the same time period
    fig, ax = plt.subplots(3,1,figsize=(15,8), sharex=True)
    _spk.scatter(ax = ax[0], c=_spk.data, s=2, cmap='Set2')
    _lfp.plot(ax = ax[1]);
    _bv.plot(ax = ax[2]);
    ax[0].spines['bottom'].set_visible(False)
    ax[1].spines['bottom'].set_visible(False)
    fig.tight_layout()
    ---------
    '''
    def __init__(self, t, data, name=None):
        self.t = t
        self.data = data
        self.name = name
        # self.fs = 1/(t[1]-t[0])
        self.ndim = self.data.ndim
        if self.ndim == 1:
            self.data = self.data.reshape(-1, 1)
        self.nch = self.data.shape[1]

    def between(self, start_time, end_time):
        idx = np.where((self.t >= start_time) & (self.t <= end_time))[0]
        return TimeSeries(self.t[idx], self.data[idx], self.name)
    
    def find_peaks(self, z_high=None, z_low=1, **kwargs):
        peaks, left, right = {}, {}, {}
        for ch in range(self.nch):
            if z_high is not None:
                self.threshold = np.median(self.data[:,ch]) + z_high * np.std(self.data[:,ch])
                _peaks_idx, _ = find_peaks(self.data[:,ch], height=self.threshold, **kwargs)
            else:
                _peaks_idx, _ = find_peaks(self.data, **kwargs)
            _left_idx, _right_idx = self.find_left_right_nearest(
                np.where(self.data[:, ch] < z_low * np.std(self.data[:, ch]))[0][:-1], _peaks_idx)
            peaks[ch] = TimeSeries(self.t[_peaks_idx], self.data[_peaks_idx], self.name+'_peaks_'+str(ch))
            left[ch]  = TimeSeries(self.t[_left_idx], self.data[_left_idx], self.name+'_left_'+str(ch))
            right[ch] = TimeSeries(self.t[_right_idx], self.data[_right_idx], self.name+'_right_'+str(ch))
        return peaks, left, right

    def find_left_right_nearest(self, x_idx, v_idx):
        """
        Find the adjacent index of v_idx (N,) in x_idx (return the N left index of a, and N right index of a)
        """
        _idx_right = np.searchsorted(x_idx, v_idx)
        _idx_left = np.searchsorted(x_idx, v_idx) - 1
        left = x_idx[_idx_left] - 1
        right = x_idx[_idx_right] 
        return left, right

    def plot(self, ax=None, **kwargs):
        if ax is None:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(1,1,figsize=(12,5))
        ax.plot(self.t, self.data, **kwargs)
        return ax

    def scatter(self, ax=None, **kwargs):
        if ax is None:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(1,1,figsize=(12,5))
        ax.scatter(self.t, self.data, **kwargs)
        return ax

    def __getitem__(self, idx):
        return TimeSeries(self.t[idx], self.data[idx], self.name)

    def __len__(self):
        return len(self.t)

    def __repr__(self):
        return f'{self.name} from {self.t[0]} to {self.t[-1]}'

    def __add__(self, other):
        return TimeSeries(self.t, self.data + other.data, self.name)

    def __sub__(self, other):
        return TimeSeries(self.t, self.data - other.data, self.name)

    def __mul__(self, other):
        return TimeSeries(self.t, self.data * other.data, self.name)

    def __truediv__(self, other):
        return TimeSeries(self.t, self.data / other.data, self.name)

    def __pow__(self, other):
        return TimeSeries(self.t, self.data ** other.data, self.name)

    def __neg__(self):
        return TimeSeries(self.t, -self.data, self.name)
    
    def __abs__(self):
        return TimeSeries(self.t, np.abs(self.data), self.name)

    def __eq__(self, other):
        return TimeSeries(self.t, self.data == other.data, self.name)

    def __ne__(self, other):
        return TimeSeries(self.t, self.data != other.data, self.name)

    def __lt__(self, other):
        return TimeSeries(self.t, self.data < other.data, self.name)

    def __le__(self, other):
        return TimeSeries(self.t, self.data <= other.data, self.name)

    def __gt__(self, other):
        return TimeSeries(self.t, self.data > other.data, self.name)

    def __ge__(self, other):
        return TimeSeries(self.t, self.data >= other.data, self.name)

    def __and__(self, other):
        return TimeSeries(self.t, self.data & other.data, self.name)

    def __or__(self, other):
        return TimeSeries(self.t, self.data | other.data, self.name)

    def __xor__(self, other):
        return TimeSeries(self.t, self.data ^ other.data, self.name)

    def __invert__(self):
        return TimeSeries(self.t, ~self.data, self.name)

    def __lshift__(self, other):
        return TimeSeries(self.t, self.data << other.data, self.name)

    def __rshift__(self, other):
        return TimeSeries(self.t, self.data >> other.data, self.name)

    def __iadd__(self, other):
        self.data += other.data
        return self

    def __isub__(self, other):
        self.data -= other.data
        return self

    def __imul__(self, other):
        self.data *= other.data
        return self

    def __itruediv__(self, other):
        self.data /= other.data
        return self

    def __ipow__(self, other):
        self.data **= other.data
        return self

    def __iand__(self, other):
        self.data &= other.data
        return self

    def __ior__(self, other): 
        self.data |= other.data
        return self

    def __ixor__(self, other):
        self.data ^= other.data
        return self

    def __ilshift__(self, other):
        self.data <<= other.data
        return self

    def __irshift__(self, other):
        self.data >>= other.data
        return self
