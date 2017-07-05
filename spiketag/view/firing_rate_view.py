import numpy as np
from .color_scheme import palette
from line_view import line_view
from scipy import signal

class firing_rate_view(line_view):

    ''' Firing rate view, extend line view.
        The result is just approximately firing rate because use gaussian window to slide over the spike train.
        
        Parameter
        ---------
        fs : float
            sample rate
    '''
    def __init__(self):
        super(firing_rate_view, self).__init__()
       
        self.unfreeze()

        self._time_tick = 0.1 

    ### ----------------------------------------------
    ###              public method 
    ### ----------------------------------------------
    
    @property
    def time_tick(self):
        '''
          For future use, to adjust dalta T  to calculate the firing rate
        '''
        return self._time_tick
    
    @time_tick.setter
    def time_tick(self, v):
        if v > 1:
            self._time_tick = 1
        elif v <= 0:
            self._time_tick = 0.1
        else:
            self._time_tick = v


    def bind(self, spktag):
        self._spktag = spktag
        self._fs = spktag.probe.fs

    def set_data(self, clu=None, spk_times=None):
        self._spike_time = spk_times
        self._clu = clu

        @self._clu.connect
        def on_select_clu(*args, **kwargs):
            self._draw(self._clu.select_clus)

        @self._clu.connect
        def on_cluster(*args, **kwargs):
            self._clu.select_clu(self._clu.index_id)

        self._draw(self._clu.index_id)

    ### ----------------------------------------------
    ###              private method 
    ### ----------------------------------------------
    def _convolve_firing_rate(self, clu):
        '''
            convolve firing rate usiing gaussian window
        '''
        times = self._spike_time[self._clu.index[clu]] / int(self._fs * self._time_tick)
        counts = np.bincount(times) * int(1/self._time_tick)
        gs_window = signal.gaussian(counts.size/10, std=counts.size/5)
        rate  = np.convolve(counts, gs_window, mode='same') / sum(gs_window)
 
        return rate

    def _draw(self, clus):
      
        poses = []
        colors = []
        

        for clu in clus:
            rate = self._convolve_firing_rate(clu) 
            x, y = np.arange(1, rate.shape[0] + 1), rate 
            color = np.hstack((palette[clu],1))
            pos = np.column_stack((x,y))
            poses.append(pos)
            colors.append(color)

        super(firing_rate_view, self).set_data(poses, colors)


        
