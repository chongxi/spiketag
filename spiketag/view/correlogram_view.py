import numpy as np
from vispy import scene, app
from ..base.CLU import CLU
from .color_scheme import palette
from ..core.correlate import correlate
from phy.plot import View

class correlogram_view(View):

    ''' For grid purpose, phy.view is much faster than vispy.grid_view
       
        Parameters
        ----------

        correlate : func
            a func to calculate correlate which defined within core/correlate.py
        fs : float
            sample rate
        window_bins : int
            the number of bins of window
        bin_size : int
            the time interval(ms) to sum spikes.
    '''
    def __init__(self, correlate=correlate, window_bins=50, bin_size=1, show=False):
        super(correlogram_view, self).__init__('grid')
        
        self._window_bins = window_bins
        self._bin_size = bin_size 
        self._default_color = np.ones(4,dtype='int32')

        # inject the function to calculate correlare
        self._correlate = correlate
    

    ### ----------------------------------------------
    ###              public method 
    ### ----------------------------------------------

    def set_bin_window(self, bin=None, window=None):
        '''
            set size of bin and window, the unit is ms
        '''
        self._window_bins = window
        self._bin_size = bin
        
        assert self._window_bins % 2 == 0
        assert self._window_bins % self._bin_size == 0

    def set_data(self, clu, spk_times):
        self._clu = clu
        self._spike_time = spk_times 

        # Not rendering immedially now, waiting for shortcut
        self._render()

        @self._clu.connect
        def on_cluster(*args, **kwargs):
            self._render()

    def bind(self, spktag):
        self._spktag = spktag
        self._fs = spktag.probe.fs

    def change_correlate_func(self, func):
        '''
            change the correlate func
        '''
        self._correlate = func
        self._render()
    
    ### ----------------------------------------------
    ###              private method
    ### ----------------------------------------------

    def _correlogram(self):
        return self._correlate(self._spike_time, self._clu.membership, self._clu.index_id, window_bins=self._window_bins, bin_size=self._bin_size) 

    def _pair_clusters(self):
        '''
            pair every clusters but ignore the duplicate one.
        '''
        for i in reversed(range(self._clu.nclu)):
            for j in range(i + 1):
                yield i,j
    
    def _render(self):
        '''
            draw correlogram within grid. eg: if we have 4 clu:
                3+ + + +
                2+ + +
                1+ + 
                0+ 
                 0 1 2 3
            on the digonal line, it is the auto-correlogram, so the cluster will have the same color which in spike view.
        '''
        self.clear()
        self.grid.shape = (self._clu.nclu,self._clu.nclu)

        hists = self._correlogram()
   
        # begin draw
        with self.building():
            for i,j in self._pair_clusters():
                color = self._default_color if (i != j) else np.hstack((palette[i],1))
                row, col = self._clu.nclu - 1 - i, j
                self[row,col].hist(hist=hists[i][j],color=color)
