import numpy as np
from vispy import scene, app
from ..base.CLU import CLU
from .color_scheme import palette
from ..core.correlate import correlate
from phy.plot import View

class correlogram_view(View):
    '''
        For grid purpose, phy.view is much faster than vispy.grid_view
    '''
    def __init__(self, correlate=correlate, fs=25e3, window_size=50, bin_size=1, show=False):
        super(correlogram_view, self).__init__('grid')
        
        self._window_size = window_size #ms
        self._bin_size = bin_size #ms
        self._fs = fs
        self._palette = palette
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
        self._window_size = window
        self._bin_size = bin
        
        assert self._window_size % 2 == 0
        assert self._window_size % self._bin_size == 0

    def set_data(self, ch, clu):
        self._clu = clu
        self._spike_time = self._get_spike_time(ch) 
        self.grid.shape = (self._clu.nclu,self._clu.nclu)

        self.clear()
        # Not rendering immedially now, waiting for shortcut
        #  self._render()

    def bind(self, data, spktag):
        self._data = data
        self._spktag = spktag

    def change_correlate_func(self, func):
        self._correlate = func
        self._render()

    def on_mouse_press(self, e):
        '''
            for now, left-click within the area of view will trigger the render
            TODO:
                collect and assign all the shortcut event within the whole GUI.
        '''
        if e.button == 1:
            self._render()
        
    ### ----------------------------------------------
    ###              private method
    ### ----------------------------------------------

    def _correlogram(self):
        return self._correlate(self._spike_time, self._clu.index) 

    def _pair_clusters(self):
        '''
            pair every clusters but ignore the duplicate one.
        '''
        for i in reversed(range(self._clu.nclu)):
            for j in range(i + 1):
                yield i,j
    
    def _get_spike_time(self, ch):
        '''
            get all global idxs according ch_no from pivotal_pos
        '''
        return self._spktag.t[self._spktag.ch == ch]   

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
        hists = self._correlogram()

        with self.building():
            for i,j in self._pair_clusters():
                color = self._default_color if (i != j) else np.hstack((self._palette[i],1))
                row, col = self._clu.nclu - 1 - i, j
                self[row,col].hist(hist=hists[i][j],color=color)
