import numpy as np
from .color_scheme import palette
from scatter_2d_view import scatter_2d_view

class amplitude_view(scatter_2d_view):
    ''' Amplitude view is sub-class of scatter_2d_view. For  marker(x, y), the x pos is the time, the y pos is the peak amplitude.
        
        Parameters
        ----------
        fs : float
            sample rate
        time_tick : int
            the unit(s) of time tick in x axis
    '''
    def __init__(self, time_tick=1):
        super(amplitude_view, self).__init__()
        super(amplitude_view, self).attach_xaxis()

        self._time_tick = time_tick 


    ### ----------------------------------------------
    ###              public method 
    ### ----------------------------------------------

    def bind(self, data, spktag):
        self._spktag = spktag
        self._fs = spktag.probe.fs
        self._scale = data.max() - data.min()

    def set_data(self, ch, spk=None, clu=None):
        self._spike_time = self._get_spike_time(ch)
        self._spk = spk
        self._clu = clu
        
        @self._clu.connect
        def on_select_clu(*args, **kwargs):
            self._draw(self._clu.select_clus, delimit=False)

        @self._clu.connect
        def on_select(*args, **kwargs):
            self.highlight(self._clu.selectlist)
        
        @self._clu.connect
        def on_cluster(*args, **kwargs):
            self._clu.select_clu(self._clu.index_id)

        self._draw(self._clu.index_id)

    @property
    def binsize(self):
        return self._fs * self._time_tick

    def highlight(self, global_idx):
        ''' Transform the global idx to the view idx:
                Listen the select event from other view, and find the intersect spikes in current clus which selected to display within amplitude view. 
        '''
        # find the intersect cluster between other view and amplitude view
        local_idx = self._clu.global2local(global_idx)
        current_clus = self._clu.select_clus
        common_clus = np.intersect1d(current_clus, np.array(local_idx.keys()))
        
        # the spike idx in parent-class is |cluster1|cluster2|cluster3|....|,
        # so the local idx in cluster2 is need to plus len(cluster1)
        view_idx = np.array([],dtype='int64')
        if len(common_clus) > 0:
            for clu in common_clus:
                before = current_clus[np.where(current_clus < clu)]
                for b in before:
                    local_idx[clu] += self._clu.index_count[b]
                view_idx = np.hstack((view_idx, local_idx[clu]))
        
        super(amplitude_view, self)._highlight(view_idx)

    def select(self, view_idx):
        ''' 
            Transfrom the view idx to the global idx.
        '''
        # all clusters within the view currently
        current_clus = self._clu.select_clus
        local_idx = {}
        
        # assign idx to different range |cluster1|cluser2|cluster3|....|
        # according the length of cluster
        left = 0
        for clu in current_clus:
            right = left + self._clu.index_count[clu]
            index = view_idx[(view_idx>=left)&(view_idx<right)]
            if len(index) >  0:
                local_idx[clu] = index - left
            left = right
        global_idx = self._clu.local2global(local_idx)
        self._clu.select(global_idx)


    ### ----------------------------------------------
    ###              private method 
    ### ----------------------------------------------

    def _get_spike_time(self, ch):
        return self._spktag.t[self._spktag.ch == ch]

    def _locate_amplitude(self, clu):
        '''
            locate the peak of amplitude, return index and peak value
        '''
        times = self._spike_time[self._clu.index[clu]]
        # peak always heppenned one offset before
        # TODO may not use constant
        amplitudes = self._spk[self._clu.index[clu], 7, 1] / self._scale
        
        return  times / self.binsize, amplitudes
 

    def _draw(self, clus, delimit=True):
        '''
            The x pos is time, the y pos is amplitude, and the color and pos is pairwise.
            Draw clu by clu because have to match the color
        '''
        poses = None
        colors = None
        
        for clu in clus:
            x, y = self._locate_amplitude(clu) 
            pos = np.column_stack((x, y))
            color = np.tile(np.hstack((palette[clu],1)),(pos.shape[0],1))
        
            if poses is None and colors is None:
                poses = pos
                colors = color
            else:
                poses = np.concatenate((poses, pos))
                colors = np.concatenate((colors, color))

        super(amplitude_view, self).set_data(pos=poses, colors=colors, delimit=delimit) 

