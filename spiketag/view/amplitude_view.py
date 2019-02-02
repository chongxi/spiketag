import numpy as np
from .color_scheme import palette
from .scatter_2d_view import scatter_2d_view
from vispy.util import keys
from numba import njit


@njit(cache=True)
def _locate_amplitude(spk, spk_times, fs, clu_idx):
    '''
        locate the peak of amplitude, return index and peak value
    '''
    times = spk_times[clu_idx]
    nspks = clu_idx.shape[0]
    spk_group = spk[clu_idx, :].reshape(nspks, -1)
    time_amplitudes = np.zeros((nspks, 2))
    
    for i in range(nspks):
        time_amplitudes[i, 0] = times[i]
        time_amplitudes[i, 1] = np.min(spk_group[i])
    return  time_amplitudes


class amplitude_view(scatter_2d_view):
    ''' Amplitude view is sub-class of scatter_2d_view. For  marker(x, y), the x pos is the time, the y pos is the peak amplitude.
        
        Parameters
        ----------
        fs : float
            sample rate
        scale : float
            normalization of amplitudes
        time_tick : int
            the unit(s) of time tick in x axis
    '''
    def __init__(self, fs=25e3, scale=1.0, time_tick=1):
        super(amplitude_view, self).__init__()
        super(amplitude_view, self).attach_xaxis()

        self._time_tick = time_tick 
        self._fs = float(fs)
        self._scale = scale



    ### ----------------------------------------------
    ###              public method 
    ### ----------------------------------------------

    def set_data(self, spk=None, clu=None, spk_times=None):
        self._spike_time = spk_times 
        self._spk = spk
        self._clu = clu
        self._draw(self._clu.index_id)


    def register_event(self):
        @self._clu.connect
        def on_select_clu(*args, **kwargs):
            self._draw(self._clu.select_clus, delimit=False)

        @self._clu.connect
        def on_select(*args, **kwargs):
            self.highlight(self._clu.selectlist)
        
        @self._clu.connect
        def on_cluster(*args, **kwargs):
            self._draw(self._clu.index_id)
            # self._clu.select_clu(self._clu.index_id)


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
        common_clus = np.intersect1d(current_clus, np.array(list(local_idx.keys())))
        
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

        if self._clu.selectlist.shape[0]==0:
            self._clu.select(global_idx) # , caller=self.__module__
        elif self._clu.selectlist.shape[0]>0:
            global_idx = np.intersect1d(global_idx, self._clu.selectlist)
            self._clu.select(global_idx) # , caller=self.__module__
            

        


    ### ----------------------------------------------
    ###              private method 
    ### ----------------------------------------------

    # def _locate_amplitude(self, clu):
    #     '''
    #         locate the peak of amplitude, return index and peak value
    #     '''
    #     self.times = self._spike_time[self._clu.index[clu]]
    #     # peak always heppenned one offset before
    #     self.amplitudes = self._spk[self._clu.index[clu], :].min(axis=1).min(axis=1) / self._scale
    #     # print amplitudes.shape
    #     # amplitudes = self._spk[self._clu.index[clu], 7].min(axis=1) / self._scale
    #     # print amplitudes.shape
    #     # print self._spk.shape
        
    #     return  self.times / self.binsize, self.amplitudes
 

    def _draw(self, clus, delimit=True):
        '''
            The x pos is time, the y pos is amplitude, and the color and pos is pairwise.
            Draw clu by clu because have to match the color
        '''
        self.poses = None
        self.colors = None
        
        for clu_id in clus:
            clu_idx = self._clu.index[clu_id]
            pos = _locate_amplitude(self._spk, self._spike_time, self.binsize, clu_idx) 
            color = np.tile(np.hstack((palette[clu_id],1)),(pos.shape[0],1))
        
            if self.poses is None and self.colors is None:
                self.poses = pos
                self.colors = color
            else:
                self.poses = np.concatenate((self.poses, pos))
                self.colors = np.concatenate((self.colors, color))

        super(amplitude_view, self).set_data(pos=self.poses, colors=self.colors, delimit=delimit) 


    def on_key_press(self, e):
        '''
            Control: control + mouse wheel to adjust the transparency 
            r:       reset the camera
        '''
        if keys.CONTROL in e.modifiers and not self._control_transparency:
            self._view.events.mouse_wheel.disconnect(self._view.camera
                    .viewbox_mouse_event)
            self._control_transparency = not self._control_transparency 
        
        if e.key.name == 'Escape':
            self._clu.select(np.array([])) 
            self._bursting_time_threshold = 0.4

        elif e.text == 'r':
            self._view.camera.reset()
            self._view.camera.set_range()
        elif e.text == 'c':
            self.x_axis_lock = not self.x_axis_lock 
        elif e.text == 'x':
            self.event.emit('clip', thres=self.amp)
        elif e.text == 'f':
            if len(self._clu.selectlist) > 0:
                self.event.emit('refine', method='time_threshold', args=self._bursting_time_threshold)
                self._bursting_time_threshold /= 2.

        self._key_option = e.key.name


    def on_mouse_release(self, e):
        """
            Control + 1: Rectangle
            Control + 2: Lasso
        """
        if keys.CONTROL in e.modifiers and e.is_dragging:
            if self._key_option in ['1','2']:
                mask = self._picker.pick(self._pos)
                self.select(mask)
                