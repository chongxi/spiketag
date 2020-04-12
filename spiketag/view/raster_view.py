import numpy as np
from ..base.CLU import CLU
from ..view import Picker, ROI_time_series
from .color_scheme import palette
from .scatter_2d_view import scatter_2d_view
from vispy import scene, app, visuals
from numba import njit, prange
from vispy.util import keys


@njit(cache=True, parallel=True, fastmath=True)
def get_population_firing_count(spike_times, fs, t_window=5e-3):
    '''
    calculate population_firing_rate
    '''
    _spk_times = spike_times/fs
    ts  = np.arange(_spk_times[0]+t_window/2, _spk_times[-1]-t_window/2, t_window) 
    firing_count = np.zeros_like(ts)
    for i in prange(ts.shape[0]):
        firing_count[i] = np.sum(np.logical_and(_spk_times >= ts[i]-t_window/2,
                                                _spk_times <  ts[i]+t_window/2))
    pfr = np.zeros((ts.shape[0], 2), np.float32)
    pfr[:, 0] = ts
    pfr[:, 1] = firing_count
    return pfr 


class raster_view(scatter_2d_view):

    def __init__(self, fs=25e3, n_units=None, time_tick=1, population_firing_count_ON=True, t_window=5e-3, view_window=10):
        super(raster_view, self).__init__(symbol='|', marker_size=6., edge_width=1e-3, second_view=population_firing_count_ON)
        super(raster_view, self).attach_xaxis()
        self._time_tick = time_tick 
        self._fs = fs
        self._n_units = n_units
        self._view_window = view_window
        self._second_view = population_firing_count_ON
        if self._second_view:
            self.attach_yaxis()
            self._t_window = t_window
        self.roi = ROI_time_series(self.scene, self._view, self._transform2view)
        self.key_option = 0
        self._control_transparency = False




    ### ----------------------------------------------
    ###              public method 
    ### ----------------------------------------------

    def set_data(self, spkid_matrix):
        '''
        spkid_matrix: n*2 matrix, n spikes, first column is #sample, second column is the spike id
        '''
        self._spike_time = spkid_matrix[:,0] 
        self._spike_id   = spkid_matrix[:,1]
        self._spike_count_clu = np.bincount(np.sort(self._spike_id)).cumsum()
        if self._n_units is None: # if not given from user (user can read from fpga.n_units), will use what's there in the data
            self._n_units = len(np.unique(self._spike_id)) + 1
            print('load {} units'.format(self._n_units))
        # self._clu = CLU(spkid_matrix[:,1].astype(np.int64))

        if self._second_view:
            self._pfr = get_population_firing_count(self._spike_time, self._fs, self._t_window)
            self._draw(self._pfr)
        else:
            self._draw()
        self.set_range()


    def attach_yaxis(self, axis_color=(0,1,1,0.8)):
        '''
            Provide y axis for the population rate
        '''
        fg = axis_color
        # text show amplitude
        self.amp_text = scene.Text("", pos=(0, 0), italic=False, bold=False, anchor_x='left', anchor_y='center',
                                       color=axis_color, font_size=10, parent=self._view2)
        self.amp_text.pos  = (0, 20)

        # x axis shows time and can be moved horizontally for clipping
        self._yaxis = scene.AxisWidget(orientation='left', text_color=fg, axis_color=fg, tick_color=fg)
        self._yaxis.stretch = (0, 1)
        self._grid.add_widget(self._yaxis, row=10, col=0, row_span=3)
        self._yaxis.link_view(self._view2)


    @property
    def binsize(self):
        return int(self._fs * self._time_tick)


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
        
        super(raster_view, self)._highlight(view_idx)

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
        self._clu.select(global_idx, caller=self.__module__)

 
    '''
      all of method follows is used for picker
      Control + 1 Rectangle
      Control + 2 Lasso
    '''
    def on_mouse_press(self, e):
        if keys.CONTROL in e.modifiers:
            if self.key_option in ['1','2']:
                self.roi.origin_point(e.pos)


    def on_mouse_move(self, e):
        if keys.CONTROL in e.modifiers and e.is_dragging:
            if self.key_option == '1':
                self.roi.cast_net(e.pos,ptype='rectangle')
            if self.key_option == '2':
                self.roi.cast_net(e.pos,ptype='lasso')


    def on_mouse_release(self,e):
        if keys.CONTROL in e.modifiers and e.is_dragging:
            if self.key_option in ['1','2']:
                self._selected_id = self.roi.pick(self._pos) # id ordered first by #neuron, then by #spike
                self._highlight(self._selected_id) # test shows this works interactively in notebook
                self.selected = self._to_spike_dict(self._selected_id)


    ### ----------------------------------------------
    ###              private method 
    ### ----------------------------------------------

    def _to_spike_dict(self, _selected_id):
        '''
        N total spikes becomes a (N,2) matrix for rendering at the (x,y) position
        self._pos is this (N,2) matrix, where the first column is spike time and second column is transformed spike id for rendering
        self._pos is rendered as a 2d scatter object
        It is ordered (N) first by #neuron, then by #spike 
        input _selected_id is the 1D index of the order 

        This function
        convert the _selected_id, which is ordered first by #neuron, then by #spike (for fast rendering)
        to      the spike_dict, which is (K,2) (K selected spikes: spike time, spike id) matrix ordered by spike time
        '''
        _spike_id   = np.unique(self._spike_id)[np.searchsorted(np.unique(self._pos[:,1]), self._pos[_selected_id][:,1])]
        _spike_time = self._pos[_selected_id][:,0]  # spike time is in the first column

        _spike_matrix = np.vstack((_spike_time, _spike_id)).T
        _spike_matrix = _spike_matrix[np.argsort(_spike_time)]  # ordered by spike time 
        return _spike_matrix


    def _draw(self, pfr=None, delimit=True):
       
        poses = None
        colors = None
        self._y_bound = (0., 10.)
        span = self._y_bound[1] / self._n_units #len(self._clu.index_id)

        for spk_id in range(self._n_units):
            times = self._spike_time[self._spike_id==spk_id]
            x, y = times / self.binsize, np.full(times.shape, spk_id * span)
            pos = np.column_stack((x,y))
            color = np.tile(np.hstack((palette[spk_id],1)),(pos.shape[0],1))

            if poses is None and colors is None:
                poses = pos
                colors = color
            else:
                poses = np.concatenate((poses, pos))
                colors = np.concatenate((colors, color))

        super(raster_view, self).set_data(pos=poses, colors=colors, delimit=delimit)   # goes to self._pos

        if self._second_view:
            # pfr is population firing rate
            self._line.set_data(pfr, symbol='o', color='w', edge_color='w',
                                     marker_size=5, face_color=(0.2, 0.2, 1))
            # self._view2.camera.set_range()


    def on_key_press(self, e):
        '''
            r:       reset the camera
        '''
        if e.text == 'r':
            self.set_range()

        if keys.CONTROL in e.modifiers and not self._control_transparency:
            self._view.events.mouse_wheel.disconnect(self._view.camera
                    .viewbox_mouse_event)
            self._control_transparency = not self._control_transparency 

        self.key_option = e.key.name


    def on_key_release(self, e):
        if self._control_transparency:
            self._view.events.mouse_wheel.connect(self._view.camera.viewbox_mouse_event)
            self._control_transparency = not self._control_transparency
        self.key_option = 0


    def set_range(self):
        self._view.camera.set_range()
        if self._second_view:
            self._view2.camera.set_range()


    def fromfile(self, filename='./fet.bin'):
        '''
        load and interact with spike rasters
        filename: the file that contains BMI feature-spike packet
        '''
        fet_packet = np.memmap(filename, dtype=np.int32).reshape(-1,7)
        spkid_packet = fet_packet[:, [0,-1]]
        spkid_packet = np.delete(spkid_packet, np.where(spkid_packet[:,1]==0), axis=0)
        self.set_data(spkid_packet)
        self.set_range()


    def update_fromfile(self, filename='./fet.bin', last_N=8000):
        '''
        filename:    the file that contains BMI feature-spike packet
        last_N:      only set_data for the last_N spikes in the file
        view_window: x second for visualization
        '''
        try:
            fet_packet = np.memmap(filename, dtype=np.int32).reshape(-1,7)
            # print(fet_packet.shape)
            N = last_N
            if fet_packet.shape[0]>N:
                spkid_packet = fet_packet[-N:, [0,-1]]
                spkid_packet = np.delete(spkid_packet, np.where(spkid_packet[:,1]==0), axis=0) 
            else:
                spkid_packet = fet_packet[:, [0,-1]]
                spkid_packet = np.delete(spkid_packet, np.where(spkid_packet[:,1]==0), axis=0)                 
            self.set_data(spkid_packet)
            xmin = (spkid_packet[-1, 0]-self._view_window*self._fs)/self._fs
            xmax = spkid_packet[-1, 0]/self._fs
            self._view.camera.set_range(x=(xmin, xmax),  y=self._y_bound)
            self._view2.camera.set_range(x=(xmin, xmax), y=(0, 3+int(self._pfr[:,1].max())))
        except:
            pass
