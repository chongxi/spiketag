import numpy as np
from vispy import scene, app
from .color_scheme import palette
from ..utils.utils import Picker

class scatter_2d_view(scene.SceneCanvas):

    def __init__(self, show=False):
        scene.SceneCanvas.__init__(self, keys=None)

        self.unfreeze()

        self._grid = self.central_widget.add_grid(bgcolor='k', border_color='k', margin=10)       
        self._view = self._grid.add_view(row=0, col=0, border_color='k')
        self._view.camera = 'panzoom'

        self._scatter = scene.visuals.Markers()
        self._view.add(self._scatter)
        
        self._transparency = 1.
        self._marker_size = 2.
        self._highlight_color = np.array([1,0,0,1],dtype='float32')
        self._cache_mask = np.array([])
        self._cache_color = np.array([])

        self._picker = Picker(self.scene, self._scatter.node_transform(self._grid))
        self._key_option = 0

    ### ----------------------------------------------
    ###              public method 
    ### ----------------------------------------------

    @property 
    def transparency(self):
        return self._transparency

    @transparency.setter
    def transparency(self, t):
        self._transparency = t
        if self._transparency >= 0.9:
            self._transparency = 0.9
        elif self._transparency <= 0.001:
            self._transparency = 0.001

        self._colors[:,-1] = self._transparency
        self._colour()

    def set_data(self, pos=None, symbol=None, colors=None):
        self._pos = pos
        self._symbol = symbol
        self._colors = colors
        
        self._render()

    def on_key_press(self, e):
        if e.text == 'r':
            self._view.camera.reset()
            self._view.camera.set_range()
        else:
            self._key_option = e.text
    
    def attach_xaxis(self, axis_color=(0,1,1,0.8)):
        fg = axis_color
        self._xaxis = scene.AxisWidget(orientation='bottom', text_color=fg, axis_color=fg, tick_color=fg)
        self._xaxis.stretch = (0.9, 0.1)
        self._grid.add_widget(self._xaxis, row=1, col=0)

        self._xaxis.link_view(self._view)

    def on_mouse_press(self, e):
        modifiers = e.modifiers
        if modifiers is not ():
            if modifiers[0].name == 'Alt':
                if self._key_option in ['1','2']:
                    self._picker.origin_point(e.pos)

    def on_mouse_move(self, e):
        modifiers = e.modifiers
        if modifiers is not () and e.is_dragging:
            if modifiers[0].name == 'Alt':
                if self._key_option == '1':
                    self._picker.cast_net(e.pos,ptype='rectangle')
                if self._key_option == '2':
                    self._picker.cast_net(e.pos,ptype='lasso')

    def on_mouse_release(self, e):
        modifiers = e.modifiers
        if modifiers is not () and e.is_dragging:
            if modifiers[0].name == 'Alt' and self._key_option in ['1','2']:
                    mask = self._picker.pick(self._pos)
                    self._highlight(mask)
                    self.select(mask)

    def on_mouse_wheel(self, e):
        modifiers = e.modifiers
        if modifiers is not ():
            if modifiers[0].name == 'Control':
                self.transparency *= np.exp(e.delta[1]/4)

    def on_key_release(self, e):
        self._key_option = 0

    ### ----------------------------------------------
    ###              private  method 
    ### -----------------------------------------cc-----
    def _highlight(self, mask, refresh=True):
        if refresh is True and len(self._cache_mask) > 0:
            cache_mask = self._cache_mask 
            self._colors[cache_mask, :] = self._cache_color[cache_mask, :]
            self._colors[cache_mask, -1] = self._transparency
            self._cache_mask = np.array([])

        if len(mask) > 0:
            self._colors[mask, :] = self._highlight_color
            self._cache_mask = np.hstack((self._cache_mask, mask)).astype('int64')
        
        self._colour()

    def _colour(self):
        self._scatter._data['a_fg_color'] = self._colors
        self._scatter._data['a_bg_color'] = self._colors
        self._scatter._vbo.set_data(self._scatter._data)
        self._scatter.update()

    def _render(self):
        self._cache_mask = np.array([])
        self._colors[:,-1] = self._transparency
        self._cache_color = self._colors.copy()
        self._scatter.set_data(self._pos, symbol=self._symbol, size=self._marker_size, edge_color=self._colors, face_color=self._colors)   
        self._view.camera.set_range()



class raster_view(scatter_2d_view):

    def __init__(self):
        super(scatter_2d_view, self).__init__()

        self._symbol = '|'




class amplitude_view(scatter_2d_view):

    def __init__(self, fs=25e3):
        super(amplitude_view, self).__init__()
        super(amplitude_view, self).attach_xaxis()

        self._fs = fs
        self._time_slice = 1 # seconds


    ### ----------------------------------------------
    ###              public method 
    ### ----------------------------------------------

    def bind(self, data, spktag):
        self._data = data
        self._spktag = spktag
        self._scale = data.max() - data.min()

    def set_data(self, ch, spk=None, clu=None):
        self._spike_time = self._get_spike_time(ch)
        self._spk = spk
        self._clu = clu
        
        @self._clu.connect
        def on_select_clu(*args, **kwargs):
            self._draw(self._clu.select_clus)

        @self._clu.connect
        def on_select(*args, **kwargs):
            self.highlight(self._clu.selectlist)
        
        @self._clu.connect
        def on_cluster(*args, **kwargs):
            self._clu.select_clu(self._clu.index_id)

        # draw all clusters when ch settled
        self._clu.select_clu(self._clu.index_id)

    @property
    def sample_rate(self):
        return self._fs * self._time_slice

    def highlight(self, global_idx):
        local_idx = self._clu.global2local(global_idx)
        current_clus = self._clu.select_clus
        common_clus = np.intersect1d(current_clus, np.array(local_idx.keys()))

        view_idx = np.array([],dtype='int64')
        if len(common_clus) > 0:
            for clu in common_clus:
                before = current_clus[np.where(current_clus < clu)]
                for b in before:
                    local_idx[clu] += self._clu.index_count[b]
                view_idx = np.hstack((view_idx, local_idx[clu]))
        
        super(amplitude_view, self)._highlight(view_idx)

    def select(self, view_idx):
        current_clus = self._clu.select_clus
        local_idx = {}
        
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

    def _draw(self, clus):
       
        poses = None
        colors = None
        
        for clu in clus:
            times = self._spike_time[self._clu.index[clu]]
            # peak always heppenned one offset before
            amplitudes = self._spk[self._clu.index[clu], 7, 1] / self._scale
            x, y = times / self.sample_rate, amplitudes
            pos = np.column_stack((x, y))
            color = np.tile(np.hstack((palette[clu],1)),(pos.shape[0],1))
        
            if poses is None and colors is None:
                poses = pos
                colors = color
            else:
                poses = np.concatenate((poses,pos))
                colors = np.concatenate((colors,color))

        super(amplitude_view, self).set_data(pos=poses, symbol='o',colors=colors) 

