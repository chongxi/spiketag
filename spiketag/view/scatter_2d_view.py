import numpy as np
from vispy import scene, app
from ..utils.utils import Picker

class scatter_2d_view(scene.SceneCanvas):
    ''' Basic scatter 2d view, any view need markers can extend this view.
        This view draw the marker use the position by set_data,  and provide basis highlight, picker and transparency feature as well,
        the child class need to implemennt the transform from origin data set to position and transform from position to origin data set.

        Parameters
        ----------
        symbol : str
            the symbol of marker, default is 'o'
        marker_size : float
            the size of marker, default is 2.0
        edge_width : float
            the edge width of symbol outline in pixels
    '''
    def __init__(self, symbol='o', marker_size=2., edge_width=1., show=False):
        scene.SceneCanvas.__init__(self, keys=None)

        self.unfreeze()

        self._grid = self.central_widget.add_grid(bgcolor='k', border_color='k', margin=10)       
        self._view = self._grid.add_view(row=0, col=0, border_color='k')
        self._view.camera = 'panzoom'

        self._scatter = scene.visuals.Markers()
        self._view.add(self._scatter)
        
        self._symbol = symbol
        self._edge_width = edge_width
        self._transparency = 1.
        self._control_transparency = False
        self._marker_size = marker_size
        self._highlight_color = np.array([1,0,0,1],dtype='float32')
        self._cache_mask = np.array([])
        self._cache_color = np.array([])

        self._transform2view = self._scatter.node_transform(self._grid)
        self._picker = Picker(self.scene, self._transform2view)
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

    def set_data(self, pos=None, colors=None):
        self._pos = pos
        self._colors = colors
        
        self._render()
    
    def attach_xaxis(self, axis_color=(0,1,1,0.8)):
        '''
            Provide x axis. This method is optional for child class.
        '''
        fg = axis_color
        self._xaxis = scene.AxisWidget(orientation='bottom', text_color=fg, axis_color=fg, tick_color=fg)
        self._xaxis.stretch = (0.9, 0.1)
        self._grid.add_widget(self._xaxis, row=1, col=0)

        self._xaxis.link_view(self._view)

    '''
        Child class can override this function, to recieve datas which picked after picker operating. But the position is view coordinate, 
        child class should transform to their coordinate.
        
        Parameters
        ---------
            view_idx: array-like
                the position within the view of selected datas.
    '''
    def select(self, view_idx):
        pass

       
    def on_key_press(self, e):
        '''
            Control: control + mouse wheel to adjust the transparency 
            r:       reset the camera
        '''
        modifiers = e.modifiers 
        if modifiers is not ():
            if modifiers[0].name == 'Control' and not self._control_transparency:
                self._view.events.mouse_wheel.disconnect(self._view.camera
                        .viewbox_mouse_event)
                self._control_transparency = not self._control_transparency 
        elif e.text == 'r':
            self._view.camera.reset()
            self._view.camera.set_range()
        
        self._key_option = e.text
    
    def on_mouse_press(self, e):
        """
            Alt + 1: Rectangle
            Alt + 2: Lasso
        """
        modifiers = e.modifiers
        if modifiers is not ():
            if modifiers[0].name == 'Alt':
                if self._key_option in ['1','2']:
                    self._picker.origin_point(e.pos)

    def on_mouse_move(self, e):
        """
            Alt + 1: Rectangle
            Alt + 2: Lasso
            Control: Highlight nearest spiks
        """
        modifiers = e.modifiers
        if modifiers is not () and e.is_dragging:
            if modifiers[0].name == 'Alt':
                if self._key_option == '1':
                    self._picker.cast_net(e.pos,ptype='rectangle')
                if self._key_option == '2':
                    self._picker.cast_net(e.pos,ptype='lasso')
            if modifiers[0].name == 'Control':
                mask = self._get_nearest_spikes(e.pos)
                self._highlight(mask)
                self.select(mask)

    def on_mouse_release(self, e):
        """
            Alt + 1: Rectangle
            Alt + 2: Lasso
        """
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
        '''
            Reconnect the mouse event to camera and record the key option which used by picker.
        '''
        if self._control_transparency:
            self._view.events.mouse_wheel.connect(self._view.camera.viewbox_mouse_event)
            self._control_transparency = not self._control_transparency
        self._key_option = 0

    ### ----------------------------------------------
    ###              private  method 
    ### ----------------------------------------------
    def _get_nearest_spikes(self, mouse_pos):
        radius = (self._xaxis.axis.domain[1] - self._xaxis.axis.domain[0]) / 2e2 
        range = (mouse_pos[0] - radius, mouse_pos[0] + radius)
        x_pos = self._transform2view.map(self._pos)[:, 0]
        return np.where((x_pos >= range[0]) & (x_pos < range[1]))[0]


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
        self._scatter.set_data(self._pos, symbol=self._symbol, size=self._marker_size, edge_color=self._colors, face_color=self._colors, edge_width=self._edge_width)   
        self._view.camera.set_range()

