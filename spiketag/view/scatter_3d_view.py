import numpy as np
from vispy import scene, app
from vispy.util import keys
from .color_scheme import palette
from ..base.CLU import CLU
from ..utils.utils import Picker
from ..utils import Timer


class scatter_3d_view(scene.SceneCanvas):
    
    def __init__(self, show=False, debug=False):
        scene.SceneCanvas.__init__(self, keys=None)

        self.unfreeze()
        self.view = self.central_widget.add_view()
        self.view.camera = 'turntable'

        self._n = 0
        self._transparency = 0.7
        self._control_transparency = False
        self._control_picker = False
        self._size = 3
        self._highlight_color = np.array([1,0,0,1]).astype('float32')
        self.color = np.array([])
        self._cache_mask_ = np.array([])
        self._cache_color = np.array([])

        self.scatter = scene.visuals.Markers()
        self.view.add(self.scatter)
        self._timer = app.Timer(0.0)
        self._timer.connect(self.on_timer)
        # self._timer.start()

        # for Picker
        self._picker = Picker(self.scene,self.scatter.node_transform(self.view))
        self.key_option = 0

        self._noise_toggle = False
        self.mode = ''
        self.dimension = None
        self.dimension_text = scene.visuals.Text(parent=self.scene)
        self.debug = debug
        # Add a 3D axis to keep us oriented
        scene.visuals.XYZAxis(parent=self.view.scene)
        if show is True:
            self.show()

    @property
    def transparency(self):
        return self._transparency

    @transparency.setter
    def transparency(self, v):
        self._transparency = v
        if self._transparency >= 0.9:
            self._transparency = 0.9
        elif self._transparency <= 0.001:
            self._transparency = 0.001
        self.color[:,-1] = self._transparency
        self._update()

    def _update(self):
        with Timer('update transperency: ', verbose=self.debug):
            self.scatter._data['a_fg_color'] = self.color
            self.scatter._data['a_bg_color'] = self.color
            self.scatter._vbo.set_data(self.scatter._data)
            self.scatter.update()

    def set_data(self, fet, clu=None, rho=None):
        # only run in the beginning, init the clu
        # and connect clu change to _render()

        self.fet = fet

        if clu is None:
            self.clu = CLU(np.zeros(fet.shape[0],).astype(np.int64))
        elif type(clu) is np.ndarray:
            self.clu = CLU(clu)
        else:
            self.clu = clu  # CLU type

        self.rho = rho

        self._n = fet.shape[0]
        self._render()

        @self.clu.connect
        def on_cluster(*args, **kwargs):
            self._render()

        @self.clu.connect
        def on_select(*args, **kwargs):
            self.highlight(self.clu.selectlist)


    def set_dimension(self, dimension):
        self.dimension = dimension
        self._render()
        self.mode = ''

    def _render(self):
        #######################################################
        ### step1: set the color for clustering
        base_color = np.asarray([palette[i] for i in self.clu.membership])
        _transparency = np.ones((len(self.fet), 1)) * self._transparency
        edge_color = np.hstack((base_color, _transparency))

        #######################################################
        ### step2: set transparency for density
        if self.rho is None:
            _transparency = np.ones((len(self.fet), 1)) * self._transparency
            edge_color = np.hstack((base_color, _transparency))
        else:
            _transparency = self.rho.reshape(-1, 1)
            edge_color = np.hstack((base_color, _transparency))

        #######################################################
        ### step3: prepare and render           
        self.color = edge_color
        self._cache_mask_ = np.array([])
        self._cache_color = self.color.copy()
        # TODO: add functionality to change :3 to input specific 3 dims
        if self.dimension is None:
            self.dimension = [0,1,2]
        self.scatter.set_data(self.fet[:, self.dimension], size=self._size, edge_color=self.color, face_color=self.color)

        self.dimension_text.text = str(self.dimension) 
        self.dimension_text.pos  = np.array([20,10])
        self.dimension_text.color = (1,1,1,0.5) 
        self.dimension_text.font_size = 5 

    def _stream_in_data(self, fet, clu=None):
        stream_size = fet.shape[0]
        self.fet = np.roll(self.fet, -stream_size, axis=0)
        self.fet[-stream_size:] = fet
        self.clu.membership = np.roll(self.clu.membership, -stream_size)
        self.clu.membership[-stream_size:] = clu
        self.clu.__construct__()

    def _stream_in_render(self, fet, clu=None, rho=None, highlight_no=None):
        #######################################################
        ### step0: roll the previous data for stream_size
        stream_size = fet.shape[0]
        self.scatter._data = np.roll(self.scatter._data, -stream_size)

        #######################################################
        ### step1: set the color for clustering
        _base_color = np.asarray([palette[i] for i in clu])
        # _transparency = np.ones((stream_size, 1)) # * self._transparency
        # _edge_color = np.hstack((_base_color, _transparency))

        #######################################################
        ### step2: set transparency for density
        if rho is None:
            _transparency = np.ones((stream_size, 1)) * self._transparency
            _edge_color = np.hstack((_base_color, _transparency))
        else:
            _transparency = np.ones((stream_size, 1)) * rho
            _edge_color = np.hstack((_base_color, _transparency))

        if highlight_no is not None:
            _edge_color[-highlight_no:, -1] = 1.

        #######################################################
        ### step3: update scatter._data for the latest stream_size

        self.scatter._data[-stream_size:]['a_position'] = fet[:,:3]
        self.scatter._data[-stream_size:]['a_fg_color'] = _edge_color
        self.scatter._data[-stream_size:]['a_bg_color'] = _edge_color

        self.scatter._vbo.set_data(self.scatter._data)
        self.scatter.update() 

    def stream_in(self, fet, clu=None, rho=None, highlight_no=None):
        '''
        stream new data into previous data
        but the total number of data is fixed through self._n

        data = np.zeros(n, dtype=[('a_position', np.float32, 3),
                                  ('a_fg_color', np.float32, 4),
                                  ('a_bg_color', np.float32, 4),
                                  ('a_size', np.float32, 1),
                                  ('a_edgewidth', np.float32, 1)])
        '''
        stream_size = fet.shape[0]
        with Timer('stream_in {0} points'.format(stream_size), verbose=self.debug):
            # update self.fet and self.clu
            self._stream_in_data(fet, clu)
            # update self.scatter._data which is used for rendering(bind to scatter._vbo)
            self._stream_in_render(fet, clu, rho, highlight_no)


    def set_range(self):
        self.view.camera.set_range()

    def attach(self, gui):
        self.unfreeze()
        gui.add_view(self)

    def highlight(self, mask, refresh=True):
        """
        highlight the nth index points (mask) with _highlight_color
        refresh is False means it will append
        """
        if refresh is True and len(self._cache_mask_)>0:
            _cache_mask_ = self._cache_mask_
            self.color[_cache_mask_, :] = self._cache_color[_cache_mask_, :]
            self.color[_cache_mask_,-1] = self._transparency
            self._cache_mask_ = np.array([])

        if len(mask)>0:
            self.color[mask, :] = self._highlight_color
            self.color[mask,-1] = 1
            self._cache_mask_ = np.hstack((self._cache_mask_, mask)).astype('int64')

        self._update()

    # ---------------------------------
    def on_timer(self, event):
        # mask = np.random.choice(self._n, np.floor(self._n/100), replace=False)
        # self.highlight(mask, refresh=True)
        with Timer('on_timer', verbose=self.debug):
            n = 10
            new_fet = np.random.randn(n,9) + np.ones((n,9))*5
            new_clu = np.ones((n,)).astype(np.int)
            self.stream_in(new_fet, new_clu)
            # if self._n != 0:
            #     fet = np.random.randn(self._n, 3)
            #     clu = np.zeros((self._n,)).astype(np.int)
            #     self.set_data(fet, clu)

    def on_mouse_wheel(self, e):
        if keys.CONTROL in e.modifiers:
            self.transparency *= np.exp(e.delta[1]/4)

    """
      all of method follows is used for picker
      Control + 1 Rectangle
      Control + 2 Lasso
    """
    def on_mouse_press(self, e):
        if keys.CONTROL in e.modifiers:
            if self.key_option in ['1','2']:
                self._picker.origin_point(e.pos)


    def on_mouse_move(self, e):
        if keys.CONTROL in e.modifiers and e.is_dragging:
            if self.key_option == '1':
                self._picker.cast_net(e.pos,ptype='rectangle')
            if self.key_option == '2':
                self._picker.cast_net(e.pos,ptype='lasso')


    def on_mouse_release(self,e):
        if keys.CONTROL in e.modifiers and e.is_dragging:
            if self.key_option in ['1','2']:
                if self.clu.selectlist.shape[0]==0:
                    mask = self._picker.pick(self.fet[:, self.dimension])
                elif self.clu.selectlist.shape[0]>0:
                    mask = self._picker.pick(self.fet[:, self.dimension])
                    mask = np.intersect1d(mask, self.clu.selectlist)
                self.highlight(mask)
                self.clu.select(mask)


    def toggle_noise_clu(self):
        if self._noise_toggle == False:
            self.color[self.clu[0],-1] = 0.02
            self._update()
            self._noise_toggle = True
        elif self._noise_toggle == True:
            self.color[self.clu[0],-1] = self._transparency
            self._update()
            self._noise_toggle = False


    def on_key_press(self, e):

        if e.key.name == 'Escape':
            self.clu.select(np.array([]))
            
        if e.text == 'd':
            self.mode = 'dimension'
            self.dimension = []

        if self.mode == 'dimension':
            if e.text.isdigit() and len(self.dimension)<3:
                self.dimension.append(int(e.text))
                self.dimension_text.text = str(self.dimension) 
                if len(self.dimension)==3:
                    self.set_dimension(self.dimension)

        if self.mode != 'dimension':
            if e.text == 'e':
                self.toggle_noise_clu()




        if keys.CONTROL in e.modifiers and not self._control_transparency:
            self.view.events.mouse_wheel.disconnect(self.view.camera
                    .viewbox_mouse_event)
            self._control_transparency = not self._control_transparency 
        self.key_option = e.key.name


    def on_key_release(self, e):
        if self._control_transparency:
            self.view.events.mouse_wheel.connect(self.view.camera.viewbox_mouse_event)
            self._control_transparency = not self._control_transparency
        self.key_option = 0
