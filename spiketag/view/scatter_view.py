import numpy as np
from vispy import scene, app
from .color_scheme import palette
from ..base.CLU import CLU
from matplotlib import path
from ..utils.utils import Picker



class scatter_3d_view(scene.SceneCanvas):
    def __init__(self, show=False):
        scene.SceneCanvas.__init__(self, keys=None)

        self.unfreeze()
        self.view = self.central_widget.add_view()
        self.view.camera = 'turntable'

        self._n = 0
        self._transparency = 0.3
        self._size = 3
        self._highlight_color = np.array([1,0,0,1]).astype('float32')
        self.color = np.array([])
        self._cache_mask_ = np.array([])
        self._cache_color = np.array([])

        self.scatter = scene.visuals.Markers()
        self.view.add(self.scatter)
        self._timer = app.Timer(0.5 / 60)
        self._timer.connect(self.on_timer)
        # self._timer.start()

        # for Picker
        self._picker = Picker(self.scene,self.view,self.scatter)
        self.key_option = 0

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
        self.scatter._data['a_fg_color'] = self.color
        self.scatter._data['a_bg_color'] = self.color
        self.scatter._vbo.set_data(self.scatter._data)
        self.scatter.update()
        
    def set_data(self, fet, clu=None, rho=None):
        # only run in the beginning, init the clu
        # and connect clu change to _render()

        self.fet = fet

        if clu is None:
            base_color = np.ones((len(fet), 3))
        elif type(clu) is np.ndarray:
            self.clu = CLU(clu)
        else:
            self.clu = clu

        self.rho = rho

        self._n = fet.shape[0]
        self._render()

        @self.clu.connect
        def on_cluster(*args, **kwargs):
            self._render()

        @self.clu.connect
        def on_select(*args, **kwargs):
            self.highlight(self.clu.selectlist)

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
        self.scatter.set_data(self.fet[:, :3], size=self._size, edge_color=self.color, face_color=self.color)

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
        mask = np.random.choice(self._n, np.floor(self._n/100), replace=False)
        self.highlight(mask, refresh=True)

    def on_mouse_wheel(self, e):
        modifiers = e.modifiers
        if modifiers is not ():
            if modifiers[0].name == 'Control':
                self.transparency *= np.exp(e.delta[1]/4)

    """
      all of method follows is used for picker
      alt + 1 Rectangle
      alt + 2 Lasso
    """
    def on_mouse_press(self,e):
        modifiers = e.modifiers
        if modifiers is not ():
            if modifiers[0].name == 'Alt':
                if self.key_option in ['1','2']:
                    self._picker.origin_point(e.pos)



    def on_mouse_move(self,e):
        modifiers = e.modifiers
        if modifiers is not () and e.is_dragging:
            if modifiers[0].name == 'Alt':
                if self.key_option == '1':
                    self._picker.cast_net(e.pos,ptype='rectangle')
                if self.key_option == '2':
                    self._picker.cast_net(e.pos,ptype='lasso')

    def on_mouse_release(self,e):
        modifiers = e.modifiers
        if modifiers is not () and e.is_dragging:
            if modifiers[0].name == 'Alt' and self.key_option in ['1','2']:
                    mask = self._picker.pick(self.fet[:, :3])
                    self.highlight(mask)
                    self.clu.select(mask)

    def on_key_press(self,e):
        self.key_option = e.text

    def on_key_release(self,e):
        self.key_option = 0
