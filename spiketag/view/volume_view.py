import numpy as np
from vispy import scene, app
from vispy.util import keys
from .color_scheme import palette
# from ..base.CLU import CLU
from ..utils.utils import Picker
from ..utils import Timer
from itertools import cycle
from vispy.color import get_colormaps, BaseColormap


opaque_cmaps = cycle(get_colormaps())
opaque_cmap = next(opaque_cmaps)

class TransGrays(BaseColormap):
    def __init__(self, alpha):
        self.colors = [(1.0, 1.0, 1.0, alpha),]

        self.glsl_map = """
        vec4 translucent_grays(float t) {
            return vec4(t, t, t, $color_0[3]*t);
        }
        """

        self.glsl_map = self.glsl_map.replace('$color_0[3]', str(alpha))


class volume_view(scene.SceneCanvas):
    
    def __init__(self, nvbs=1, show=False, debug=False):
        scene.SceneCanvas.__init__(self, size=(800, 600), keys='interactive')

        self.unfreeze()
        self.grid = self.central_widget.add_grid()
        self.grid.unfreeze()

        self.vbs = {}
        self.vol = {}
        self.data = {}

        self._transparency = 0.03
        self._control_transparency = False
        self.key_option = 0

        for i in range(nvbs):
            self.init_vb(i)
        
        # self._timer = app.Timer(0.0)
        # self._timer.connect(self.on_timer)
        # self._timer.start()

        # TODO: for Picker

        # Add a 3D axis to keep us oriented
        # scene.visuals.XYZAxis(parent=self.view.scene)
        if show is True:
            self.show()

    def __call__(self, nvbs=1):
        self.unfreeze()
        self.grid = self.central_widget.add_grid()
        self.vbs = {}
        self.vol = {}
        self.data = {}
        for i in range(nvbs):
            self.init_vb(i)

        for i in range(nvbs):
            self.init_vb(i)


    def init_vb(self, i):
        self.vbs[i] = self.grid.add_view(name='vb'+str(i), border_color=palette[i])
        self.vbs[i].camera = 'turntable'
        self.vol[i] = scene.visuals.Volume()
        self.vol[i].unfreeze()
        self.vol[i].method = 'translucent'
        self.vol[i].cmap = TransGrays(alpha=self._transparency)
        # self.vol[i].cmap = opaque_cmap
        self.vbs[i].add(self.vol[i])


    def set_data(self, data, vb_id=0, clim=None):
        # volumetric data for example: data.shape = (128,128,128)

        self.data[vb_id] = data
        self.vol[vb_id].set_data(data, clim)
        cam = scene.cameras.TurntableCamera(parent=self.vbs[vb_id].scene, fov=60)
        self.vbs[vb_id].camera = cam


    @property
    def transparency(self):
        return self._transparency

    @transparency.setter
    def transparency(self, v):
        self._transparency = v
        if self._transparency >= 0.99:
            self._transparency = 0.99
        elif self._transparency <= 0.001:
            self._transparency = 0.001

        for _vol in self.vol.values():
            # _vol.method = 'translucent'
            _vol.cmap = TransGrays(alpha=self._transparency)



    # def on_mouse_wheel(self, e):
    #     if keys.CONTROL in e.modifiers:
    #         self.transparency *= np.exp(e.delta[1]/4)

    def on_key_press(self, e):

        # if e.key.name == 'Escape':
        #     self.clu.select(np.array([]))

        if e.text == '/':
            for _vol in self.vol.values():
                if _vol.method == 'translucent':
                    _vol.method = 'mip'
                    self.transparency = 1
                else:
                    _vol.method = 'translucent'
                    self.transparency = 0.03

        if e.text == '=':
            self.transparency += 0.0025

        if e.text == '-':
            self.transparency -= 0.0025

        # if keys.CONTROL in e.modifiers and not self._control_transparency:
        #     self.view.events.mouse_wheel.disconnect(self.view.camera
        #             .viewbox_mouse_event)
        #     self._control_transparency = not self._control_transparency 
        # self.key_option = e.key.name
                # _vol.cmap = TransGrays(alpha=0.01)            

        # if e.text == 'd':
        #     self.mode = 'dimension'
        #     self.dimension = []

        # if self.mode == 'dimension':
        #     if e.text.isdigit() and len(self.dimension)<3:
        #         self.dimension.append(int(e.text))
        #         self.dimension_text.text = str(self.dimension) 
        #         if len(self.dimension)==3:
        #             self.set_dimension(self.dimension)

        # if self.mode != 'dimension':
        #     if e.text == 'e':
        #         self.toggle_noise_clu()


    # def on_key_release(self, e):
    #     if self._control_transparency:
    #         self.view.events.mouse_wheel.connect(self.view.camera.viewbox_mouse_event)
    #         self._control_transparency = not self._control_transparency
    #     self.key_option = 0
