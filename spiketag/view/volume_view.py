import numpy as np
from vispy import scene, app
from vispy.util import keys
from .color_scheme import palette
# from ..base.CLU import CLU
from ..utils.utils import Picker
from ..utils import Timer


class volume_view(scene.SceneCanvas):
    
    def __init__(self, nvbs=1, show=False, debug=False):
        scene.SceneCanvas.__init__(self, size=(800, 600), keys='interactive')

        self.unfreeze()
        self.grid = self.central_widget.add_grid()

        self.vbs = {}
        self.vol = {}
        self.data = {}

        self.init_vb(nvbs)
        
        # self._timer = app.Timer(0.0)
        # self._timer.connect(self.on_timer)
        # self._timer.start()

        # TODO: for Picker

        # Add a 3D axis to keep us oriented
        # scene.visuals.XYZAxis(parent=self.view.scene)
        if show is True:
            self.show()

    def init_vb(self, nvbs):
        for i in range(nvbs):
            self.vbs[i] = self.grid.add_view(name='vb'+str(i), border_color=palette[i])
            self.vbs[i].camera = 'turntable'
        for i in range(nvbs):
            self.vol[i] = scene.visuals.Volume()
            self.vol[i].unfreeze()
            self.vbs[i].add(self.vol[i])


    def set_data(self, data, vb_id=0, clim=None):
        # volumetric data for example: data.shape = (128,128,128)

        self.data[vb_id] = data
        self.vol[vb_id].set_data(data, clim)
        cam = scene.cameras.TurntableCamera(parent=self.vbs[vb_id].scene, fov=60)
        self.vbs[vb_id].camera = cam
