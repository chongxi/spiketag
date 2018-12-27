import numpy as np
from vispy import scene, app
from vispy.util import keys
from .color_scheme import palette
# from ..base.CLU import CLU
from ..utils.utils import Picker
from ..utils import Timer


class volume_view(scene.SceneCanvas):
    
    def __init__(self, show=False, debug=False):
        scene.SceneCanvas.__init__(self, keys=None)

        self.unfreeze()
        self.view = self.central_widget.add_view()
        self.view.camera = 'turntable'

        self.vol = scene.visuals.Volume()
        self.vol.unfreeze()
        self.view.add(self.vol)
        # self._timer = app.Timer(0.0)
        # self._timer.connect(self.on_timer)
        # self._timer.start()

        # TODO: for Picker

        # Add a 3D axis to keep us oriented
        # scene.visuals.XYZAxis(parent=self.view.scene)
        if show is True:
            self.show()



    def set_data(self, data, clim=None):
        # volumetric data for example: data.shape = (128,128,128)

        self.data = data
        self.vol.set_data(data)

