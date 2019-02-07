import numpy as np
from vispy import scene, app
from vispy.visuals.transforms import STTransform
from vispy.util import keys
from .color_scheme import palette
from ..base.CLU import CLU
from ..utils.utils import Picker
from ..utils import Timer


class image_view(scene.SceneCanvas):
    
    def __init__(self, show=False, debug=False):
        scene.SceneCanvas.__init__(self, keys=None)

        self.unfreeze()
        self.view = self.central_widget.add_view()
        self.view.camera = 'panzoom'

        self._transparency = 1.
        self.image = scene.visuals.Image(parent=self.view.scene, method='subdivide')

        self.debug = debug
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
        # self.color[:,-1] = self._transparency
        self._update()

    def set_data(self, data):
        self.image.set_data(np.flipud(data))
        self.view.camera.set_range()
        # # scale and center image in canvas
        # s = 700. / max(self.image.size)
        # t = 0.5 * (700. - (self.image.size[0] * s)) + 50
        # self.image.transform = STTransform(scale=(s, s), translate=(t, 50))

    def on_key_press(self, e):
        if e.text == 'r':
            self.view.camera.set_range()

    def run(self):
        self.show()
        self.app.run()



