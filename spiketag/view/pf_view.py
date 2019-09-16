import numpy as np
from vispy import scene, app
from vispy.visuals.transforms import STTransform
from vispy.util import keys
from .color_scheme import palette
from ..base.CLU import CLU
from ..utils.utils import Picker
from ..utils import Timer


class pf_view(scene.SceneCanvas):
    
    def __init__(self, pc=None, show=False, debug=False, title='place field'):
        scene.SceneCanvas.__init__(self, keys=None, title=title)

        self.unfreeze()

        self.pc = pc
        self.view = self.central_widget.add_view()
        self.view.camera = 'panzoom'
        self.image = scene.visuals.Image(parent=self.view.scene, method='subdivide', cmap='hot', clim=[0.05, 1.05])

        self.debug = debug
        if show is True:
            self.show()

    def set_data(self, clu, gtimes):
        self.clu = clu
        self.gtimes = gtimes

    def _get_field(self, spk_times):
        place_field =  self.pc._get_field(spk_times)
        return place_field

    def _render(self, place_field):
        self.image.set_data(place_field/place_field.max())
        self.view.camera.set_range()

    def set_pc(self, pc):
        self.pc = pc

    def register_event(self):
        if self.pc is not None:
            @self.clu.connect
            def on_select(*args, **kwargs): 
                if len(self.clu.selectlist) > 0:
                    place_field = self._get_field(self.gtimes[self.clu.selectlist])
                    self._render(place_field)
                else:
                    pass

    def on_key_press(self, e):
        if e.text == 'r':
            self.view.camera.set_range()

    def run(self):
        self.show()
        self.app.run()
