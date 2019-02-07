import numpy as np
from vispy import scene, app
from vispy.visuals.transforms import STTransform
from vispy.util import keys
from .color_scheme import palette
from ..base.CLU import CLU
from ..utils.utils import Picker
from ..utils import Timer


class pf_view(scene.SceneCanvas):
    
    def __init__(self, pc, show=False, debug=False):
        scene.SceneCanvas.__init__(self, keys=None)

        self.unfreeze()

        self.pc = pc
        self.view = self.central_widget.add_view()
        self.view.camera = 'panzoom'
        self.image = scene.visuals.Image(parent=self.view.scene, method='subdivide', cmap='grays', clim=[0,1])

        self.debug = debug
        if show is True:
            self.show()

    def set_data(self, clu, gtimes):
        self.clu = clu
        self.gtimes = gtimes

    def render(self, spk_times):
        place_field =  self.pc._get_field(spk_times)
        self.image.set_data(place_field/place_field.max())
        self.view.camera.set_range()

    def set_pc(self, pc):
        self.pc = pc

    def register_event(self):
        @self.clu.connect
        def on_select(*args, **kwargs): 
            self.render(self.gtimes[self.clu.selectlist])

    def on_key_press(self, e):
        if e.text == 'r':
            self.view.camera.set_range()

    def run(self):
        self.show()
        self.app.run()
