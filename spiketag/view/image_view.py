from vispy import scene, app


class image_view(scene.SceneCanvas):
    
    def __init__(self, **kwargs):
        '''
        title='VisPy canvas',
        size=(800, 600),
        position=None,
        show=False,
        autoswap=True,
        app=None,
        create_native=True,
        vsync=False,
        resizable=True,
        decorate=True,
        fullscreen=False,
        config=None,
        shared=None,
        keys=None,
        parent=None,
        dpi=None,
        always_on_top=False,
        px_scale=1,
        bgcolor='black',
        '''
        scene.SceneCanvas.__init__(self, **kwargs)
        self.unfreeze()
        self.view = self.central_widget.add_view()
        self.view.camera = 'panzoom'
        self.image = scene.visuals.Image(parent=self.view.scene, method='subdivide')
        self.t = 0
        self.evolving = False  # no update via time (no dynamics)
        self._timer = app.Timer('auto', connect=self.on_timer, start=self.evolving)

    def set_data(self, X):
        self.image.set_data(X)
        self.view.camera.set_range()

    def set_func(self, func, t_step=0.03):
        self.t_step = t_step
        self.func = func

    def on_key_press(self, e):
        if e.text == 'r':
            self.view.camera.set_range()
        if e.text == 'g':
            self.evolving_state_flip()

    def evolving_state_flip(self):
        self.evolving = not self.evolving
        self._timer.start() if self.evolving else self._timer.stop()

    def on_timer(self, event):
        self.t += self.t_step
        self.func(self.t)
        self.set_data(self.pixels.to_numpy().T)
        self.update()

    def run(self):
        self.show()
        self.app.run()

