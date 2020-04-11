from vispy import scene, app
import taichi as ti
# from spiketag.view.image_view import image_view

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

# _______________ Taichi Julia Set: start__________________

n = 640
X = ti.var(dt=ti.f32, shape=(n*2, n))

@ti.func
def complex_sqr(z):
    return ti.Vector([z[0] * z[0] - z[1] * z[1], z[1] * z[0] * 2])

@ti.kernel
def paint(t: ti.f32):
    for i, j in X: # Parallized over all pixels
        # c = ti.Vector([-0.8, ti.sin(t) * 0.2])
        c = ti.Vector([0.9*ti.cos(t), 0.9*ti.sin(t)])
        z = ti.Vector([float(i) / n - 1, float(j) / n - 0.5]) * 2
        iterations = 0
        while z.norm() < 20 and iterations < 50:
            z = complex_sqr(z) + c
            iterations += 1
        X[i, j] = 1 - iterations * 0.02

# _______________ Taichi Julia Set: end __________________

if __name__ == '__main__':
    imview = image_view(title='Julia Set', always_on_top=True, size=X.shape())
    # imview.image.cmap='grays' # check vispy.color.get_colormaps()
    # here we set dynamic function and time step for visualization
    imview.pixels = X
    imview.set_func(func=paint, t_step=0.03)
    imview.run()
