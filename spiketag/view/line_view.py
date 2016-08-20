from vispy import scene, app
import numpy as np


class line_view(scene.SceneCanvas):
    def __init__(self, connect='strip', method='gl'):
        '''
        Parameters
        ----------
        connect : str or array
            Determines which vertices are connected by lines.

                * "strip" causes the line to be drawn with each vertex
                  connected to the next.
                * "segments" causes each pair of vertices to draw an
                  independent line segment
                * numpy arrays specify the exact set of segment pairs to
                  connect.  
        method : str
            Mode to use for drawing.

                * "agg" uses anti-grain geometry to draw nicely antialiased lines
                  with proper joins and endcaps.
                * "gl" uses OpenGL's built-in line rendering. This is much faster,
                  but produces much lower-quality results and is not guaranteed to
                  obey the requested line width or join/endcap styles.
        '''

        scene.SceneCanvas.__init__(self, keys=None)
        self.unfreeze()

        self.grid = self.central_widget.add_grid(spacing=0, bgcolor='k',
                                                  border_color='k')       
        self.view = self.grid.add_view(row=2, col=3, border_color='k')
        
        # self.view = self.central_widget.add_view()
        self.view.camera = 'panzoom'
        self.lines = []
        self.line = scene.visuals.Line(connect=connect, method=method)
        self.view.add(self.line)
        self.add_axis()
        self.freeze()


    def add_axis(self, axis_color=(0,1,1,0.8)):

        fg = axis_color

        self.unfreeze()
        self.yaxis = scene.AxisWidget(orientation='left', text_color=fg,
                                      axis_color=fg, tick_color=fg)
        self.yaxis.stretch = (0.015, 0.05)
        self.grid.add_widget(self.yaxis, row=2, col=2)

        self.ylabel = scene.Label("", rotation=-90, color=fg)
        self.ylabel.stretch = (0.01, 0.01)
        self.grid.add_widget(self.ylabel, row=2, col=1)

        self.xaxis = scene.AxisWidget(orientation='bottom', text_color=fg,
                                      axis_color=fg, tick_color=fg)
        self.xaxis.stretch = (0.05, 0.025)
        self.grid.add_widget(self.xaxis, row=3, col=3)

        self.xlabel = scene.Label("")
        self.xlabel.stretch = (0.01, 0.01)
        self.grid.add_widget(self.xlabel, row=4, col=3)

        self.view.camera = 'panzoom'
        self.camera = self.view.camera

        self.xaxis.link_view(self.view)
        self.yaxis.link_view(self.view)

        self.freeze()


    def set_data(self, x, y, **kwargs):
        '''
        pos : array
            Array of shape (..., 2) or (..., 3) specifying vertex coordinates.
        color : Color, tuple, or array
            The color to use when drawing the line. If an array is given, it
            must be of shape (..., 4) and provide one rgba color per vertex.
            Can also be a colormap name, or appropriate `Function`.
        width:
            The width of the line in px. Line widths > 1px are only
            guaranteed to work when using 'agg' method.
        '''
        # current_palette = sns.color_palette()
        pos = np.vstack((x,y))
        if pos.shape[1] != 2:
            pos = pos.T

        line = scene.visuals.Line(connect='strip', method='gl')
        line.set_data(pos=pos, **kwargs)
        self.lines.append(line)
        self.view.add(line)
        # self.line.set_data(pos=pos, **kwargs)
        # recompute the bounds
        self.view.camera.set_range(x=line._compute_bounds(0, self.view),
                                   y=line._compute_bounds(1, self.view))
        # set the new bounds as default
        self.view.camera.set_default_state()
        self.view.camera.reset()


    def attach(self, gui):
        self.unfreeze()
        gui.add_view(self)


    def on_key_press(self, e):
        if e.text=='r':
            self.view.camera.reset()
