import numpy as np
import seaborn as sns
from vispy import scene, app
# from xview import scatter_3d_view
from .MyWaveVisual import MyWaveVisual
from .color_scheme import palette


class Axis(scene.AxisWidget):
    """from scene.AxisWidget"""
    def set_x_transform(self, x_transform):
        self.glpos_to_time = x_transform

    def set_y_transform(self, y_transform):
        self.glpos_to_value = y_transform

    def glpos_to_time(self, gl_pos):
        '''
        get time from the x postion of y_axis
        Important: check the affine transformation in MyWaveVisual!
        '''
        xn = np.ceil((gl_pos / 0.95 + 1) * (self.npts - 1) / 2)
        t = (xn + self._time_slice * self._time_span) / self.fs
        return t

    def glpos_to_value(self, gl_pos):
        '''
        get value from the y postion of y_axis
        Important: check the affine transformation in MyWaveVisual!
        '''
        y = (gl_pos+1-1./self.nCh) / 0.95 * self.nCh
        y = y*self.yscale
        return y
    
    def _view_changed(self, event=None):
        """Linked view transform has changed; update ticks.
        """
        tr = self.node_transform(self._linked_view.scene)
        p1, p2 = tr.map(self._axis_ends())
        if self.orientation in ('left', 'right'):
            # yaxis
            self.axis.domain = (self.glpos_to_value(p1[1]), self.glpos_to_value(p2[1]))
            # self.axis.domain = (p1[1],p2[1])
        else:
            # xaxis
            self.axis.domain = (self.glpos_to_time(p1[0]), self.glpos_to_time(p2[0]))


class Cross(object):
    def __init__(self, cursor_color):

        self.cross_state = False
        self.cursor_color = cursor_color
        self._time_slice = 0
        self.x_axis = Axis(orientation='bottom', text_color=cursor_color, tick_color=cursor_color, axis_color=cursor_color)
        self.x_axis.stretch = (1, 0.1)
        self.y_axis = Axis(orientation='left', text_color=(1, 1, 1, 0), tick_color=cursor_color, axis_color=cursor_color)
        self.y_axis.stretch = (0, 1)
        self.y_axis_ref = Axis(orientation='left', text_color=(1, 1, 1, 0), tick_color=(1, 1, 1, 0), axis_color=(0, 1, 1))
        self.y_axis_ref.stretch = (0, 1)

    def set_params(self, nCh, npts, fs, time_slice, time_span, yscale=1):
        self.x_axis.unfreeze()
        self.x_axis.npts = npts
        self.x_axis.fs = fs
        self.x_axis._time_slice = 0
        self.x_axis._time_span = npts
        self.x_axis.freeze()

        self.y_axis.unfreeze()
        self.y_axis.nCh = nCh
        self.y_axis.npts = npts
        self.y_axis.fs = fs
        self.y_axis._time_slice = 0
        self.y_axis._time_span = npts
        self.y_axis.yscale = yscale
        self.y_axis.freeze()

        self.y_axis_ref.unfreeze()
        self.y_axis_ref.nCh = nCh
        self.y_axis_ref.npts = npts
        self.y_axis_ref.fs = fs
        self.y_axis_ref._time_slice = 0
        self.y_axis_ref._time_span = npts
        self.y_axis_ref.yscale = yscale
        self.y_axis_ref.freeze()

    def attach(self, parent):
        parent.add_widget(self.x_axis)
        parent.add_widget(self.y_axis)
        parent.add_widget(self.y_axis_ref)

    def enable_tick(self, axis=1):
        if axis==1:
            self.y_axis.axis._text.color = self.cursor_color
            # print self.y_axis.orientation

    def disable_tick(self, axis=1):
        if axis==1:
            self.y_axis.axis._text.color = (1, 1, 1, 0)      

    def link_view(self, view):
        self.x_axis.link_view(view)
        self.y_axis.link_view(view)
        self.y_axis_ref.link_view(view)
        self.y_axis_ref.visible = False
        self.parentview = view

    def moveto(self, pos):
        pos = pos - self.parentview.pos - self.parentview.margin
        self.x_axis.transform.translate = (0, pos[1])
        self.y_axis.transform.translate = (pos[0], 0)

    def flip_state(self):
        self.cross_state = not self.cross_state

    def ref_enable(self, pos):
        pos = pos - self.parentview.pos - self.parentview.margin
        self.y_axis_ref.transform.translate = (pos[0], 0)
        self.y_axis_ref.visible = True

    def ref_disable(self):
        self.y_axis_ref.visible = False

    @property
    def time_slice(self):
        return self._time_slice

    @time_slice.setter
    def time_slice(self, time_slice_no):
        self._time_slice = time_slice_no
        self.x_axis._time_slice = time_slice_no
        self.x_axis._view_changed()
        self.y_axis._time_slice = time_slice_no
        self.y_axis._view_changed()




class wave_view(scene.SceneCanvas):
    def __init__(self, data, color=None, fs=25000.0, ncols=1, gap_value=0.5*0.95):
        scene.SceneCanvas.__init__(self, keys=None)
        self.unfreeze()
        self.grid1 = self.central_widget.add_grid(spacing=0, bgcolor='gray',
                                                 border_color='k')
        self.view1 = self.grid1.add_view(row=0, col=0, col_span=1, margin=10, bgcolor=(0, 0, 0, 1),
                              border_color=(1, 0, 0))
        self.view1.camera = scene.cameras.PanZoomCamera()
        self.view1.camera.set_range()
        self.view1.camera.interactive = False
        self.view2 = self.grid1.add_view(row=0, col=1, col_span=8, margin=10, bgcolor=(0, 0, 0, 1),
                              border_color=(0, 1, 0))

        self.view2.camera = scene.cameras.PanZoomCamera()
        self.view2.camera.set_range()
        self.cursor_color = '#0FB6B6'
        self.cursor_text = scene.Text("", pos=(0, 0), italic=False, bold=True, anchor_x='left', anchor_y='center',
                                 color=self.cursor_color, font_size=24, parent=self.view2.scene)
        self.cursor_text_ref = scene.Text("", pos=(0, 0), italic=True, bold=False, anchor_x='left', anchor_y='center',
                                     color=(0, 1, 1, 1), font_size=24, parent=self.view2.scene)

        self.cursor_rect = scene.Rectangle(center=(0, 0, 0), height=1.,
                                      width=1.,
                                      radius=[0., 0., 0., 0.],
                                      color=(0.1, 0.3, 0.3, 0.5),
                                      border_width=0,
                                      border_color=(0, 0, 0, 0),
                                      parent=self.view2.scene)
        self.cursor_rect.visible = False

        self._gap_value = gap_value
        wav_visual = scene.visuals.create_visual_node(MyWaveVisual)
        if data.ndim == 1:
            data = data.reshape(-1,1)
        npts = data.shape[0]  # Number of samples per channel.
        nCh = data.shape[1]  # 32 channel number
        nrows = nCh / ncols
        if nCh==1:
            self._gap_value=0
        self.nCh = nCh
        self.waves1 = wav_visual(data, nrows, ncols, npts, 
                                 ls='-', parent=self.view2.scene, 
                                 color=color,
                                 gap=self._gap_value)
        self.set_range()

        self.grid2 = self.view2.add_grid(spacing=0, bgcolor=(0, 0, 0, 0), border_color='k')

        self.cross = Cross(cursor_color=self.cursor_color)
        self.cross.set_params(nCh, npts, fs, 0, 0, self.waves1._scale)
        self.cross.attach(self.grid2)
        self.cross.link_view(self.view2)

        self.timer_cursor = app.Timer(connect=self.update_cursor, interval=0.01, start=True)
        self.timer_cursor.start()


    def set_data(self, data, time_slice=0):
        self.cross.time_slice = time_slice
        self.waves1.set_data(data.astype('float32'))
        self.set_range()


    @property
    def gap_value(self):
        return self._gap_value

    @gap_value.setter
    def gap_value(self, value):
        if value >= 1:
            value == 1
        elif value <= 0:
            value = 0
        self._gap_value = value
        self.waves1.set_gap(self._gap_value)
        self.set_range()

        if value == 0:
            self.cross.enable_tick(axis=1)
        else:
            self.cross.disable_tick(axis=1)

    def set_range(self):
        gap = self.gap_value
        N = self.nCh
        bottom = -1 + 1./N - gap/N
        top    = bottom + gap*2
        self.view2.camera.set_range(x=(-1,1), y=(bottom, top))
        self.view2.camera.set_default_state()
        self.view2.camera.reset()

    def attach(self, gui):
        self.unfreeze()
        gui.add_view(self)

    def update_cursor(self, ev):
        pos = (self.cross.y_axis.pos[0], 0)
        gl_pos = self.view2.camera.transform.imap(pos)[0]
        t = self.cross.y_axis.glpos_to_time(gl_pos)
        n = np.ceil(t*self.cross.y_axis.fs)
        self.cursor_text.text = "   t0=%.6f sec, n=%d point" % (t,n)
        offset_x = self.view2.camera.transform.imap(self.cross.y_axis.pos)[0]
        _pos = self.view2.pos[1] + self.view2.size[1]*0.99 # bottom
        offset_y = self.view2.camera.transform.imap((0,_pos))[1]
        self.cursor_text.pos = (offset_x, offset_y)

        if self.cross.y_axis_ref.visible is True:
            # 1. cursor_text
            self.cursor_text_ref.visible = True
            pos_ref = (self.cross.y_axis_ref.pos[0], 0)
            gl_pos = self.view2.camera.transform.imap(pos_ref)[0]
            t_ref = self.cross.y_axis_ref.glpos_to_time(gl_pos)
            # calculate the time difference between t_ref and t
            delta_t = (t_ref - t)*1000
            self.cursor_text_ref.text = "   t1-t0=%.2f ms" % delta_t
            offset_x = self.view2.camera.transform.imap(self.cross.y_axis_ref.pos)[0]
            offset_y = self.view2.camera.transform.imap(self.cross.x_axis.pos)[1]
            self.cursor_text_ref.pos = (offset_x, offset_y)

            # 2. cursor_rect
            self.cursor_rect.visible = True
            y_axis_pos = (self.cross.y_axis.pos[0]+self.view2.margin,0)
            start_x = self.view2.camera.transform.imap(y_axis_pos)[0]
            self.cursor_rect.center = (start_x+self.cursor_rect._width/2.+self.cursor_rect._border_width ,0, 0)
            y_axis_ref_pos = (self.cross.y_axis_ref.pos[0]+self.view2.margin,0)
            end_x   = self.view2.camera.transform.imap(y_axis_ref_pos)[0]
            width = end_x - start_x
            if width <= 0:
                width = 1e-15
            self.cursor_rect.width = width
            height = self.view2.camera.transform.imap((0,-self.view2.size[1]))[1]
            self.cursor_rect.height = height*2
        else:
            self.cursor_text_ref.visible = False
            self.cursor_rect.visible = False

    def on_key_press(self, event):
        # if event.key.name == 'PageDown':
        #     print 'next page'
        if event.text == 'r':
            self.view2.camera.reset()
        if event.text == 'c':
            self.cross.flip_state()

    def on_mouse_move(self, event):
        modifiers = event.modifiers
        if 1 in event.buttons and modifiers is not ():
            p1 = event.press_event.pos
            p2 = event.last_event.pos
            if modifiers[0].name == 'Shift':
                self.cross.ref_enable(p2)

        elif self.cross.cross_state:
            if event.press_event is None:
                self.cross.moveto(event.pos)
                self.cross.ref_disable()

    def on_mouse_wheel(self, event):
        modifiers = event.modifiers
        if modifiers is not ():
            if modifiers[0].name=='Control':
                self.gap_value = self.gap_value + 0.05*event.delta[1]



# if __name__ == '__main__':
#     from phy.gui import GUI, create_app, run_app
#     create_app()
#     gui = GUI(position=(0, 0), size=(600, 400), name='GUI')
#     ##############################################
#     ### Test scatter_view
#     # from sklearn.preprocessing import normalize
#     # n = 1000000
#     # fet = np.random.randn(n,3)
#     # fet = normalize(fet,axis=1)
#     # print fet.shape
#     # clu = np.random.randint(3,size=(n,1))
#     # scatter_view = scatter_3d_view()
#     # scatter_view.attach(gui)
#     # scatter_view.set_data(fet, clu)
#     #############################################################################################
#     ### Set Parameters ###
#     filename  = '/tmp_data/pcie.bin'
#     nCh       = 32
#     fs        = 25000
#     numbyte   = 4
#     time_span = 1 # 1 seconds
#     global time_slice
#     time_slice = 0
#     span = time_span * fs
#     highlight = True
#     #############################################################################################
#     from Binload import Binload
#     ### Load data ###
#     lf = Binload(nCh=nCh, fs=fs)
#     lf.load(filename,'i'+str(numbyte), seekpos=0)
#     t, data = lf.tonumpyarray()
#     data = data/float(2**14)

#     ### wave_view ###
#     wview = wave_view(data[time_slice*span:(time_slice+1)*span,:], fs=25000)
#     wview.gap_value = 0.5*0.95
#     wview.attach(gui)

#     ############################################################################################
#     ### Add actions ###

#     from phy.gui import Actions
#     actions = Actions(gui)

#     @actions.add(shortcut=',')
#     def page_up():
#         global time_slice
#         time_slice -= 1
#         if time_slice >= 0:
#             wview.set_data(data[time_slice*span:(time_slice+1)*span,:], time_slice)
#         else:
#             time_slice = 0

#     @actions.add(shortcut='.')
#     def page_down():
#         global time_slice
#         time_slice += 1
#         wview.set_data(data[time_slice*span:(time_slice+1)*span,:], time_slice)

#     ##############################################################################################
#     gui.show()
#     run_app()
