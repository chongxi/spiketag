# Define a simple vertex shader. We use $template variables as placeholders for
# code that will be inserted later on. In this example, $position will become
# an attribute, and $transform will become a function. Important: using
# $transform in this way ensures that users of this visual will be able to
# apply arbitrary transformations to it.
from vispy import app, gloo, visuals, scene, keys
from vispy.util import ptime
import numpy as np

VERT_SHADER = """
#version 120

// y coordinate of the position.
attribute float y;

// row, col, and time index.
attribute vec3 a_index;
varying vec3 v_index;

// Size of the table.
uniform vec2 u_size;

// Number of samples per signal.
uniform float u_npts;

// Vertical gap
uniform float u_gap;

// Color.
attribute vec4 a_color;
varying vec4 v_color;

void main() {
    float nrows = u_size.x;
    float ncols = u_size.y;

    // Compute the x coordinate from the time index.
    float x = -1 + 2*a_index.z / (u_npts-1);
    vec2 position = vec2(x, y);

    // Find the affine transformation for the subplots.
    vec2 a = vec2(1./ncols, 1./nrows)*.95;
    vec2 b = vec2(-1 + 2*(a_index.x+.5) / ncols,
                  -1 + 2*(a_index.y * u_gap+.5) / nrows);

    // Apply the static subplot transformation + scaling.
    gl_Position = $transform(vec4(a*position+b, 0.0, 1.0));

    v_color = a_color;
    v_index = a_index;

}
"""

# Very simple fragment shader. Again we use a template variable "$color", which
# allows us to decide later how the color should be defined (in this case, we
# will just use a uniform red color).

FRAG_SHADER = """
#version 120

varying vec4 v_color;
varying vec3 v_index;

void main() {
    gl_FragColor = v_color;

    // Discard the fragments between the signals (emulate glMultiDrawArrays).
    if ((fract(v_index.x) > 0.) || (fract(v_index.y) > 0.))
        discard;

}
"""

# Start the new Visual class.
# By convention, all Visual subclass names end in 'Visual'.
# (Custom visuals may ignore this convention, but for visuals that are built
# in to vispy, this is required to ensure that the VisualNode subclasses are
# generated correctly.)
class MyWaveVisual(visuals.Visual):
    """Visual that draws a red rectangle.

    Parameters
    ----------
    x : float
        x coordinate of rectangle origin
    y : float
        y coordinate of rectangle origin
    w : float
        width of rectangle
    h : float
        height of rectangle

    All parameters are specified in the local (arbitrary) coordinate system of
    the visual. How this coordinate system translates to the canvas will
    depend on the transformation functions used during drawing.
    """

    # There are no constraints on the signature of the __init__ method; use
    # whatever makes the most sense for your visual.
    def __init__(self, ncols=1, color=None, ls='-', gap=1):
        # Initialize the visual with a vertex shader and fragment shader
        visuals.Visual.__init__(self, VERT_SHADER, FRAG_SHADER)
       
        self.nCh = 0
        self.npts = 0
        self.ncols = ncols
        self.nrows = 0
        self.index = np.array([])
        self.data = np.array([])
        self.color = color
        self.ls = ls
        self.gap = gap
        self._scale = 1
        # self.pcie_open = False
        # self.pcie_read_open()
        # self.timer0 = app.Timer(interval=0, connect=self._timer_data, start=False)
        self.timer1 = app.Timer(interval=0, connect=self._timer_show, start=False)
        # self._last_time = 0
        if self.ls == '.':
            self._draw_mode = 'points' 
        elif self.ls == '-':
            self._draw_mode = 'line_strip'

    def _prepare_transforms(self, view):
        # This method is called when the user or the scenegraph has assigned
        # new transforms to this visual (ignore the *view* argument for now;
        # we'll get to that later). This method is thus responsible for
        # connecting the proper transform functions to the shader program.

        # The most common approach here is to simply take the complete
        # transformation from visual coordinates to render coordinates. Later
        # tutorials detail more complex transform handling.
        view.view_program.vert['transform'] = view.get_transform()

    def _timer_show(self, ev):
        self.data = self.data.astype('float32')
        self.shared_program['y'] = self.data.reshape(-1,self.nCh).T.ravel()/300
        self.update()

    def highlight(self, spacial_code, temporal_code, highlight_color=None):
        '''
        highlight segment of the signals in one or several channels
        group of channels is defined in spacial_code: (0,2,4,6) means ch0,ch2,ch4,ch6
        segment of signal is defined in temporal_code: (n0, n1) means from point n0 to point n1
        there can be many rows of temporal_code coressponding to several segments: [[n0,n1],[n2,n3]...]
        '''
        if highlight_color is None:
            highlight_color = (0,1,0,1)
        # npts = self.color.shape[0]/self.nCh
        for chNo in spacial_code:
            for nrange in temporal_code:
                n0, n1 = nrange   # from n0 to n1
                start  = n0+chNo*self.npts
                end    = n1+chNo*self.npts
                self.color[start:end,:] = np.asarray(highlight_color)       
        self.shared_program['a_color'] = self.color
        self.update()


    def highlight_reset(self):
        self.color = np.repeat(np.ones((self.nCh,4)),
                                        self.npts, axis=0).astype(np.float32)
        self.shared_program['a_color'] = self.color
        self.update()


    def highlight_ch(self, ch, highlight_color=None, mask_others=False):
        '''
        highlight segment of the signals in one or several channels
        group of channels is defined in spacial_code: (0,2,4,6) means ch0,ch2,ch4,ch6
        segment of signal is defined in temporal_code: (n0, n1) means from point n0 to point n1
        there can be many rows of temporal_code coressponding to several segments: [[n0,n1],[n2,n3]...]
        '''
        if highlight_color is None:
            highlight_color = (0,1,0,1)
        npts = self.color.shape[0]/self.nCh
        if isinstance(ch, int):
            chNo = ch
            n0, n1 = 0, npts   # from n0 to n1
            start  = n0 + chNo*self.npts
            end    = n1 + chNo*self.npts
            self.color[start:end,:] = np.asarray(highlight_color)      
            if mask_others is True:
                mask = np.delete(np.arange(self.nCh), ch)
                for chNo in mask:
                    n0, n1 = 0, npts   # from n0 to n1
                    start  = n0 + chNo*self.npts
                    end    = n1 + chNo*self.npts
                    self.color[start:end,:] = (0,0,0,0)                

        elif isinstance(ch, list) or isinstance(ch, tuple) or isinstance(ch, np.ndarray):
            for i, chNo in enumerate(ch):
                n0, n1 = 0, npts   # from n0 to n1
                start  = n0 + chNo*self.npts
                end    = n1 + chNo*self.npts
                if np.ndim(highlight_color) == 1:
                    self.color[start:end,:] = np.asarray(highlight_color)   
                elif np.ndim(highlight_color) == 2:
                    self.color[start:end,:] = np.asarray(highlight_color[i])   
            if mask_others is True:
                mask = np.delete(np.arange(self.nCh), ch)
                for chNo in mask:
                    n0, n1 = 0, npts   # from n0 to n1
                    start  = n0 + chNo*self.npts
                    end    = n1 + chNo*self.npts
                    self.color[start:end,:] = (1,1,1,0.5)    

        self.shared_program['a_color'] = self.color
        self.update()


    def highlight_spikes(self, highlight_list, color=(0,1,0,1)):
        pre_peak  = 5
        post_peak = 5
        for tup in highlight_list:
            temporal_code = [[tup[0]-pre_peak, tup[0]+post_peak]]
            # print temporal_code
            spacial_code = [tup[1],]
            # print spacial_code
            self.highlight(spacial_code, temporal_code, color)


    def set_data(self, data):
        
        self.data = data.astype('float32')
        if self.data.ndim == 1:
            self.data = self.data.reshape(-1,1)
     
        ####### extract meta data from data #######
        self.nCh = self.data.shape[1]
        self.npts = self.data.shape[0]
        self.nrows = self.nCh / self.ncols
    
        ####### scale data #######
        self._scale = self.data.max()-self.data.min()
        self.data = self.data.T.ravel()/self._scale
        
        self.highlight_reset()
        
        self._render()

    def append_data(self, data):
        newdata = data.astype('float32')
        newdata = newdata.T.ravel()/self._scale
        self.shared_program['y'] = np.hstack((self.data, newdata))
        self.update()

    def get_gl_pos(self):
        '''
            convert data to opengl position, then we can use this position to transform to other coordinate system, eg:
            document coordinate system or viewport coordinate system
        '''
        x_pos = -1 + 2 * self.index[:,2]/ (self.npts - 1)
        y_pos = self.data
        pos = np.column_stack((x_pos,y_pos))

        a = np.array([1.0 / self.ncols,1.0 / self.nrows]) * 0.95
        b = np.array([(-1 + 2 * (self.index[:,0] + 0.5) / self.ncols),(-1 + 2 * (self.index[:,1] * self.gap
             + 0.5) / self.nrows)])

        return a * pos + b.T
        
    def set_gap(self, gap):
        self.gap = gap
        self.shared_program['u_gap'] = gap
        self.update()

    def _render(self):

        # index is (#cols*#rows*#npts, 3) # each row of index is (col_idx, row_idx, npts_idx) 
        # (col,row):
        # (0,0)->(0,1)->(0,2)->(0,3)->(0,4)-...->(0,7)->(1,0)->(1,1)-...->(1,7)
        # index = np.c_[np.repeat(np.repeat(np.arange(ncols), nrows), npts),
        #               np.repeat(np.tile(np.arange(nrows), ncols), npts),
        #               np.tile(np.arange(npts), nCh)].astype(np.float32)

        # (col,row):
        # (0,0)->(1,0)->(0,1)->(1,1)->(0,2)->(1,2)...->(0,7)->(1,7)
        self.index = np.c_[np.repeat(np.tile(np.arange(self.ncols), self.nrows), self.npts),
                      np.repeat(np.arange(self.nrows), self.ncols*self.npts),
                      np.tile(np.arange(self.npts), self.nCh)].astype(np.float32)
        
        if self.color is 'random':
            self.color = np.repeat(np.random.uniform(size=(self.nCh, 4), low=.2, high=.9),
                              self.npts, axis=0).astype(np.float32)            
        elif self.color is None:
            self.color = np.repeat(np.ones((self.nCh,4)),
                              self.npts, axis=0).astype(np.float32)
        
        self.shared_program['y'] = self.data
        self.shared_program['a_color'] = self.color
        self.shared_program['a_index'] = self.index
        self.shared_program['u_size'] = (self.nrows, self.ncols)
        self.shared_program['u_npts'] = self.npts
        self.shared_program['u_gap'] = self.gap
        # self.shared_program['clip'] = 1.0

        # self.shared_program.vert['position'] = self.vbo
        # self.shared_program.frag['color'] = (0, 1, 0, 1)

    def draw(self):
        '''
            if MyWaveVisual have no data, should not be drawed, otherwise the vispy/visual will raise a RuntimeError
        '''
        if len(self.data) == 0:
            return
        super(MyWaveVisual, self).draw()


