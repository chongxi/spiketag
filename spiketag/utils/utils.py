import re
import os
from collections import defaultdict
from functools import partial
from time import time
from vispy import scene, app
import numpy as np
from matplotlib import path
import spiketag
from . import conf

#------------------------------------------------------------------------------
# Simple Timer for performance test
#------------------------------------------------------------------------------


class Timer(object):
    '''
    For any chunk of code A:
    with Timer('task'):
        A
    '''
    def __init__(self, task, verbose=True):
        self.verbose = verbose
        self.task = task

    def __enter__(self):
        self.start = time()
        return self

    def __exit__(self, *args):
        self.end = time()
        self.secs = self.end - self.start
        self.msecs = self.secs * 1000  # millisecs
        if self.verbose:
            print('{0}, elapsed time: {1} ms'.format(self.task, self.msecs))


#------------------------------------------------------------------------------
# Event system
#------------------------------------------------------------------------------

class EventEmitter(object):
    """Class that emits events and accepts registered callbacks.

    Derive from this class to emit events and let other classes know
    of occurrences of actions and events.

    Example
    -------

    ```python
    class MyClass(EventEmitter):
        def f(self):
            self.emit('my_event', 1, key=2)

    o = MyClass()

    # The following function will be called when `o.f()` is called.
    @o.connect
    def on_my_event(arg, key=None):
        print(arg, key)

    ```

    """

    def __init__(self):
        self._reset()

    def _reset(self):
        """Remove all registered callbacks."""
        self._callbacks = defaultdict(list)

    def _get_on_name(self, func):
        """Return `eventname` when the function name is `on_<eventname>()`."""
        r = re.match("^on_(.+)$", func.__name__)
        if r:
            event = r.group(1)
        else:
            raise ValueError("The function name should be "
                             "`on_<eventname>`().")
        return event

    def _registered_func_name(self, event):
        funcNamelist = []
        for func in self._callbacks[event]:
            funcName = func.__module__ + '.' + func.__name__ + '_id' + str(id(func))
            funcNamelist.append(funcName)
        return funcNamelist

    def _create_emitter(self, event):
        """Create a method that emits an event of the same name."""
        if not hasattr(self, event):
            setattr(self, event,
                    lambda *args, **kwargs: self.emit(event, *args, **kwargs))

    def connect(self, func=None, event=None, set_method=False):
        """Register a callback function to a given event.

        To register a callback function to the `spam` event, where `obj` is
        an instance of a class deriving from `EventEmitter`:

        ```python
        @obj.connect
        def on_spam(arg1, arg2):
            pass
        ```

        This is called when `obj.emit('spam', arg1, arg2)` is called.

        Several callback functions can be registered for a given event.

        The registration order is conserved and may matter in applications.

        """
        if func is None:
            return partial(self.connect, set_method=set_method)

        # Get the event name from the function.
        if event is None:
            event = self._get_on_name(func)

        # We register the callback function.
        # if func is not in self._callbacks[event]:
        funcName = func.__module__ + '.' + func.__name__ + '_id' + str(id(func))
        if funcName not in self._registered_func_name(event):
            self._callbacks[event].append(func)

        # A new method self.event() emitting the event is created.
        if set_method:
            self._create_emitter(event)

        return func



    def unconnect(self, *funcs):
        """Unconnect specified callback functions."""
        for func in funcs:
            for callbacks in self._callbacks.values():
                if func in callbacks:
                    callbacks.remove(func)

    def emit(self, event, caller=None, *args, **kwargs):
        """Call all callback functions registered with an event.

        Any positional and keyword arguments can be passed here, and they will
        be forwarded to the callback functions.

        Return the list of callback return results.

        """
        res = []
        for callback in self._callbacks.get(event, []):
            if caller and caller == callback.__module__:
               continue 

            with Timer('[Event] emit -- {}'.format(callback.__module__), verbose=conf.ENABLE_PROFILER):
                res.append(callback(*args, **kwargs))
        return res

#------------------------------------------------------------------------------
# Picker
#------------------------------------------------------------------------------
class Picker(object):

    """Class that pick the markers by lasso or rectangle.



    Parameters
    ----------
    cur_scene :  scene
        the current scene of canvas.
    cur_view  :  view
        the current view of scene, consider this as local coordinate
    cur_scatter: Markers
        the be selected markers

    Example
    -------

    ```
       picker = Picker(self.scene,self.view,markers)
       picker.origin_point(point)
       picker.cast_net(cur_position,ptype='lasso')
       selected = picker.pick(beSelectedPoints)
    ```

    """

    def __init__(self,cur_scene,mapping):
        """
           origin: save origin because when we draw rectangle, we need to use origin to calculate the width and height
           vertices, line: vertices is point position which be used to draw a line
           mapping: the mapping from markers coordinate to view coordinate
        """
        self._origin = (0, 0)
        self._vertices = []
        self._line = None
        self._mapping = mapping 
        self._scene = cur_scene
        self._trigger = False

    """
        save the origin point when draw begin.

        Parameters
        ----------
        point :  array
            2d, screen coordinate,usually the point is the first press of mouse.
    """
    def origin_point(self,point):
        self.reset()

        self._origin = point
        self._line = scene.visuals.Line(color='white', method='gl',
                                       parent=self._scene)
        self._trigger = True

    """
        cast a net by rectange or lasso, rectange is default

        Parameters
        ----------
        pos :    array
            2d, screen coordinate,usually the point when mouse moving.
        ptype :  string
            type of cast, rectangle or lasso
    """
    def cast_net(self,pos,ptype='rectangle'):
        if not self._trigger:
            return 
        
        if ptype == 'rectangle':
            self._cast_rectangle(pos)
        elif ptype == 'lasso':
            self._cast_lasso(pos)
        else:
            raise RuntimeError('not support yet!')


    """
        pick points from given samples because the mapping from marker coordinate to screen coordinate which given before,
        return indices of points be selected

        Parameters
        ----------
        samples :    array
            samples which be selected
        ptype :      string
            type of cast, rectangle or lasso
        return:      array
            points be selected
    """
    def pick(self, samples, auto_disappear=True):
        if not self._trigger:
            return np.array([])

        mask = np.array([])
        if len(self._vertices):
            data = self._mapping.map(samples[:, :3])[:, :2]
            select_path = path.Path(self._vertices, closed=True)
            selected = select_path.contains_points(data)
            mask = np.where(selected)[0]
        if auto_disappear:
            self.reset()
        return mask

    """
        clear all values
    """
    def reset(self):
        self._vertices = []
        if self._line:
            self._line.parent = None
            self._line = None
        self._origin = None
        self._trigger = False



    """
        cast by rectangle, basically get the vertices which can draw the rectangle
    """
    def _cast_rectangle(self,pos):
        width = pos[0] - self._origin[0]
        height = pos[1] - self._origin[1]
        if width and height:
            center = (width/2. + self._origin[0],
                      height/2.+self._origin[1], 0)
            self._vertices = self._gen_rectangle_vertice(center, abs(height), abs(width))
            self._line.set_data(np.array(self._vertices))

    """
        cast by lasso, need all position when mouse moving
    """
    def _cast_lasso(self,pos):
        self._vertices.append(pos)
        self._line.set_data(np.array(self._vertices))


    """
        use the scene.visuals.Rectangle to get the vertices.
    """
    def _gen_rectangle_vertice(self,center, height, width):
        # TODO: here is little wired, but becuase the function generate_vertices of Rectangle.py doesn't update the
        #       value of height and width, why they doesnt do this, so wired
        rectangle = scene.visuals.Rectangle(height=height,width=width)
        radius = np.array([.0,.0,.0,.0])
        return rectangle._generate_vertices(center=center,radius=radius,height=height,width=width)[1:, ..., :2]


#------------------------------------------------------------------------------
# key buffer: press number key to push, press 'g' to pop and execute
#------------------------------------------------------------------------------
class key_buffer(object):
    '''
    string of numbers that can be pop and push
    '''
    def __init__(self):
        self._buf = ''

    def push(self, v):
        self._buf += v

    def pop(self):
        buf = self._buf
        self._buf = ''
        return buf

    def __repr__(self):
        return self._buf
