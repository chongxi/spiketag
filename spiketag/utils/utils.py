import re
from collections import defaultdict
from functools import partial
from time import time


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
            print '{0}, elapsed time: {1} ms'.format(self.task, self.msecs)


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
            funcName = func.__module__ + '.' + func.__name__
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
        funcName = func.__module__ + '.' + func.__name__
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

    def emit(self, event, *args, **kwargs):
        """Call all callback functions registered with an event.

        Any positional and keyword arguments can be passed here, and they will
        be forwarded to the callback functions.

        Return the list of callback return results.

        """
        res = []
        for callback in self._callbacks.get(event, []):
            res.append(callback(*args, **kwargs))
        return res