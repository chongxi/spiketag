import numpy as np
import sys
from vispy import app, visuals, scene
from spiketag.view import wave_view

fs = 20000.
dt = 1/fs
t  = np.arange(0., 20., dt)

phase = np.arange(0,2*np.pi,2*np.pi/32).reshape(-1,1)
x = np.sin(2*np.pi*2*t + phase)
x = x.T
wview = wave_view(x, fs=fs)


if __name__ == '__main__':
    if sys.flags.interactive != 1:
        wview.show()
        app.run()
