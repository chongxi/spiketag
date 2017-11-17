
# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import io
import numpy as np
%gui qt

#%%
from vispy import app,keys
from spiketag.utils import Timer

#%%
r32 = io.open('/dev/xillybus_fet_clf_32', 'rb')
r32_buf = io.BufferedReader(r32)
_size=7*4  # 7*32bits


#%%
from spiketag.view.grid_scatter3d import grid_scatter3d as gdview
rows, cols = 6, 8
win = gdview(rows, cols) 
win.show()


#%%
def fet_frombuffer(buf, npts, dtype='int32', fix_points=16):
    '''
    buffer structure:
    {0:time, 1:grpNo, 2:fet0, 3:fet1, 4:fet2, 5:fet3,  6:1nn_id}
    '''
    fet = np.frombuffer(buf, dtype).reshape(npts, 7)
    t       = fet[:, 0]
    grpNo   = fet[:, 1]   
    fet_val = fet[:, 2:6].reshape(-1,4)/float(2**fix_points)
    label   = fet[:, 6]
    return t, grpNo, fet_val, label


#%%
def update():
    n = 160
    with Timer('read data from FPGA'):
        buf = r32.read(n*_size)
        t, grpNo, fet_val, label = fet_frombuffer(buf, n, 'int32', 16)
        ssrate = 2
#    print t[-1], grpNo[-1]
    with Timer('update all views'):
        for i in range(40):
    #        print fet_val
    #        assert(grpNo<48)
            idx = np.where(grpNo==i)[0]
            if idx.shape[0]>0:
                new_fet = fet_val[idx, :]
                new_clu = np.ones((new_fet.shape[0],), dtype=np.int32)
                with Timer('update single views', verbose=False):
                    win.fet_view[i].stream_in(new_fet, new_clu)


#%%
def reset_fet_viz(N):
    fet = np.zeros((N,4), dtype=np.float32)
    clu = np.zeros((N,),  dtype=np.int32)
    for idx in range(rows*cols):
        win.fet_view[idx].set_data(fet, clu)

#%%
reset_fet_viz(500)

#%%
#timer = app.Timer(connect=update, interval=0.030)
from PyQt4 import QtCore
timer = QtCore.QTimer()
timer.timeout.connect(update)

#%%
timer.start(10)

#%%
timer.stop()
