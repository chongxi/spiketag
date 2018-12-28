import numpy as np
import sys
from vispy import app, visuals, scene
from spiketag.view import volume_view

#%%
V = np.zeros((3,200,200,200))

#%%
'''
The square
'''
s=40
r=20
m = 100
V[0, m-s:m+s, m-s:m+s, m-s:m+s] = 0.3
V[0, m-r:m+r, m-r:m+r, m-r:m+r] = 0.7
V[0, m-10:m+10, m-10:m+10, m-10:m+10] = 0.15

#%%
'''
sine wave
'''
X = np.arange(-25, 25, 0.25*1)
Y = np.arange(-25, 25, 0.25*1)
Z = np.arange(-25, 25, 0.25*1)
X, Y, Z = np.meshgrid(X, Y, Z)
R = np.sqrt(X**2 + Y**2 + Z**2)
V[1, ...] = np.sin(R)

#%%
'''
randn
'''
V[2, ...] = np.random.randn(200,200)*10

#%%
N = 3
vol_view = volume_view(nvbs=N)
for i in range(N):
    vol_view.set_data(V[i], vb_id=i)

#%%
if __name__ == '__main__':
    if sys.flags.interactive != 1:
        vol_view.show()
        app.run()

