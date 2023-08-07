#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  7 13:08:34 2023

@author: saimon
"""

import k3d
import numpy as np

print('importing')
A = np.genfromtxt('fig4/caso_a/uuu.csv',skip_header=1,delimiter=',')

idx = np.argsort(A[:,0])
x   = A[idx,0]
y   = A[idx,1]
z   = A[idx,2]
Q   = A[idx,3]
U   = A[idx,4]

#%%
print('meshgridding')
[X,Y,Z] = np.meshgrid(np.arange(-110,1040,3), np.arange(-80,141,2), np.arange(0,281,2));
#%%
print('griddataing')
from scipy.interpolate import griddata
vq = griddata((z,x,y),Q,(Z,X,Y),'nearest');
#%%
print('plotting')
plt_marching = k3d.marching_cubes(vq.astype(np.float32), level=0.002,
                                  color=0xf4ea0e,
                                  opacity=1,
                                  scaling=[280/130,1150/130, 220/130])

vertices = [[-0.2,-4,-1.0],[-0.2,-4,1.0],[-0.2,4,-1.0],[-0.2,4,1.0]]
indices = [[0, 1, 2], [0, 2, 3], [0, 3, 1], [3, 2, 1]]
plt_mesh = k3d.mesh(vertices, indices,
                          color=0x83f52c,
                          opacity = 0.25)

plot = k3d.Plot(camera_auto_fit=False,grid=(-1,-5,-1,1,5,1))
plot += plt_marching
plot += plt_mesh
plot.axes = ['z [D]','x [D]','y [D]']
plot.camera = [1.5,-9,-5,-1,0,0,1,0,0]

plot.display()

plot.get_snapshot()
with open('Figures/Figure2a.html','w') as fp:
  fp.write(plot.get_snapshot())
