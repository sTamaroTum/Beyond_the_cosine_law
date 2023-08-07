#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  7 13:09:06 2023

@author: saimon
"""

import k3d
import numpy as np

print('importing')
A = np.genfromtxt('fig4/caso_d/u.csv',skip_header=1,delimiter=',')

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
#%% Define wake deflection plane
xxx = np.array([-4.423076923076923,4.423076923076923])*130
yyy = np.linspace(-130,130,2)
zzz = np.array([110+130,110-90])-20
vettore = np.array([0    ,      0.46885193, -0.17364818     ])
i0 = (xxx)/130
i1 = -(-vettore[2]*(-zzz+110)/vettore[1])/130
i2 = (-zzz+110)/130
#%#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#%#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#%#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
print('plotting')
plt_marching = k3d.marching_cubes(vq.astype(np.float32), level=0.002,
                                  color=0xff7800,
                                  opacity=1,
                                  scaling=[280/130,1150/130, 220/130])

vertices = [[i2[1],i0[0],i1[0]],[i2[0],i0[0],i1[1]],[i2[1],i0[1],i1[0]],[i2[0],i0[1],i1[1]]]
indices = [[0, 1, 2], [0, 2, 3], [0, 3, 1], [3, 2, 1]]
plt_mesh = k3d.mesh(vertices, indices,
                          color=0x83f52c,
                          opacity = 0.25)

plot = k3d.Plot(camera_auto_fit=False,grid=(-1,-5,-1,1,5,1))
plot += plt_marching
plot += plt_mesh
plot.axes = ['z [D]','x [D]','y [D]']
# camera [posx,posy,posz,targetx,targety,targetz,upx,upy,upz]"
plot.camera = [1.5,-9,-5,-1,0,0,1,0,0]
plot.display()

plot.get_snapshot()
with open('Figure2b.html','w') as fp:
  fp.write(plot.get_snapshot())
