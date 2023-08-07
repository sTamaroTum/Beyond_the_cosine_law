#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  2 10:41:49 2023

@author: saimon
"""

import numpy as np
import matplotlib.pyplot as plt

plt.close("all")
plt.rc( 'text', usetex=True ) 
plt.rc('font',family = 'sans-serif',  size=18)
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Helvetica"],
})

#%% Define constants
hub_height = 110
R          = 65
k_arr      = np.array([0, 0.1, 0.3])
z          = np.linspace(hub_height-R*1.1,hub_height+R*1.1,21)
#%% Compute velocity profile (power law) and fit linear shear
plt.figure(figsize=(5.8,3.7))
mark      = np.array(["o","^","s"])
scenarios = np.array(['Scenarios \#1,2','Scenarios \#3','Scenarios \#4'])
c = 0
for k in k_arr:
    u = (z/hub_height)**k
    plt.scatter(u,-(z-hub_height)/(2*R), 60,'C' + str(c),marker=mark[c],linewidth=0,label=scenarios[c],zorder=10)
    poli = -np.polyfit(-(z-hub_height)/R,1-u,1)
    plt.plot(np.polyval(poli,-(z-hub_height)/R)+1,-(z-hub_height)/(2*R),':',color='C' + str(c))
    c += 1
plt.legend(frameon=False,handlelength=0.5,fontsize=18,bbox_to_anchor=(0.53,0.425))
plt.plot([0,10],[0.5,0.5],'k',  linewidth=0.75,zorder=4)
plt.plot([0,10],[0,0]    ,'--k',linewidth=0.75,zorder=4)
plt.plot([0,10],[-0.5,-0.5],'k',linewidth=0.75,zorder=4)
plt.ylim([-0.6,0.6])
plt.xlim([0.7,1.2])
plt.xlabel('$u/u_{\infty,\mathrm{hub}}$ [-]',fontsize=18)
plt.ylabel('$z_g$ [$D$]',fontsize=18)
plt.yticks(np.linspace(-0.6,0.6,5),fontsize=18)
plt.xticks(np.linspace(0.7,1.2,6),fontsize=18)
plt.gca().invert_yaxis()
plt.tight_layout()
#%% Save figure
plt.savefig('Figures/fig6.png',dpi=300)