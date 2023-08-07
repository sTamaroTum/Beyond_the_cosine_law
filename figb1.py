#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  1 11:59:05 2023

@author: saimon
"""


import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

plt.close("all")
plt.rc( 'text', usetex=True ) 
plt.rc('font',family = 'sans-serif',  size=18)

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Helvetica"],
})

plt.rcParams["text.latex.preamble"] = r"\DeclareUnicodeCharacter{0007}{\ensuremath{\alpha}}"
#%%
x = loadmat('exp_data/opt_7_param_INFLOW_ExpNumComparison_pitch.mat')
plt.figure(1,figsize=(12,5))
plt.subplot(2,3,3)
xx = x['x_num'][0][0]
xx = xx.astype(float)
plt.plot(np.squeeze(xx),np.squeeze(x['y_num'][0][0]), 'C1o',label='Model',linewidth=2)
plt.grid()
plt.yticks([0.4,0.45,0.5],color='white')
plt.ylim([0.4,0.5])
plt.xticks(np.linspace(-20,20,5),color='white')
plt.xlim([-20,20])
plt.title('High-TI',fontsize=18)
plt.subplot(2,3,2)
xx = x['x_num'][1][0]
xx = xx.astype(float)
plt.plot(np.squeeze(xx),np.squeeze(x['y_num'][1][0]), 'C1o',linewidth=2)
plt.grid()
plt.ylim([0.4,0.5])
plt.xticks(np.linspace(-40,40,5),color='white')
plt.xlim([-40,40])
plt.yticks([0.4,0.45,0.5],color='white')
plt.title('Mid-TI',fontsize=18)
plt.subplot(2,3,1)
xx = x['x_num'][2][0]
xx = xx.astype(float)
plt.plot(np.squeeze(xx),np.squeeze(x['y_num'][2][0]), 'C1o',label='Model',linewidth=2)
plt.ylabel(r'$\theta_p$ [$^\circ$]')
plt.grid()
plt.ylim([0.4,0.5])
plt.xticks(np.linspace(-40,40,5),color='white')
plt.xlim([-40,40])
plt.yticks([0.4,0.45,0.5],color='k')
plt.title('Low-TI',fontsize=18)
plt.figure(1)
plt.tight_layout()
#%%
x = loadmat('exp_data/opt_7_param_INFLOW_ExpNumComparison_TSR.mat')
plt.figure(1,figsize=(12,3))
plt.subplot(2,3,6)
xx = x['x_num'][0][0]
xx = xx.astype(float)
plt.plot(np.squeeze(xx),np.squeeze(x['y_num'][0][0]), 'C1o',label='Model',linewidth=2)
plt.xlabel('$\gamma$ [$^\circ$]')
plt.grid()
plt.ylim([7.2,8.4])
plt.xticks(np.linspace(-20,20,5))
plt.xlim([-20,20])
plt.yticks([7.2,7.6,8,8.4],color='white')
plt.subplot(2,3,5)
xx = x['x_num'][1][0]
xx = xx.astype(float)
plt.plot(np.squeeze(xx),np.squeeze(x['y_num'][1][0]), 'C1o',linewidth=2)
plt.xlabel('$\gamma$ [$^\circ$]')
plt.grid()
plt.ylim([7.2,8.4])
plt.yticks([7.2,7.6,8,8.4],color='white')
plt.xticks(np.linspace(-40,40,5))
plt.xlim([-40,40])
plt.subplot(2,3,4)
xx = x['x_num'][2][0]
xx = xx.astype(float)
plt.plot(np.squeeze(xx),np.squeeze(x['y_num'][2][0]), 'C1o',label='Model',linewidth=2)
plt.ylabel('$\lambda$ [-]')
plt.xlabel('$\gamma$ [$^\circ$]')
plt.grid()
plt.ylim([7.2,8.4])
plt.yticks([7.2,7.6,8,8.4])
plt.xticks(np.linspace(-40,40,5))
plt.xlim([-40,40])

plt.subplots_adjust(left=0.1, bottom=0.2, right=0.95, top=0.9, wspace=0.2, hspace=0.2)
#%%
plt.savefig('Figures/figb1.png',dpi=300)