import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

plt.close("all")
#%% Load data
x        = loadmat('exp_data/Sulution_opt_7_param.mat')
#%%
fig, ax1 = plt.subplots(figsize=(5.5,3.5))
ax2      = ax1.twinx()
xx       = x['x_num'][1][0]
xx       = xx.astype(float)
ax1.plot(np.squeeze(xx),np.squeeze(x['y_num'][1][0]), 'C0s-',label='$C_D$',markersize=12, zorder=1000, clip_on=False)
ax1.plot(np.squeeze(xx)+870,np.squeeze(x['y_num'][0][0]), 'C1^--',label='$C_{L,\a}$',markersize=12)
ax1.legend(frameon=False,bbox_to_anchor=(0.08,1))
xx       = x['x_pat'][1][0]
xx       = xx.astype(float)
ax1.fill_between(np.squeeze(xx)[0:3],np.squeeze(x['y_pat'][1][0])[0:3],np.flip(np.squeeze(x['y_pat'][1][0])[3:6]),alpha=0.2,color='C0',edgecolor=None)#, 'C0s-',label='$c_D$')
xx       = x['x_pat'][0][0]
xx       = xx.astype(float)
ax2.fill_between(np.squeeze(xx)[0:3],np.squeeze(x['y_pat'][0][0])[0:3],np.flip(np.squeeze(x['y_pat'][0][0])[3:6]),alpha=0.2,color='C1',edgecolor=None)#, 'C0s-',label='$c_D$')
xx = x['x_num'][0][0]
xx= xx.astype(float)
ax2.plot(np.squeeze(xx),np.squeeze(x['y_num'][0][0]), 'C1^--',label='$C_{L,\a}$',markersize=12, zorder=1000, clip_on=False)
ax1.set_xlabel('$\Omega$ [RPM]',color='k')
ax1.set_ylabel('$C_D$ [-]', color='C0')
ax2.set_ylabel("$C_{L,\a}$ [1/rad]", color='C1')
ax1.set_ylim([0,0.05])
ax1.set_yticks([0,0.01,0.02,0.03,0.04,0.05])#,color='C0')
ax2.set_yticks([3.2,3.6,4.0,4.4,4.8,5.2])#,color='C0')
ax1.tick_params(axis='y', colors='C0')
ax2.tick_params(axis='y', colors='C1')
ax2.set_ylim([3.2,5.2])
ax1.set_xticks([400,550,700,850])
ax1.set_xlim([400,850])
ax2.grid(None)
ax1.grid(which='major', axis='both')
plt.tight_layout()
#%% Save figure
# plt.savefig('Figures/fig11.png',dpi=300)
