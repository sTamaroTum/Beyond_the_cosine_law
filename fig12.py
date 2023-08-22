import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
plt.close("all")
#%% Plot ETA_P
x = loadmat('exp_data/opt_7_param_INFLOW_ExpNumComparison_power.mat')
plt.figure(1,figsize=(12,5.5))
plt.subplot(2,3,3)
xx       = x['x_num'][0][0]
xx       = xx.astype(float)
plt.plot(np.squeeze(xx),np.squeeze(x['y_num'][0][0]), 'C0-',label='Model',linewidth=2)
xx       = x['x_mod'][0][0]
xx       = xx.astype(float)
plt.errorbar(np.squeeze(xx),np.squeeze(x['y_mod'][0][0]),np.squeeze(x['err_mod'][0][0]),color='C1',fmt='o',barsabove=True,linestyle='',capsize=5,elinewidth=1,label='Exp.')
plt.grid()
plt.ylim([0.6,1.1])
plt.xticks(np.linspace(-40,40,5),color='white')
plt.xlim([-40,40])
plt.yticks([0.6,0.8,1.0],color='white')
plt.title('High-TI',fontsize=18)
plt.subplot(2,3,2)
xx       = x['x_num'][1][0]
xx       = xx.astype(float)
plt.plot(np.squeeze(xx),np.squeeze(x['y_num'][1][0]), 'C0-',label='Model',linewidth=2)
xx       = x['x_mod'][1][0]
xx       = xx.astype(float)
plt.errorbar(np.squeeze(xx),np.squeeze(x['y_mod'][1][0]),np.squeeze(x['err_mod'][1][0]),color='C1',fmt='o',barsabove=True,linestyle='',capsize=5,elinewidth=1,label='Exp.')
plt.grid()
plt.ylim([0.6,1.1])
plt.xticks(np.linspace(-40,40,5),color='white')
plt.xlim([-40,40])
plt.yticks([0.6,0.8,1.0],color='white')
plt.title('Mid-TI',fontsize=18)
plt.legend(handlelength=1,loc='upper left',frameon=False,ncol=2,bbox_to_anchor=(0.03, 1.5),fancybox=True, shadow=False)
#
plt.subplot(2,3,1)
xx       = x['x_num'][2][0]
xx       = xx.astype(float)
plt.plot(np.squeeze(xx),np.squeeze(x['y_num'][2][0]), 'C0-',label='Model',linewidth=2)
xx       = x['x_mod'][2][0]
xx       = xx.astype(float)
plt.errorbar(np.squeeze(xx),np.squeeze(x['y_mod'][2][0]),np.squeeze(x['err_mod'][2][0]),color='C1',fmt='o',barsabove=True,linestyle='',capsize=5,elinewidth=1)
plt.ylabel(r'$\eta_P$ [-]')
plt.grid()
plt.ylim([0.6,1.1])
plt.xticks(np.linspace(-40,40,5),color='white')
plt.xlim([-40,40])
plt.yticks([0.6,0.8,1.0],color='k')
plt.title('Low-TI',fontsize=18)
plt.figure(1)
plt.tight_layout()
#%% Plot ETA_T
x = loadmat('exp_data/opt_7_param_INFLOW_ExpNumComparison_CT.mat')
plt.subplot(2,3,6)
xx       = x['x_num'][0][0]
xx       = xx.astype(float)
novo_y = np.squeeze(x['y_mod'][0][0])/np.squeeze(x['y_mod'][0][0])[5]
novo_error = novo_y*(  (np.squeeze(x['err_mod'][0][0])/np.squeeze(x['y_mod'][0][0]))**2  + (np.squeeze(x['err_mod'][0][0])[5]/np.squeeze(x['y_mod'][0][0])[5])**2 )**0.5
plt.plot(np.squeeze(xx),np.squeeze(x['y_num'][0][0])/np.squeeze(x['y_num'][0][0])[5], 'C0-',label='Model',linewidth=2)
xx       = x['x_mod'][0][0]
xx       = xx.astype(float)
plt.errorbar(np.squeeze(xx),novo_y,novo_error,color='C1',fmt='o',barsabove=True,linestyle='',capsize=5,elinewidth=1,label='Exp.')
plt.xlabel('$\gamma$ [$^\circ$]')
plt.grid()
plt.ylim([0.6,1.1])
plt.xticks(np.linspace(-40,40,5))
plt.xlim([-40,40])
plt.yticks([0.6,0.8,1.0],color='white')
plt.subplot(2,3,5)
xx       = x['x_num'][1][0]
xx       = xx.astype(float)
plt.plot(np.squeeze(xx),np.squeeze(x['y_num'][1][0])/np.squeeze(x['y_num'][1][0])[5], 'C0-',linewidth=2)
novo_y = np.squeeze(x['y_mod'][1][0])/np.squeeze(x['y_mod'][1][0])[5]
novo_error = novo_y*(np.squeeze(  (x['err_mod'][1][0])/np.squeeze(x['y_mod'][1][0]))**2  + (np.squeeze(x['err_mod'][1][0])[5]/np.squeeze(x['y_mod'][1][0])[5])**2 )**0.5
xx       = x['x_mod'][1][0]
xx       = xx.astype(float)
plt.errorbar(np.squeeze(xx),novo_y,novo_error,color='C1',fmt='o',barsabove=True,linestyle='',capsize=5,elinewidth=1)
plt.xlabel('$\gamma$ [$^\circ$]')
plt.grid()
plt.ylim([0.6,1.1])
plt.xticks(np.linspace(-40,40,5))
plt.xlim([-40,40])
plt.yticks([0.6,0.8,1.0],color='white')
plt.subplot(2,3,4)
xx       = x['x_num'][2][0]
xx       = xx.astype(float)
plt.plot(np.squeeze(xx),np.squeeze(x['y_num'][2][0])/np.squeeze(x['y_num'][2][0])[7], 'C0-',label='Model',linewidth=2)
novo_y = np.squeeze(x['y_mod'][2][0])/np.squeeze(x['y_mod'][2][0])[7]
novo_error = novo_y*(  (np.squeeze(x['err_mod'][2][0])/np.squeeze(x['y_mod'][2][0]))**2  + (np.squeeze(x['err_mod'][2][0])[7]/np.squeeze(x['y_mod'][2][0])[7] )**2 )**0.5
xx       = x['x_mod'][2][0]
xx       = xx.astype(float)
plt.errorbar(np.squeeze(xx),novo_y,novo_error,color='C1',fmt='o',barsabove=True,linestyle='',capsize=5,elinewidth=1)
plt.ylabel('$\eta_T$ [-]')
plt.xlabel('$\gamma$ [$^\circ$]')
plt.grid()
plt.ylim([0.6,1.1])
plt.xticks(np.linspace(-40,40,5))
plt.xlim([-40,40])
plt.subplots_adjust(left=0.1, bottom=0.15, right=0.95, top=0.85, wspace=0.1, hspace=0.15)
#%% Save figure
# plt.savefig('Figures/fig12.png',dpi=300)