import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
plt.close("all")
#%%
legend_array         = np.array([r"iso-$\lambda$, $P_d=100\%$",r"iso-$\lambda$, $P_d=90\%$",r"iso-$\lambda$, $P_d=80\%$",r"iso-$\lambda$, $P_d=70\%$",r"iso-$\lambda$, $P_d=60\%$",r"iso-$\lambda$, $P_d=50\%$",r"min-$C_T$, $P_d=80\%$",r"min-$C_T$, $P_d=50\%$"])
legend_array_conc    = np.concatenate((np.squeeze(legend_array),np.squeeze(legend_array)))
alphabet_array = np.array(["(c)","(c)","(c)","(c)","(d)","(d)","(d)","(d)","(a)","(a)","(a)","(a)","(b)","(b)","(b)","(b)"])
alphabet_array = np.flip(alphabet_array)
#%% Plot ETA_P
x = loadmat('exp_data/opt_7_param_Derating_ExpNumComparison_power.mat')
plt.figure(1,figsize=(11,8))
for i in np.arange(8):
    plt.subplot(4,4,8-i)
    xx       = x['x_num'][i][0]
    xx= xx.astype(float)
    plt.plot(np.squeeze(xx),np.squeeze(x['y_num'][i][0]), 'C0-',label='Model',linewidth=2)
    xx       = x['x_mod'][i][0]
    xx= xx.astype(float)
    plt.errorbar(np.squeeze(xx),np.squeeze(x['y_mod'][i][0]),np.squeeze(x['err_mod'][i][0]),color='C1',fmt='o',barsabove=True,linestyle='',capsize=5,elinewidth=1,label='Exp.')
    if i == 3:
        plt.ylabel(r'$\eta_P$ [-]')
    if i == 7:
        plt.ylabel(r'$\eta_P$ [-]')
    plt.grid()
    if i == 7:
        plt.legend(handlelength=1,loc='upper left',frameon=False,ncol=2,bbox_to_anchor=(1.4, 1.7),fancybox=True, shadow=False)
    plt.ylim([0.55,1.025])
    plt.xticks(np.linspace(-30,30,5),color='white',fontsize=0.002)
    if (i == 3) | (i==7):
        plt.yticks(np.linspace(0.6,1,3))
    else:
        plt.yticks(np.linspace(0.6,1,3),color='white')
    plt.xlim([-35,35])
    plt.title(legend_array_conc[7-i],fontsize=18)
#%% Plot ETA_T
plt.figure(1)
x = loadmat('exp_data/opt_7_param_Derating_ExpNumComparison_CT.mat')
for i in np.arange(8):
    plt.subplot(4,4,8-i+8)
    xx       = x['x_num'][i][0]
    xx= xx.astype(float)    
    idx = np.where(xx==0) 
    plt.plot(np.squeeze(xx),np.squeeze(x['y_num'][i][0])/np.squeeze(x['y_num'][i][0])[idx[1][0]], 'C0-',label='Model',linewidth=2)
    xx       = x['x_mod'][i][0]
    xx= xx.astype(float)  
    idx = np.where(xx==0)
    novo_y = np.squeeze(x['y_mod'][i][0])/np.squeeze(x['y_mod'][i][0])[idx[1][0]]
    novo_error = novo_y*( (np.squeeze(x['err_mod'][i][0])/np.squeeze(x['y_mod'][i][0]))**2  + (np.squeeze(x['err_mod'][i][0])[idx[1][0]]/np.squeeze(x['y_mod'][i][0])[idx[1][0]] )**2)**0.5
    plt.errorbar(np.squeeze(xx),novo_y,novo_error,color='C1',fmt='o',barsabove=True,linestyle='',capsize=5,elinewidth=1,label='Exp.')
    if (i == 3) | (i==7):
        plt.ylabel('$\eta_T$ [-]')
        plt.yticks(np.linspace(0.6,1,3))
    else:
        plt.yticks(np.linspace(0.6,1,3),color='white')
    if i < 4:
        plt.xlabel('$\gamma$ [$^\circ$]')
        plt.xticks(np.linspace(-30,30,5))
    else:
        plt.xticks(np.linspace(-30,30,5),color='white',fontsize=0.002)
    plt.grid()
    plt.ylim([0.55,1.1])
    plt.xlim([-35,35])
    plt.title(legend_array_conc[7-i],fontsize=18)
plt.subplots_adjust(left=0.1, bottom=0.125, right=0.95, top=0.9, wspace=0.1, hspace=0.3)
#%% Save figure
# plt.savefig('Figures/fig13.png',dpi=300)