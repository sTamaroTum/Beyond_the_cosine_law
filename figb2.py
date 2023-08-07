import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
plt.close("all")
x = loadmat('exp_data/opt_7_param_Derating_ExpNumComparison_pitch.mat')
legend_array0 = np.array([r"iso-$\lambda$, $P_d=100\%$",r"iso-$\lambda$, $P_d=90\%$",r"iso-$\lambda$, $P_d=80\%$",r"iso-$\lambda$, $P_d=70\%$",r"iso-$\lambda$, $P_d=60\%$",r"iso-$\lambda$, $P_d=50\%$",r"min-$C_T$, $P_d=80\%$",r"min-$C_T$, $P_d=50\%$"])
legend_array = np.concatenate((np.squeeze(legend_array0),np.squeeze(legend_array0)))
letters = np.array(["(i)","(j)","(k)","(l)","(m)","(n)","(o)","(p)","(a)","(b)","(c)","(d)","(e)","(f)","(g)","(h)"])
letters = np.flip(letters)
plt.figure(1,figsize=(11,8))
for i in np.arange(8):
    plt.subplot(4,4,8-i)
    xx = x['x_num'][i][0]
    xx = xx.astype(float)
    plt.plot(np.squeeze(xx),np.squeeze(x['y_num'][i][0]), 'C1o',label='Model',linewidth=2)
    if i == 3:
        plt.ylabel(r'$\theta_p$ [$^\circ$]')
    if i == 7:
        plt.ylabel(r'$\theta_p$ [$^\circ$]')
    plt.grid()
    if (i==3) | (i==7):
        plt.yticks([0,4,8,12])
    else:
        plt.yticks([0,4,8,12],color='white')
    plt.xticks(np.linspace(-30,30,5),color='white',fontsize=0.002)
    plt.ylim([0,12])
    plt.xlim([-35,35])
    plt.title(legend_array[7-i],fontsize=18)
x = loadmat('exp_data/opt_7_param_Derating_ExpNumComparison_TSR.mat')
plt.figure(1,figsize=(14,6.5))
for i in np.arange(8):
    plt.subplot(4,4,8-i+8)
    xx = x['x_num'][i][0]
    xx = xx.astype(float)
    plt.plot(np.squeeze(xx),np.squeeze(x['y_num'][i][0]), 'C1o',label='Model',linewidth=2)
    if i == 3:
        plt.ylabel('$\lambda$ [-]')
        
    if i == 7:
        plt.ylabel('$\lambda$ [-]')
    plt.grid()
    plt.ylim([3,11])
    if (i==3) | (i==7):
        plt.yticks([3,5,7,9])
    else:
        plt.yticks([3,5,7,9],color='white')
    plt.ylim([3,10])
    if i < 4:
        plt.xticks(np.linspace(-30,30,5))
        plt.xlabel('$\gamma$ [$^\circ$]')
    else:
        plt.xticks(np.linspace(-30,30,5),color='white',fontsize=0.002)
    plt.xlim([-35,35])
    plt.title(legend_array[7-i],fontsize=18)
plt.subplots_adjust(left=0.1, bottom=0.125, right=0.95, top=0.925, wspace=0.1, hspace=0.3)
#%%
# plt.savefig('Figures/figb2.png',dpi=300)