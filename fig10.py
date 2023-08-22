import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
plt.close("all")
x = loadmat('exp_data/Inflow_All3.mat')
from scipy.optimize import curve_fit
def func(x,a):
    return 1 - a*x/0.55
plt.figure(1,figsize=(9,4.0))
#%% Plot mean velocity
plt.subplot(1,2,1)
plt.plot([0,10],[0.5,0.5],'k',linewidth=0.75,zorder=0)
plt.plot([0,10],[0,0],'--k',linewidth=0.75,zorder=0)
plt.plot([0,10],[-0.5,-0.5],'k',linewidth=0.75,zorder=0)
xx = x['x_num'][0][0]
xx = xx.astype(float)
plt.scatter(np.squeeze(xx)[::2],-1*(np.squeeze(x['y_num'])[0][0]-0.75)[::2], 20,'C0',marker='o',linewidth=2,label='Low-TI',zorder=10)
# Fit linear shear
yL1 = np.linspace(-0.55,0.55,3)
P1  = np.polyfit(-1*(np.squeeze(x['y_num'])[0][0]-0.75),np.squeeze(xx),1)
plt.plot(np.polyval(P1,yL1),yL1, 'C0:',label='Lin. shear $k=-0.01$',linewidth=2)
xx = x['x_num'][1][0]
xx = xx.astype(float)
plt.scatter(np.squeeze(xx)[::2],-1*(np.squeeze(x['y_num'])[1][0]-0.75)[::2], 20, 'C1',marker='s',linewidth=2,label='Mod-TI',zorder=10)
# Fit linear shear
xL2 = np.linspace(0.86,1.11,31)
P2  = np.polyfit(np.squeeze(xx),-1*(np.squeeze(x['y_num'])[1][0]-0.75),1)
xK2 = -1*(np.squeeze(x['y_num'])[1][0]-0.75)
popt1, pcov = curve_fit(func,xK2*1.1,np.squeeze(xx),p0=0.2)
plt.plot(xL2,np.polyval(P2,xL2), 'C1:',label='Lin. shear $k=$' + str(np.round(popt1[0],2)),linewidth=2)
xx = x['x_num'][5][0]
xx = xx.astype(float)
plt.scatter(np.squeeze(xx)[::2],-1*(np.squeeze(x['y_num'])[5][0]-0.75)[::2], 20,'C2',marker='^',linewidth=2,label='High-TI',zorder=10)
# Fit linear shear
xL3 = np.linspace(0.8,1.15,31)
P3  = np.polyfit(np.squeeze(xx),-1*(np.squeeze(x['y_num'])[5][0]-0.75),1)
xK2 = -1*(np.squeeze(x['y_num'])[5][0]-0.75)
popt2, pcov = curve_fit(func,xK2*1.1,np.squeeze(xx),p0=0.2)
plt.plot(xL3,np.polyval(P3,xL3), 'C2:',label='Lin. shear $k=$' + str(np.round(popt2[0],2)),linewidth=2)
plt.legend(loc='upper center', bbox_to_anchor=(1, 1.375),handlelength=0.75,ncol=3,fancybox=False,frameon=False, shadow=False,columnspacing=0.7)
plt.xlim([0.7,1.2])
plt.ylim([0.75,-0.75])
plt.ylabel('$z_g$ [$D$]')
plt.xlabel(r'$u/u_{{pitot}}$ [-]')
plt.grid()
plt.text(1.15,-0.6,'(a)')
#%% Plot turbulence intensity
plt.subplot(1,2,2)
plt.plot([0,50],[1.25,1.25],'k',linewidth=0.75,zorder=0)
plt.plot([0,50],[0.75,0.75],'--k',linewidth=0.75,zorder=0)
plt.plot([0,50],[0.25,0.25],'k',linewidth=0.75,zorder=0)
xx = x['x_num'][6][0]
xx = xx.astype(float)
plt.scatter(np.squeeze(xx)[::2],np.squeeze(x['y_num'])[6][0][::2], 20,'C0',marker='o',linewidth=2,zorder=10)
xx = x['x_num'][8][0]
xx = xx.astype(float)
plt.scatter(np.squeeze(xx)[::2],np.squeeze(x['y_num'])[8][0][::2], 20,'C1',marker='s',label='Lin. shear $k=0$',linewidth=2,zorder=10)
xx = x['x_num'][12][0]
xx = xx.astype(float)
plt.scatter(np.squeeze(xx)[::2],np.squeeze(x['y_num'])[12][0][::2], 20,'C2',marker='^',label='Lin. shear $k=0$',linewidth=2,zorder=10)
plt.yticks([])
plt.xlim([0,20])
plt.xticks([0,5,10,15,20])
plt.ylim([0,1.5])
plt.xlabel('$TI$ [\%]')
plt.grid()
plt.text(17.5,1.35,'(b)')
plt.subplots_adjust(left=0.15, bottom=0.2, right=0.95, top=0.8, wspace=0.1, hspace=0)
# plt.savefig('Figures/fig10.png',dpi=300)