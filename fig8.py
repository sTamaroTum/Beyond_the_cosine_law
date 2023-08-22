import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy import interpolate
plt.close("all")
#%% Load LES data
files = ['num_data/tsr8/tsr_8_rigid.mat','num_data/tsr9_5/tsr_9_5_rigid.mat','num_data/shear01/shear_01_rigid.mat']
#%% load constants
u_inf = np.array([9.5,7,9.5])       # free-stream velocity [m/s]
rho   = 1.17                        # air density  [kg m^-3]
R     = 65                          # rotor radius [m]
#%% pre allocate arrays
ct_array    = np.zeros((len(files),4))
eta_p_array = np.zeros((len(files),4))
tsr_array   = np.zeros(len(files))
gamma_les   = np.linspace(-30,30,7)
#%% Load results for 
c = 0
for filetto in files:
    A = loadmat(filetto)
    cp_les      = A['dati_pow']
    ct_les      = A['dati_ct']
    gamma_array = A['dati_gamma']
    tsr         = A['dati_tsr']
    tsr_array[c] = tsr[0]
    cc = 0   
    for i in np.arange(4):
        eta_p_array[c,cc] = ( (cp_les[i*7]) + (cp_les[i*7+6]) ) /2
        ct_array[c,cc]    = ( (ct_les[i*7]) + (ct_les[i*7+6]) ) /2/(0.5*rho*np.pi*R**2*u_inf[c]**2)
        cc += 1     
    c += 1         
#%% Interpolate results on ct_target    
ct_target  = np.array([0.3,0.4,0.5,0.6])
idx        = np.argsort(tsr_array)
tsr_lambda = np.sort(tsr_array)
eta_new    = np.zeros((len(ct_target),len(tsr_array)))
for i in np.arange(len(tsr_array)):
    f = interpolate.interp1d(ct_array[i,:], eta_p_array[i,:],kind='cubic',bounds_error=False)#,fill_value = 'extrapolate')
    eta_new[:,i] = f(ct_target)
marker_array = np.array(['^', 's', 'o', '*']) 
plt.close("all")
plt.figure(1,figsize=(6,2.75)) 
for i in np.arange(len(ct_target)):
    plt.scatter(tsr_array,eta_new[i,:],90,marker=marker_array[i],linewidth=1.5,label=r'$C_T=$' +str(np.round(ct_target[i],2)),zorder=10) 
plt.xlabel(r'$\lambda$ [-]')
plt.ylabel(r'$\overline{\eta}_{P,\gamma=\pm30^\circ}$ [-]')
plt.legend(ncol=4,bbox_to_anchor=(1.05,1.4),frameon=False,handlelength=1,labelspacing=0.3,columnspacing=0.8,handletextpad=0.1)
plt.ylim([0.6,0.8])
plt.xlim([7.5,10])
plt.grid()
plt.subplots_adjust(left=0.15, bottom=0.25, right=0.95, top=0.8, wspace=0.1, hspace=0.15)
#%% Save figure
# plt.savefig('Figures/fig8.png',dpi=300)
