#%%
import numpy as np
import matplotlib.pyplot as plt
from floris.tools import FlorisInterface

plt.close("all")
plt.rc( 'text', usetex=True ) 
plt.rc('font',family = 'sans-serif',  size=18)
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Helvetica"],
})
# Define function to get power and thrust from wind farm through modified floris
def get_ct_power(x,wind_direction,nD):
    fi = FlorisInterface("inputs/gch.yaml")
    D = 130
    layout_xxx = [0, nD * D]
    layout_yyy = [0, 0]

    fi.reinitialize(wind_directions=[wind_direction],layout_x=layout_xxx, layout_y=layout_yyy)
    
    turb_type = fi.floris.farm.turbine_type[0]
    turb_type_derated = []
    
    turb_type["tsr"]          = x[0]
    turb_type["theta"]        = x[1]
    turb_type["turbine_type"] = 'WT0'
    turb_type_derated.append(turb_type.copy())
    
    turb_type["tsr"]          = 8.20949995#x[3]
    turb_type["theta"]        = 1.08500285#x[4]
    turb_type["turbine_type"] = 'WT1'
    turb_type_derated.append(turb_type.copy())
    
    fi.reinitialize(turbine_type=turb_type_derated)
    yaw_angles        = np.zeros((1,1,2))
    yaw_angles[0,0,:] = np.array([x[2],0])
    fi.calculate_wake(yaw_angles=yaw_angles)    
    thrust_coeff  = fi.get_turbine_Cts()
    p=np.array(fi.get_turbine_powers()) 

    return thrust_coeff, p

#%% Load results of optimization with modified floris
nD = 5
i_ct                 = np.load('data_floris/res_' + str(int(nD)) + 'D/4d_i_ct.npy')
i_ct0                = np.load('data_floris/res_' + str(int(nD)) + 'D/4d_i_ct0.npy')
idx                  = np.load('data_floris/res_' + str(int(nD)) + 'D/4d_idx.npy')
i_pow                = np.load('data_floris/res_' + str(int(nD)) + 'D/4d_i_pow.npy')
max_trust_constr     = np.load('data_floris/res_' + str(int(nD)) + 'D/4d_max_trust_constr.npy')
wind_directions_plot = np.load('data_floris/res_' + str(int(nD)) + 'D/4d_direzioni_vento_plottabili.npy')
x_trust_constr       = np.load('data_floris/res_' + str(int(nD)) + 'D/4d_x_trust_constr.npy')

#%% Load results of optimization with standard floris
floris_p_o           = np.load('data_floris/res_' + str(int(nD)) + 'D/flo_opt_i_pow.npy')
floris_ct_o          = np.load('data_floris/res_' + str(int(nD)) + 'D/flo_opt_i_ct.npy')
floris_gamma_o       = np.load('data_floris/res_' + str(int(nD)) + 'D/flo_opt_i_x.npy')
floris_directions    = np.load('data_floris/res_' + str(int(nD)) + 'D/flo_direzioni_vento.npy')
optimal_pitch        = np.ones(len(floris_p_o))*1.08500285
optimal_tsr          = 8.20949995*np.cos(np.deg2rad(floris_gamma_o))**(1.88/3)

#%% Get optimal power and ct from modified floris
optimum_ct_floris    = np.zeros((len(floris_directions),2))
optimum_power_floris = np.zeros((len(floris_directions),2))
conto = 0
for wDirection in floris_directions:
    optimum_ct_floris[conto,:]    = get_ct_power(np.array([optimal_tsr[conto],optimal_pitch[conto],floris_gamma_o[conto]]),wDirection,nD)[0]
    optimum_power_floris[conto,:] = get_ct_power(np.array([optimal_tsr[conto],optimal_pitch[conto],floris_gamma_o[conto]]),wDirection,nD)[1]
    conto += 1

#%% Load results of greedy cointrol obtained with standard floris
floris_p_g        = np.load('data_floris/res_' + str(int(nD)) + 'D/flo_greedy_i_pow.npy')
floris_ct_g       = np.load('data_floris/res_' + str(int(nD)) + 'D/flo_greedy_i_ct.npy')
floris_gamma_g    = np.zeros(len(floris_p_g))

#%% Get power and ct from greedy case with TSR 8.2094 and Pitch 1.085
ct_greedy_floris = np.zeros((len(floris_directions),2))
pow_greedy_floris = np.zeros((len(floris_directions),2))
conto = 0
for wDirection in floris_directions:
    ct_greedy_floris[conto,:]  = get_ct_power(np.array([8.20949995,1.08500285,0]),wDirection,nD)[0]
    pow_greedy_floris[conto,:] = get_ct_power(np.array([8.20949995,1.08500285,0]),wDirection,nD)[1]
    conto += 1

# Sort wind directions in incremental order
idxF = np.argsort(floris_directions[0:-1])
wind_directions_plot_FLORIS = floris_directions[idxF]

# Apply power losses
max_trust_constr     = max_trust_constr*0.9175980513708618
i_pow                = i_pow*0.9175980513708618
optimum_power_floris = optimum_power_floris*0.9175980513708618
pow_greedy_floris    = pow_greedy_floris*0.9175980513708618
#%% Plot
plt.figure(figsize=(15,6))
plt.subplot(2,3,1)
plt.plot(wind_directions_plot_FLORIS,optimal_tsr[idxF],'-',label='Floris opt.')
plt.plot(wind_directions_plot,x_trust_constr[idx,0],'--',label='Model')
plt.plot(np.array([240,300]),np.array([8.20949995,8.20949995]),':C2')
plt.ylabel(r'$\lambda$ [-]')
plt.ylim([7.5,8.5])

plt.subplot(2,3,2)
plt.plot(np.array([240,300]),np.array([1.08500285,1.08500285]),':C2',label='Greedy')
plt.plot(wind_directions_plot_FLORIS,optimal_pitch[idxF],'-',label='Opt. (FLORIS)')
plt.plot(wind_directions_plot,x_trust_constr[idx,1],'--',label='Opt. (Model)')
plt.ylabel(r'$\theta_p$ [$^\circ$]')
plt.ylim([-1,1.5])
plt.legend(bbox_to_anchor=(1.5,1.325),ncol=3,frameon=False)

plt.subplot(2,3,3)
plt.plot(wind_directions_plot_FLORIS,np.abs(floris_gamma_o[idxF]),'-',label='Floris opt.')
plt.plot(wind_directions_plot,np.abs(x_trust_constr[idx,2]),'--',label='Model')
plt.plot(np.array([240,300]),np.array([0,0]),':C2')
plt.ylabel(r'$|\gamma|$ [$^\circ$]')
plt.ylim([-5,30])

plt.subplot(2,3,6)
plt.plot(wind_directions_plot_FLORIS,((np.sum(optimum_power_floris[idxF,:],axis=1)/np.sum(pow_greedy_floris[idxF,:],axis=1))-1)*100,'-',label='Floris opt.')
podenza = np.interp(wind_directions_plot_FLORIS,wind_directions_plot,((max_trust_constr[idx])))
plt.plot(wind_directions_plot_FLORIS,((podenza/np.sum(pow_greedy_floris[idxF,:],axis=1))-1)*100,'--',label='Model')
plt.plot(np.array([240,300]),np.array([0,0]),'C2:',zorder=-1)
plt.ylabel(r'$\Delta P_{\textrm{TOT}}$ [\%]')
plt.xlabel(r'$\Phi$ [$^\circ$]')

plt.subplot(2,3,4)
plt.plot(wind_directions_plot_FLORIS,((optimum_power_floris[idxF,0]/(pow_greedy_floris[idxF,0]))-1)*100,'C0-')#,label='FLORIS (WT1)')
podenza = np.interp(wind_directions_plot_FLORIS,wind_directions_plot,((i_pow[idx,0])))
plt.plot(wind_directions_plot_FLORIS,((podenza/(pow_greedy_floris[idxF,0]))-1)*100,'C1--')#,label='Model (WT1)')
plt.plot(np.array([240,300]),np.array([0,0]),'C2:',zorder=-1)
plt.xlim([240,300])
plt.plot(np.array([200,210]),np.array([0,0]),'-k',label='WT1')
plt.plot(np.array([200,210]),np.array([0,0]),'k--',label='WT2')
plt.ylabel(r'$\Delta P_1$ [\%]')
plt.xlabel(r'$\Phi$ [$^\circ$]')

plt.subplot(2,3,5)
plt.plot(wind_directions_plot_FLORIS,((optimum_power_floris[idxF,1]/(pow_greedy_floris[idxF,1]))-1)*100,'C0-')#,label='FLORIS (WT2)')
podenza = np.interp(wind_directions_plot_FLORIS,wind_directions_plot,((i_pow[idx,1])))
plt.plot(wind_directions_plot_FLORIS,((podenza/(pow_greedy_floris[idxF,1]))-1)*100,'C1--')#,label='Model (WT2)')
plt.plot(np.array([240,300]),np.array([0,0]),'C2:',zorder=-1)
plt.xlim([240,300])
plt.plot(np.array([200,210]),np.array([0,0]),'-k',label='WT1')
plt.plot(np.array([200,210]),np.array([0,0]),'k--',label='WT2')
plt.ylabel(r'$\Delta P_2$ [\%]')
plt.xlabel(r'$\Phi$ [$^\circ$]')

for i in np.arange(6):
    plt.subplot(2,3,i+1)
    plt.xlim([250,290])
#%% Add LES results
from scipy.io import loadmat
data = loadmat('cp_ct_tables_iea_3mw/Cp_335.mat')
cp = np.squeeze(data['num'])
data = loadmat('cp_ct_tables_iea_3mw/Ct_335.mat')
ct = np.squeeze(data['num'])
data = loadmat('cp_ct_tables_iea_3mw/pitch_335.mat')
pitch = np.squeeze(data['num'])
data = loadmat('cp_ct_tables_iea_3mw/TSR_335.mat')
tsr = np.squeeze(data['num'])
data = loadmat('cp_ct_tables_iea_3mw/U_335.mat')
u = np.squeeze(data['num'])
del data
from scipy.interpolate import RegularGridInterpolator as rgi
interopolation_1 = rgi((u,tsr,pitch), cp*0.9175980513708618,method='linear')

### UPSTREAM
THR = np.load('LES_OPT/res2/THR.npy')
POW = np.load('LES_OPT/res2/POW.npy')
PHI = np.array([264.2608295227332,267.5,270,272.5,275.7391704772668])
### DOWNSTREAM
uuu = np.load('LES_OPT/res2/u_wake.npy')
phi2 = np.array([264.2608295227332,267.5,270,272.5,275.7391704772668])

cpp = np.zeros_like(uuu)
for i in np.arange(uuu.shape[0]):
    for j in np.arange(uuu.shape[1]):
        cpp[i,j] = interopolation_1(np.array([uuu[i,j],8.20949995,1.08500285]))

ppp = cpp*uuu**3*(0.5*1.19*np.pi*65**2)

#%% Add to subplots
plt.subplot(2,3,1)
plt.scatter(phi2,np.array([7.971624176830841, 7.827879563749586, 7.913594812836695,7.837035641612392,7.976773533259332]),color='C0',label='floris',marker='o')
plt.scatter(phi2,np.array([8.124163592825822,8.030666241654806,8.01227337846487,8.039962163186885,8.132238856257297]),color='C1',label='floris',marker='o')
plt.subplot(2,3,2)
plt.scatter(phi2,np.array([1.08500285,1.08500285,1.08500285,1.08500285,1.08500285]),color='C0',label='floris',marker='o')
plt.scatter(phi2,np.array([0.8026487336572626,0.2887554470923204,0.3839288903095631,0.3207476002821303,0.8367043355324543]),color='C1',label='floris',marker='o')
plt.subplot(2,3,3)
plt.scatter(phi2,np.abs(np.array([-17.41470904098508,-22.049588305395208,-19.417331380641855,21.783805647385744,17.220567756031347])),color='C0',label='floris',marker='o')
plt.scatter(phi2,np.abs(np.array([-20.193898143602926,-26.404169711739435,-26.4907131832505,25.942086790841064,19.92084003228294])),color='C1',label='floris',marker='o')
plt.subplot(2,3,4)
plt.scatter(phi2,(POW[:,1]/POW[2,0]-1)*100,color='C0',label='floris')
plt.scatter(phi2,(POW[:,2]/POW[2,0]-1)*100,color='C1',label='model')
plt.subplot(2,3,5)
plt.scatter(phi2,(ppp[1,:]/ppp[0,:]-1)*100,color='C0',label='floris',marker='o')
plt.scatter(phi2,(ppp[2,:]/ppp[0,:]-1)*100,color='C1',label='model',marker='o')
plt.subplot(2,3,6)
plt.scatter(phi2,((ppp[1,:]+POW[:,1]/1)/(ppp[0,:]+POW[2,0]/1)-1)*100,color='C0',label='floris')
plt.scatter(phi2,((ppp[2,:]+POW[:,2]/1)/(ppp[0,:]+POW[2,0]/1)-1)*100,color='C1',label='model')


for i in np.arange(6):
    plt.subplot(2,3,i+1)
    plt.xlim([245,295])
    if i < 3:
        plt.xticks([240,255,270,285,300],color='w')
    else:
        plt.xticks([240,255,270,285,300],color='k')
    plt.grid()

#%% Add letters
plt.subplot(2,3,1)
plt.ylim([7.6,8.4])
plt.text(290,8.3,'(a)')
plt.subplot(2,3,2)
plt.text(290,1.3,'(b)')
plt.ylim([0,1.5])
plt.subplot(2,3,3)
plt.text(290,26,'(c)')
plt.yticks([0,10,20,30])
plt.ylim([-1,30])
plt.subplot(2,3,4)
plt.text(290,-3,'(d)')
plt.ylim([-20,1])
plt.subplot(2,3,5)
plt.text(290,67.5,'(e)')
plt.ylim([-1,80])
plt.yticks([0,20,40,60,80])
plt.subplot(2,3,6)
plt.text(290,8.5,'(f)')
plt.ylim([-0.15,10])
plt.subplots_adjust(left=0.1, bottom=0.15, right=0.95, top=0.9, wspace=0.25, hspace=0.2)

#%% Save figure
plt.savefig('../Figures/fig15.png',dpi=300)
