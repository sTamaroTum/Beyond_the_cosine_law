# Import packages
import numpy as np
import warnings
from scipy.optimize import minimize
from floris.tools import FlorisInterface
warnings.filterwarnings("ignore", category=RuntimeWarning) 
from scipy.io import loadmat
import matplotlib.pyplot as plt
plt.close("all")
# Define function to get power and thrust from wind farm through modified floris
def get_ct_power(x,yaw,wind_direction,nD):
    
    fi         = FlorisInterface("inputs/gch.yaml")
    D          = 130
    layout_xxx = [0, nD * D]
    layout_yyy = [0, 0]
    fi.reinitialize(wind_directions=[wind_direction],layout_x=layout_xxx, layout_y=layout_yyy)

    turb_type                 = fi.floris.farm.turbine_type[0]
    turb_type_derated         = []   
    turb_type["tsr"]          = x[0]
    turb_type["theta"]        = x[1]
    turb_type["turbine_type"] = 'WT0'
    turb_type_derated.append(turb_type.copy()) 
    
    turb_type["tsr"]          = 8.27071841e+00 # SET TSR
    turb_type["theta"]        = 1.13652081e+00 # SET Pitch
    turb_type["turbine_type"] = 'WT1'
    turb_type_derated.append(turb_type.copy())
    fi.reinitialize(turbine_type=turb_type_derated)

    yaw_angles        = np.zeros((1,1,2))
    yaw_angles[0,0,:] = np.array([yaw,0])
    fi.calculate_wake(yaw_angles=yaw_angles)
    thrust_coeff      = fi.get_turbine_Cts()
    p                 = np.array(fi.get_turbine_powers())

    return thrust_coeff[0], p[0]
# Define function to optimize
def optimization_f(x,yaw,wind_direction,nD):
    fi = FlorisInterface("inputs/gch.yaml")
    D = 130
    layout_xxx = [0, nD * D]
    layout_yyy = [0, 0]
    fi.reinitialize(wind_directions=[wind_direction],layout_x=layout_xxx, layout_y=layout_yyy)

    turb_type                 = fi.floris.farm.turbine_type[0]
    turb_type_derated         = []
    turb_type["tsr"]          = x[0]
    turb_type["theta"]        = x[1]
    turb_type["turbine_type"] = 'WT0'
    turb_type_derated.append(turb_type.copy()) 

    turb_type["tsr"]          = 8.20956689 # SET TSR
    turb_type["theta"]        = 1.08507142 # SET Pitch angle
    turb_type["turbine_type"] = 'WT1'
    turb_type_derated.append(turb_type.copy())
    fi.reinitialize(turbine_type=turb_type_derated)

    yaw_angles = np.zeros((1,1,2))
    yaw_angles[0,0,:] = np.array([yaw,0])
    fi.calculate_wake(yaw_angles=yaw_angles)  
    p=np.array(fi.get_turbine_powers())

    return -(p[0])
#%%
# Define constants
bnds      = ((3,10),(-7.5,10)) # Optimization bounds
x0        = [8.2,1.0602]       # Initial condition for optimization       
nD        = 5                  # Turbine spacing [D]
nTurbines = 2                  # Number of windturbines
D = 130                        # Rotor diameter [m]
R = D/2                        # Rotor radius [m]
distance_wt = nD*D             # distance between turbines
wDirection = 270               # wind direction [deg]
gamma_array = np.concatenate((np.linspace(0,30,61),np.linspace(-0.0001,-30,61))) # array of yaw angles to optimize for
# Allocate arrays for optimal control
x_trust_constr          = np.zeros((gamma_array.size,2))
max_trust_constr        = np.zeros(gamma_array.size)
i_ct                    = np.zeros((gamma_array.size,nTurbines))
i_pow                   = np.zeros((gamma_array.size,nTurbines))
# Allocate arrays for standard control
i_ct0                   = np.zeros((gamma_array.size))
wf_pow_standard_control = np.zeros((gamma_array.size))
pitch_standard_control  = np.zeros((gamma_array.size))
tsr_standard_control    = np.zeros((gamma_array.size))
i_ct_standard_control   = np.zeros((gamma_array.size))
#
conto = 0 # counter
for yaw in gamma_array:    
    print('Optimizing for gamma = ' + str(yaw) + ' deg')
    res = minimize(optimization_f, x0,args=(yaw,wDirection,nD),bounds=bnds,method='Nelder-Mead',options={'ftol':1e-8,'maxiter':80000})
    # Save solution (optimize for pitch, tsr and yaw)
    max_trust_constr[conto] = -res.fun
    x_trust_constr[conto,:] = res.x
    # get power and thrust solution from optimal solution
    i_ct[conto,:]  = get_ct_power(res.x,yaw,wDirection,nD)[0]
    i_pow[conto,:] = get_ct_power(res.x,yaw,wDirection,nD)[1]
    # Get solution for standard floris control (optimize only for yaw)
    wf_pow_standard_control[conto] = optimization_f(np.array([x_trust_constr[0,0]*np.cos(np.deg2rad(yaw))**0.6266,x_trust_constr[0,1]]),yaw,wDirection,nD)
    pitch_standard_control[conto]  = x_trust_constr[0,1]
    tsr_standard_control[conto]    = x_trust_constr[0,0]*np.cos(np.deg2rad(yaw))**0.6266
    i_ct_standard_control[conto]   = get_ct_power(np.array([x_trust_constr[0,0]*np.cos(np.deg2rad(yaw))**0.6266,x_trust_constr[0,1]]),yaw,wDirection,nD)[0]
    #
    if conto < len(gamma_array)-1:
        if (gamma_array[conto])*(gamma_array[conto+1]) < 0:
            x0 = [8.2,1.0602]#,6,0,0]
        else:    
            x0 =res.x
    else:
        x0 =res.x
    conto += 1        
# Sort by yaw (incremental)
idx = np.argsort(gamma_array)
gamma_array2 = gamma_array[idx]
#%% Plot
plt.figure(figsize=(9,4))
plt.subplot(2,2,1)
plt.plot(gamma_array2,tsr_standard_control[idx],'-',label='Standard')
plt.plot(gamma_array2,x_trust_constr[idx,0],'--',label='Optimal')
plt.ylabel(r'$\lambda$ [-]')
plt.legend(ncol=2,frameon=False,bbox_to_anchor=(1.85,1.6))
plt.ylim([7.5,8.5])
plt.yticks([7.5,8,8.5])
plt.grid()
plt.xlim([-30,30])
plt.xticks([-30,-15,0,15,30],color='w')
plt.text(24,8.3,'(a)')
plt.subplot(2,2,2)
plt.plot(gamma_array2,pitch_standard_control[idx],'-',label='Model')
plt.plot(gamma_array2,x_trust_constr[idx,1],'--',label='Model')
plt.ylabel(r'$\theta_p$ [deg]')
plt.ylim([0,1.5])
plt.yticks([0,0.5,1,1.5])
plt.grid()
plt.xlim([-30,30])
plt.xticks([-30,-15,0,15,30],color='w')
plt.text(23,1.2,'(b)')
plt.subplot(2,2,3)
plt.plot(gamma_array2,(i_ct[idx,0]/(i_ct_standard_control[idx])-1)*100,':k',label='Model')
plt.ylabel(r'$\Delta T$ [\%]')
plt.ylim([0,20])
plt.grid()
plt.xlim([-30,30])
plt.xticks([-30,-15,0,15,30])
plt.xlabel(r'$\gamma$ [$^\circ$]')
plt.text(24,16,'(c)')
plt.subplot(2,2,4)
plt.plot(gamma_array2,(i_pow[idx,0]/(-wf_pow_standard_control[idx])-1)*100,':k',label='Model')
plt.ylim([0,4])
plt.grid()
plt.ylabel(r'$\Delta P$ [\%]')
plt.xlim([-30,30])
plt.xticks([-30,-15,0,15,30])
plt.xlabel(r'$\gamma$ [$^\circ$]')
plt.text(23,3.25,'(d)')
plt.subplots_adjust(left=0.1, bottom=0.2, right=0.95, top=0.85, wspace=0.3, hspace=0.3)
#%% Save figure
# plt.savefig('../Figures/fig14.png',dpi=300)
