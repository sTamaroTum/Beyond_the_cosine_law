import numpy as np
import matplotlib.pyplot as plt
plt.close("all")
#%% Define function to compute C_T
def find_ct(x,*data):
    sigma,cd,cl_alfa,gamma,delta,k,cosMu,sinMu,tsr,theta,mu = data
    CD = np.cos(np.deg2rad(delta))
    CG = np.cos(np.deg2rad(gamma))
    SD = np.sin(np.deg2rad(delta))
    SG = np.sin(np.deg2rad(gamma))
    a = (1- ( (1+np.sqrt(1-x-1/16*x**2*sinMu**2))/(2*(1+1/16*x*sinMu**2))) )    
    k_1s = -1*(15*np.pi/32*np.tan((mu+np.cos(mu)**2*np.sin(mu)*(x/2))/2));
    I1 = -(np.pi*cosMu*(tsr - CD*SG*k)*(a - 1) 
           + (CD*CG*cosMu*k_1s*SD*a*k*np.pi*(2*tsr - CD*SG*k))/(8*sinMu))/(2*np.pi)
    I2 = (np.pi*sinMu**2 + (np.pi*(CD**2*CG**2*SD**2*k**2 
                                   + 3*CD**2*SG**2*k**2 - 8*CD*tsr*SG*k 
                                   + 8*tsr**2))/12)/(2*np.pi)
    return (sigma*(cd+cl_alfa)*(I1) - sigma*cl_alfa*theta*(I2)) - x
# Define function to compute C_P
def find_cp(sigma,cd,cl_alfa,gamma,delta,k,cosMu,sinMu,tsr,theta,ct,mu):
    a = 1-((1+np.sqrt(1-ct-1/16*sinMu**2*ct**2))/(2*(1+1/16*ct*sinMu**2)))
    SG = np.sin(np.deg2rad(gamma))
    CG = np.cos(np.deg2rad(gamma))                
    SD = np.sin(np.deg2rad(delta))  
    CD = np.cos(np.deg2rad(delta))  
    k_1s = -1*(15*np.pi/32*np.tan((mu+np.cos(mu)**2*np.sin(mu)*(ct/2))/2));
    
    cp = sigma*((np.pi*cosMu**2*tsr*cl_alfa*(a - 1)**2 
                 - (tsr*cd*np.pi*(CD**2*CG**2*SD**2*k**2 + 3*CD**2*SG**2*k**2 - 8*CD*tsr*SG*k + 8*tsr**2))/16 
                 - (np.pi*tsr*sinMu**2*cd)/2 - (2*np.pi*cosMu*tsr**2*cl_alfa*theta)/3 
                 + (np.pi*cosMu**2*k_1s**2*tsr*a**2*cl_alfa)/4 
                 + (2*np.pi*cosMu*tsr**2*a*cl_alfa*theta)/3 + (2*np.pi*CD*cosMu*tsr*SG*cl_alfa*k*theta)/3 
                 + (CD**2*cosMu**2*tsr*cl_alfa*k**2*np.pi*(a - 1)**2*(CG**2*SD**2 + SG**2))/(4*sinMu**2) 
                 - (2*np.pi*CD*cosMu*tsr*SG*a*cl_alfa*k*theta)/3 
                 + (CD**2*cosMu**2*k_1s**2*tsr*a**2*cl_alfa*k**2*np.pi*(3*CG**2*SD**2 + SG**2))/(24*sinMu**2) 
                 - (np.pi*CD*CG*cosMu**2*k_1s*tsr*SD*a*cl_alfa*k)/sinMu 
                 + (np.pi*CD*CG*cosMu**2*k_1s*tsr*SD*a**2*cl_alfa*k)/sinMu 
                 + (np.pi*CD*CG*cosMu*k_1s*tsr**2*SD*a*cl_alfa*k*theta)/(5*sinMu) 
                 - (np.pi*CD**2*CG*cosMu*k_1s*tsr*SD*SG*a*cl_alfa*k**2*theta)/(10*sinMu))/(2*np.pi))
    
    return cp
#%%
delta = -5
sigma = 0.0416
cd              = 0.0051                         # drag coefficient      [-]
c_l_alpha       = 4.7662                         # lift slope            [1/rad]
beta            = -3.336                         # blade twist angle     [deg]
from scipy.optimize import fsolve
from scipy.io import loadmat
goal_etaP   = np.zeros((28*4))
goal_ct     = np.zeros((28*4))
tsr_in      = np.zeros((28*4))
pitch_in    = np.zeros((28*4))
gamma_in    = np.zeros((28*4))
shear_in    = np.zeros((28*4))
goal_eta_T  = np.zeros((28*4))
s1 = loadmat('num_data/tsr8/tsr_8_rigid.mat')
s2 = loadmat('num_data/tsr9_5/tsr_9_5_rigid.mat')
s3 = loadmat('num_data/shear01/shear_01_rigid.mat')
s4 = loadmat('num_data/shear03/shear_03_rigid.mat')
goal_etaP[0:28]      = np.squeeze(s1['dati_pow'])
goal_etaP[28*1:28*2] = np.squeeze(s2['dati_pow'])
goal_etaP[28*2:28*3] = np.squeeze(s3['dati_pow'])
goal_etaP[28*3:28*4] = np.squeeze(s4['dati_pow'])
goal_ct[0:28]      = np.squeeze(s1['dati_ct'])
goal_ct[28*1:28*2] = np.squeeze(s2['dati_ct'])
goal_ct[28*2:28*3] = np.squeeze(s3['dati_ct'])
goal_ct[28*3:28*4] = np.squeeze(s4['dati_ct'])
tsr_in[0:28]      = np.squeeze(s1['dati_tsr'])
tsr_in[28*1:28*2] = np.squeeze(s2['dati_tsr'])
tsr_in[28*2:28*3] = np.squeeze(s3['dati_tsr'])
tsr_in[28*3:28*4] = np.squeeze(s4['dati_tsr'])
pitch_in[0:28]      = np.deg2rad(np.squeeze(s1['dati_theta']))
pitch_in[28*1:28*2] = np.deg2rad(np.squeeze(s2['dati_theta']))
pitch_in[28*2:28*3] = np.deg2rad(np.squeeze(s3['dati_theta']))
pitch_in[28*3:28*4] = np.deg2rad(np.squeeze(s4['dati_theta']))
gamma_in[0:28]      = np.squeeze(s1['dati_gamma'])
gamma_in[28*1:28*2] = np.squeeze(s2['dati_gamma'])
gamma_in[28*2:28*3] = np.squeeze(s3['dati_gamma'])
gamma_in[28*3:28*4] = np.squeeze(s4['dati_gamma'])
shear_in[0:28]      = np.squeeze(s1['dati_k'])
shear_in[28*1:28*2] = np.squeeze(s2['dati_k'])
shear_in[28*2:28*3] = np.squeeze(s3['dati_k'])
shear_in[28*3:28*4] = np.squeeze(s4['dati_k'])
for i in np.arange(len(goal_ct)):
    idx = np.where(( (gamma_in == 0) & (tsr_in == tsr_in[i]) & (np.round(pitch_in,4) == np.round(pitch_in[i],4)) & (shear_in == shear_in[i])))
    goal_eta_T[i] = goal_ct[i]/goal_ct[idx[0]]
#%%
gamma_array    = np.linspace(-30,30,7)
gamma_arrayF   = np.linspace(-30,30,61)
idx0           = np.where(gamma_arrayF == 0)
colors   = np.array(["#1f77b4","#ff7f0e","#2ca02c","#d62728"])
styles   = np.array(["-","--",":","-."])
marks    = np.array(["*","^","o","s"])
legend_a = np.array([r'$\theta_p=1.4^\circ$',r'$\theta_p=4.9^\circ$',r'$\theta_p=6.7^\circ$',r'$\theta_p=8.1^\circ$'])
plt.figure(1,figsize=(8.5,5.5))
plt.figure(2,figsize=(8.5,5.5))
for i in np.arange(4):
    p_counter = 0
    for p in  np.arange(4):
        eta_p_les = np.zeros(len(gamma_array))
        eta_t_les = np.zeros(len(gamma_array))
        c = 0
        for j in gamma_array:
            tsr          = tsr_in    [len(gamma_array)*4*i+len(gamma_array)*p+c]
            gamma        = gamma_in  [len(gamma_array)*4*i+len(gamma_array)*p+c]
            eta_p_les[c] = goal_etaP [len(gamma_array)*4*i+len(gamma_array)*p+c]
            eta_t_les[c] = goal_eta_T[len(gamma_array)*4*i+len(gamma_array)*p+c]
            theta        = pitch_in  [len(gamma_array)*4*i+len(gamma_array)*p+c]
            shear        = shear_in  [len(gamma_array)*4*i+len(gamma_array)*p+c]            
            c += 1
        cp = np.zeros(len(gamma_arrayF))
        ct = np.zeros(len(gamma_arrayF))
        c = 0
        for jj in gamma_arrayF:
            gamma = gamma_arrayF[c]
            mu    = np.arccos(np.cos(np.deg2rad(gamma))*np.cos(np.deg2rad(delta)))
            data  = (sigma,cd,c_l_alpha,gamma,delta,shear,np.cos(mu),np.sin(mu),tsr,theta+np.deg2rad(beta),mu)
            x0    = 0.6
            ct[c] = fsolve(find_ct, x0,args=data)
            cp[c] = find_cp(sigma,cd,c_l_alpha,gamma,delta,shear,np.cos(mu),np.sin(mu),tsr,theta+np.deg2rad(beta),ct[c],mu)
            c += 1
        eta_t_mod = ct/ct[idx0]        
        eta_p_mod = cp/cp[idx0]
        plt.figure(1)
        plt.subplot(2,2,i+1)
        if i == 0:
            plt.plot(gamma_arrayF,eta_t_mod,color=colors[p_counter],label=legend_a[p_counter],linestyle=styles[p_counter],linewidth=1.75)
        else:
            plt.plot(gamma_arrayF,eta_t_mod,color=colors[p_counter],linestyle=styles[p_counter],linewidth=1.75)
        plt.scatter(gamma_array,eta_t_les,40,color=colors[p_counter],marker=marks[p_counter],zorder=100,clip_on=False)
        plt.figure(2)
        plt.subplot(2,2,i+1)
        if i == 0:
            plt.plot(gamma_arrayF,eta_p_mod,color=colors[p_counter],label=legend_a[p_counter],linestyle=styles[p_counter],linewidth=1.75)
        else:
            plt.plot(gamma_arrayF,eta_p_mod,color=colors[p_counter],linestyle=styles[p_counter],linewidth=1.75)
            
        plt.scatter(gamma_array,eta_p_les,40,color=colors[p_counter],marker=marks[p_counter],zorder=100,clip_on=False)
                
        p_counter += 1 
    plt.figure(1)
    plt.subplot(2,2,i+1)
    plt.xlim([-30,30])
    plt.xticks(np.linspace(-30,30,7))
    plt.ylim([0.7,1.005])
    plt.yticks(np.linspace(0.7,1,4))
    plt.grid(axis='y')
    if (i == 0 or i == 2):
        plt.ylabel('$\eta_T$ $\mathrm{[-]}$')
    if i > 1:
        plt.xlabel(r'$\gamma$ $[^\circ]$')
    plt.figure(2)
    plt.subplot(2,2,i+1)
    plt.xlim([-30,30])
    plt.xticks(np.linspace(-30,30,7))
    plt.ylim([0.5,1.01])
    plt.yticks(np.linspace(0.5,1,6))
    plt.grid(axis='y')
    if (i == 0 or i == 2):
        plt.ylabel('$\eta_P$ [-]')
    if i > 1:
        plt.xlabel('$\gamma$ [$^\circ$]')  
plt.figure(1)
plt.subplot(2,2,1)
plt.text(23,0.965,'(a)')
plt.legend(ncol=4,frameon=False,handlelength=1.5,columnspacing=0.6,bbox_to_anchor=(0.06,0.925))
plt.subplot(2,2,2)
plt.text(23,0.965,'(b)')
plt.subplot(2,2,3)
plt.text(23,0.965,'(c)')
plt.subplot(2,2,4)
plt.text(23,0.965,'(d)')
plt.subplots_adjust(left=0.12, bottom=0.14, right=0.95, top=0.9, wspace=0.4, hspace=0.3)
plt.savefig('Figures/fig9.png',dpi=300)
plt.figure(2)
plt.subplot(2,2,1)
plt.text(23,0.94,'(a)')
plt.legend(ncol=4,frameon=False,handlelength=1.5,columnspacing=0.6,bbox_to_anchor=(0.06,0.925))
plt.subplot(2,2,2)
plt.text(23,0.94,'(b)')
plt.subplot(2,2,3)
plt.text(23,0.94,'(c)')
plt.subplot(2,2,4)
plt.text(23,0.94,'(d)')
plt.subplots_adjust(left=0.12, bottom=0.14, right=0.95, top=0.9, wspace=0.4, hspace=0.3)
plt.savefig('Figures/fig7.png',dpi=300)