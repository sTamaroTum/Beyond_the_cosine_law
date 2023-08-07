#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  2 09:51:00 2023

@author: saimon
"""

import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt  
plt.rcParams['text.usetex'] = True
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Helvetica"],
})
plt.close("all")
colors = np.array([[0,0,0],[0.4,0.4,0.4],[0.7,0.7,0.7]])
style  = np.array(["-",":"])
# Define function to compute C_T
def find_ct(x,*data):
    sigma,cd,cl_alfa,gamma,delta,k,cosMu,sinMu,tsr,theta = data
    CD = np.cos(np.deg2rad(delta))
    CG = np.cos(np.deg2rad(gamma))
    SD = np.sin(np.deg2rad(delta))
    SG = np.sin(np.deg2rad(gamma))

    a = (1- ( (1+np.sqrt(1-x-1/16*x**2*sinMu**2))/(2*(1+1/16*x*sinMu**2))) )

    I1 = -(cosMu*(tsr - CD*SG*k)*(a - 1))/2
    I2 = (np.pi*sinMu**2 + (np.pi*(CD**2*CG**2*SD**2*k**2 
                                   + 3*CD**2*SG**2*k**2 - 8*CD*tsr*SG*k 
                                   + 8*tsr**2))/12)/(2*np.pi)

    return (sigma*(cd+cl_alfa)*(I1) - sigma*cl_alfa*theta*(I2)) - x
# Define function to compute C_P
def find_cp(sigma,cd,cl_alfa,gamma,delta,k,cosMu,sinMu,tsr,theta,ct):

  a = 1-((1+np.sqrt(1-ct-1/16*sinMu**2*ct**2))/(2*(1+1/16*ct*sinMu**2)))
  SG = np.sin(np.deg2rad(gamma))
  CG = np.cos(np.deg2rad(gamma))                
  SD = np.sin(np.deg2rad(delta))  
  CD = np.cos(np.deg2rad(delta))  

  cp = sigma*((np.pi*cosMu**2*tsr*cl_alfa*(a - 1)**2 
               - (tsr*cd*np.pi*(CD**2*CG**2*SD**2*k**2 + 3*CD**2*SG**2*k**2 - 8*CD*tsr*SG*k + 8*tsr**2))/16 
               - (np.pi*tsr*sinMu**2*cd)/2 + (2*np.pi*cosMu*tsr**2*cl_alfa*theta*(a - 1))/3 
               + (2*np.pi*CD*cosMu*tsr*SG*cl_alfa*k*theta)/3 
               + (CD**2*cosMu**2*tsr*cl_alfa*k**2*np.pi*(a - 1)**2*(CG**2*SD**2 + SG**2))/(4*sinMu**2) 
               - (2*np.pi*CD*cosMu*tsr*SG*a*cl_alfa*k*theta)/3)/(2*np.pi))
  return cp

# Load variables
sigma           = 0.0416                        # rotor solidity        [-]
cd              = 0.004                         # drag coefficient      [-]
c_l_alpha       = 4.796                         # lift slope            [1/rad]
beta            = -3.177                        # blade twist angle     [deg]
gamma_array     = np.linspace(-30,30,61)        # rotor yaw angle       [deg]
delta           = -5                            # rotor tilt angle      [deg]
tsr             = 8                             # tip speed ratio       [-]
theta_arr       = np.deg2rad(np.array([1,5,8])) # blade pitch angle     [deg]
k_arr           = np.array([0, 0.3])            # inflow shear (linear) [-]
# solve for C_T with initial condition x0
x0 = 0.6

idx0 = np.where(gamma_array==0)
plt.figure(figsize=(7, 3.0), dpi=120)

c = 0
for k in k_arr:
    cc = 0
    for theta_p in theta_arr:
        ct = np.zeros(np.size(gamma_array))
        cp = np.zeros(np.size(gamma_array))
        theta = theta_p + np.deg2rad(beta)
        ccc  = 0
        for gamma in gamma_array:
          # define total misalignment angle mu               
          mu = np.arccos(np.cos(np.deg2rad(gamma))*np.cos(np.deg2rad(delta)))
          data = (sigma,cd,c_l_alpha,gamma,delta,k,np.cos(mu),np.sin(mu),tsr,theta)
          ct[ccc] = fsolve(find_ct, x0,args=data)
          # get C_P
          cp[ccc] = find_cp(sigma,cd,c_l_alpha,gamma,delta,k,np.cos(mu),np.sin(mu),tsr,theta,ct[ccc])
          ccc += 1
        plt.subplot(1,2,1)
        if c == 0:
            plt.plot(gamma_array,ct/ct[idx0],color=colors[cc,:],linestyle=style[c],label=r'$\theta_p=$' + str(np.rad2deg(theta_p)) + '$^\circ$')
        else:
            plt.plot(gamma_array,ct/ct[idx0],color=colors[cc,:],linestyle=style[c])           
        plt.subplot(1,2,2)
        if cc == 0:
            plt.plot(gamma_array,cp/cp[idx0],color=colors[cc,:],linestyle=style[c],label=r'$k=$' + str(k))
        else:
            plt.plot(gamma_array,cp/cp[idx0],color=colors[cc,:],linestyle=style[c])            
        cc += 1
    c += 1

plt.subplot(1,2,1)
plt.ylabel('$C_T/C_{T,0}$ $[-]$',fontsize=16)
plt.legend(ncol=3,handlelength=1,columnspacing=0.5,frameon=False,bbox_to_anchor = (2,1.35),fontsize=16)
plt.subplot(1,2,2)    
plt.ylabel('$C_P/C_{P,0}$ $[-]$',fontsize=16)
plt.legend(ncol=1,handlelength=1,columnspacing=0.5,frameon=True,fontsize=16)

letters = np.array(["(a)","(b)"])
for i in np.arange(2):
    plt.subplot(1,2,i+1)
    plt.xlabel('$\gamma$ [$^\circ$]',fontsize=16)
    plt.xlim([-30,30])
    plt.xticks(np.linspace(-30,30,7),fontsize=16)
    plt.ylim([0.6,1.01])
    plt.yticks(np.linspace(0.6,1,5),fontsize=16)
    plt.grid()
    plt.text(23.7,0.96,letters[i],fontsize=16)

plt.subplots_adjust(left=0.12, bottom=0.25, right=0.95, top=0.8, wspace=0.4, hspace=0.3)
plt.savefig('Figures/fig5.png',dpi=300)
