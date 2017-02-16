# -*- coding: utf-8 -*-
"""
Created on Mon Jan 23 12:35:32 2017

@author: dfs1
"""

import numpy as np
from scipy.special import jn as besselj
#import CDSAXSfunctions as CD
#from multiprocessing import Pool
import time
import matplotlib.pyplot as plt

Intensity=np.loadtxt('A1Int1.txt')
Qr = np.loadtxt('A1Qr1.txt')
Qz = np.loadtxt('A1Qz1.txt')


ConeNumber = 1
Pitch = 120

CPAR=np.zeros([ConeNumber+1,3])

CPAR[0,0]= 16; CPAR[0,1]=10; CPAR[0,2]=1;
CPAR[1,0]=23; CPAR[1,1]=100; CPAR[1,2]=1;
#CPAR[2,0]=27; 

Disc=np.zeros([ConeNumber])
Disc[0]=10;
#Disc[1]=100;



DW = 1.
I0 = 1
Bk =1.
Pitch = 120



SLD2=1.05
SPAR=np.zeros([4]) 
SPAR[0]=DW; SPAR[1]=I0; SPAR[2]=Bk;SPAR[3]=SLD2;


def ConeSim(Qr,Qz,CPAR,ConeNumber,Disc,SPAR):
    H1=0;
    H2=0;
    Form=np.zeros([int(len(Qr[:,1])),int(len(Qr[1,:]))])
    
    for i in range(ConeNumber):
        H2=H2+CPAR[i,1]
        z=np.zeros(Disc[i]+1)
        stepsize=CPAR[i,1]/Disc[i]
        for ii in np.arange(1,int(Disc[i])+.001,1):
            z[ii,]=z[ii-1]+stepsize
        if i > 0:
            H1=H1+CPAR[i-1,1]
        R1=CPAR[i,0]
        R2=CPAR[i+1,0]
        if R1 ==R2:
            R1=R1+0.0000001
        Slope=(H2-H1)/(R2-R1)
        for ii in range(int(Disc[i])):
            RI1=(z[ii]-H1)/Slope+R1
            RI2=(z[ii+1]-H1)/Slope+R1
            fa=2*np.pi*RI1/Qr*besselj(1,Qr*RI1)*np.exp(1j*Qr*z[ii])
            fb=2*np.pi*RI2/Qr*besselj(1,Qr*RI2)*np.exp(1j*Qr*z[ii+1])
            Form=Form[:,:]+stepsize*(fa+fb)/2*CPAR[i,2]
        M=np.power(np.exp(-1*(np.power(Qr,2)+np.power(Qz,2))*np.power(SPAR[0],2)),0.5);
        Form=Form*M
        SimInt = np.power(abs(Form),2)*SPAR[1]+SPAR[2]
        return(SimInt,Form)

start_time = time.perf_counter()   
(SimInt,form)=ConeSim(Qr,Qz,CPAR,ConeNumber,Disc,SPAR)     
end_time=time.perf_counter()   
print(end_time-start_time)    
Simtest=np.power(abs(form),2)
plt.figure(7)
plt.semilogy(Qz[:,0],Simtest[:,0])