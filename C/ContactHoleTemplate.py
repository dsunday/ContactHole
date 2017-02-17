# -*- coding: utf-8 -*-
"""
Created on Mon Jan 23 12:35:32 2017

@author: dfs1
"""
import numpy as np
import ContactHoleFunctions as CD
import ContactPlotting as CDplot
from multiprocessing import Pool
import time
import scipy.special as sp
import matplotlib.pyplot as plt

# Imports Intensity and Qx/Qr positions, normalizes lowest intensity to 1
Intensity=np.loadtxt('CInt.txt')
Qr = np.loadtxt('CQr.txt')
Qz = np.loadtxt('CQz.txt')
#SM=np.load('LAMtest.npy')
Intensity[np.isnan(Intensity)]=1
IM=np.min(Intensity)
Intensity=np.loadtxt('CInt.txt')

Intensity=Intensity/IM

# end Data import section

ConeNumber = 1

CPAR=np.zeros([ConeNumber+1,2])
SLD=np.zeros(ConeNumber+1)
Discretization = np.zeros(ConeNumber)
Pitch = 120

CPAR[0,0]= 22; CPAR[0,1]=160; SLD[0]=1;
CPAR[1,0]= 27; CPAR[1,1]= 0;  SLD[1]=2.2;
 

Coord=CD.CoordAssign(CPAR,SLD,ConeNumber,Pitch)

Discretization[0]=40
#Discretization[1]=10

I0=0.000001
Bk=1
DW=1.5
SPAR=np.zeros(3)
SPAR[0]=DW; SPAR[1]=I0; SPAR[2]=Bk;

def ConeIntensitySim(FITPAR):
    H1 = 0
    H2 = 0
    Form=np.zeros([int(len(Qr[:,0])),int(len(Qr[0,:]))])
    CPAR=np.zeros([ConeNumber+1,2])
    CPAR[:,0:2]=np.reshape(FITPAR[0:(ConeNumber+1)*2],(ConeNumber+1,2))
    SPAR=FITPAR[ConeNumber*2+2:ConeNumber*2+5]
    for i in range (ConeNumber):
        H2=H2+CPAR[i,1]
        z=np.zeros([int(Discretization[i])])
        stepsize=CPAR[i,1]/Discretization[i]
        z=np.arange(H1,H2+0.01,stepsize)
        if i > 0 :
            H1=H1+CPAR[i-1,1]
            
        z=np.arange(H1,H2+0.01,stepsize)
        R1=CPAR[i,0]
        R2=CPAR[i+1,0]
        if R1==R2:
            R1=R1+0.000001
        Slope=(H2-H1)/(R2-R1)
        for ii in range(len(z)-1):
            RI1=(z[ii]-H1)/Slope+R1
            RI2=(z[ii+1]-H1)/Slope+R1
            fa=2*np.pi*RI1/Qr*sp.jv(1,Qr*RI1)*np.exp(1j*Qz*z[ii])
            fb=2*np.pi*RI2/Qr*sp.jv(1,Qr*RI2)*np.exp(1j*Qz*z[ii+1])
            Form=Form+stepsize*(fb+fa)/2*SLD[i]
    M=np.power(np.exp(-1*(np.power(Qr,2)+np.power(Qz,2))*np.power(SPAR[0],2)),0.5)
    Formfactor=Form*M
    Formfactor=abs(Formfactor)
    SimInt = np.power(Formfactor,2)*SPAR[1]+SPAR[2]
    return (SimInt)

    
(FITPAR,FITPARLB,FITPARUB)=CD.PBA_Cone(CPAR,SPAR,ConeNumber)    
SimInt=ConeIntensitySim(FITPAR)
Chi=sum(sum(CD.Misfit(Intensity,SimInt)))
plt.figure(1)
CDplot.plotCone(Coord,ConeNumber,Pitch)
plt.figure(2)
CDplot.PlotQzCut(Qz,Intensity,SimInt,int(len(Intensity[0,:])))

