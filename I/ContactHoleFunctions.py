# -*- coding: utf-8 -*-
"""
Created on Thu Feb 16 15:52:13 2017

@author: dfs1
"""
import numpy as np
import scipy.special as sp

def CoordAssign(TPAR,SLD,Trapnumber,Pitch):
    Coord=np.zeros([Trapnumber+1,5,1])
    for T in range (Trapnumber+1):
        if T==0:
            Coord[T,0,0]=0
            Coord[T,1,0]=TPAR[0,0]
            Coord[T,2,0]=TPAR[0,1]
            Coord[T,3,0]=0
            Coord[T,4,0]=SLD[0]
        else:
            Coord[T,0,0]=Coord[T-1,0,0]+0.5*(TPAR[T-1,0]-TPAR[T,0])
            Coord[T,1,0]=Coord[T,0,0]+TPAR[T,0]
            Coord[T,2,0]=TPAR[T,1]
            Coord[T,3,0]=0
            Coord[T,4,0]=SLD[T]
 
    return (Coord)
    
    
def Misfit(Exp,Sim):
    Chi2= abs(np.log(Exp)-np.log(Sim))
    #ms=np.zeros([len(Exp[:,1]),len(Exp[1,:]),2])
    #ms[:,:,0]=Sim
    #ms[:,:,1]=Exp
    #MS= np.nanmin(ms,2)
    #Chi2=np.power((D/MS),2)
    Chi2[np.isnan(Chi2)]=0
    return Chi2
    
def ConeFourierTransform(CPAR,ConeNumber,Qr,Qz,Discretization,SLD):
    H1 = 0
    H2 = 0
    Form=np.zeros([int(len(Qr[:,0])),int(len(Qr[0,:]))])
    
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
    return Form
    
def ConeIntensitySim(CPAR,ConeNumber,Qr,Qz,Discretization,SLD,SPAR):
    H1 = 0
    H2 = 0
    Form=np.zeros([int(len(Qr[:,0])),int(len(Qr[0,:]))])
    
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
    
def PBA_Cone(CPAR,SPAR,ConeNumber):
     
    SPARLB=SPAR[0:4]*0.7
    SPARUB=SPAR[0:4]*1.3

    FITPAR=CPAR[:,0:2].ravel()
    FITPARLB=FITPAR*0.7
    FITPARUB=FITPAR*1.3
    FITPAR=np.append(FITPAR,SPAR)
       
    FITPARLB=np.append(FITPARLB,SPARLB)
    
    FITPARUB=np.append(FITPARUB,SPARUB)
    
    return (FITPAR,FITPARLB,FITPARUB)