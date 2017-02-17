# -*- coding: utf-8 -*-
"""
Created on Mon Jan 23 12:35:32 2017

@author: dfs1
"""
import numpy as np
import ContactHoleFunctions as CD
from multiprocessing import Pool
import time
import scipy.special as sp
import os

# Imports Intensity and Qx/Qr positions, normalizes lowest intensity to 1
Intensity=np.loadtxt('C3_Int1.txt')
Qr = np.loadtxt('C3_Qr1.txt')
Qz = np.loadtxt('C3_Qz1.txt')

Intensity[np.isnan(Intensity)]=1
IM=np.min(Intensity)
Intensity=np.loadtxt('C3_Int1.txt')

Intensity=Intensity/IM

# end Data import section

ConeNumber = 2

CPAR=np.zeros([ConeNumber+1,2])
SLD=np.zeros(ConeNumber+1)
Discretization = np.zeros(ConeNumber)
Pitch = 120

CPAR[0,0]= 27; CPAR[0,1]=140; SLD[0]=1;
CPAR[1,0]= 30; CPAR[1,1]= 33;  SLD[1]=2.2;
CPAR[2,0]=34; CPAR[2,1]=0; 

Coord=CD.CoordAssign(CPAR,SLD,ConeNumber,Pitch)

Discretization[0]=40
Discretization[1]=10

I0=0.000001
Bk=1
DW=1.5
SPAR=np.zeros(3)
SPAR[0]=DW; SPAR[1]=I0; SPAR[2]=Bk;

(FITPAR,FITPARLB,FITPARUB)=CD.PBA_Cone(CPAR,SPAR,ConeNumber)  

MCPAR=np.zeros([7])
MCPAR[0] = 24 # Chainnumber
MCPAR[1] = len(FITPAR)
MCPAR[2] = 50000 #stepnumber
MCPAR[3] = 0 #randomchains
MCPAR[4] = 100 # Resampleinterval
MCPAR[5] = 100 # stepbase
MCPAR[6] = 100 # steplength

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

    
    
def MCMCInit_Cone(FITPAR,FITPARLB,FITPARUB,MCPAR):
    
    MCMCInit=np.zeros([int(MCPAR[0]),int(MCPAR[1])+1])
    for i in range(int(MCPAR[0])):
        if i <MCPAR[3]: #reversed from matlab code assigns all chains below randomnumber as random chains
            for c in range(int(MCPAR[1])):
                MCMCInit[i,c]=FITPARLB[c]+(FITPARUB[c]-FITPARLB[c])*np.random.random_sample()
                SimInt=ConeIntensitySim(MCMCInit[i,:])
            C=np.sum(CD.Misfit(Intensity,SimInt))
            
            MCMCInit[i,int(MCPAR[1])]=C
            
        else:
            MCMCInit[i,0:int(MCPAR[1])]=FITPAR
            SimInt=ConeIntensitySim(MCMCInit[i,:])
            C=np.sum(CD.Misfit(Intensity,SimInt))
            MCMCInit[i,int(MCPAR[1])]=C
            
           
    return MCMCInit
    
def MCMC_Cone(MCMC_List):
    np.random.seed(os.getpid())
    MCMCInit=MCMC_List
    
    L = int(MCPAR[1])
    Stepnumber= int(MCPAR[2])
        
    SampledMatrix=np.zeros([Stepnumber,L+1]) 
    SampledMatrix[0,:]=MCMCInit
    Move = np.zeros([L+1])
    
    ChiPrior = MCMCInit[L]
    for step in np.arange(1,Stepnumber,1): 
        Temp = SampledMatrix[step-1,:].copy()
        for p in range(L-1):
            StepControl = MCPAR[5]+MCPAR[6]*np.random.random_sample()
            Move[p] = (FITPARUB[p]-FITPARLB[p])/StepControl*(np.random.random_sample()-0.5) # need out of bounds check
            Temp[p]=Temp[p]+Move[p]
            if Temp[p] < FITPARLB[p]:
                Temp[p]=FITPARLB[p]+(FITPARUB[p]-FITPARLB[p])/1000
            elif Temp[p] > FITPARUB[p]:
                Temp[p]=FITPARUB[p]-(FITPARUB[p]-FITPARLB[p])/1000
        SimPost=ConeIntensitySim(Temp)
        ChiPost=np.sum(CD.Misfit(Intensity,SimPost))
        if ChiPost < ChiPrior:
            SampledMatrix[step,0:L]=Temp[0:L]
            SampledMatrix[step,L]=ChiPost
            ChiPrior=ChiPost
            
        else:
            MoveProb = np.exp(-0.5*np.power(ChiPost-ChiPrior,2))
            if np.random.random_sample() < MoveProb:
                SampledMatrix[step,0:L]=Temp[0:L]
                SampledMatrix[step,L]=ChiPost
                ChiPrior=ChiPost
            else:
                SampledMatrix[step,:]=SampledMatrix[step-1,:]
    AcceptanceNumber=0;
    Acceptancetotal=len(SampledMatrix[:,1])

    for i in np.arange(1,len(SampledMatrix[:,1]),1):
        if SampledMatrix[i,0] != SampledMatrix[i-1,0]:
            AcceptanceNumber=AcceptanceNumber+1
    AcceptanceProbability=AcceptanceNumber/Acceptancetotal
    print(AcceptanceProbability)
    ReSampledMatrix=np.zeros([int(MCPAR[2])/int(MCPAR[4]),len(SampledMatrix[1,:])])

    c=-1
    for i in np.arange(0,len(SampledMatrix[:,1]),MCPAR[4]):
        c=c+1
        ReSampledMatrix[c,:]=SampledMatrix[i,:]
    return (ReSampledMatrix)

    


MCMCInitial=MCMCInit_Cone(FITPAR,FITPARLB,FITPARUB,MCPAR)

MCMC_List=[0]*int(MCPAR[0])
for i in range(int(MCPAR[0])):
    MCMC_List[i]=MCMCInitial[i,:]    
    
start_time = time.perf_counter()
if __name__ =='__main__':  
    pool = Pool(processes=24)
    SampledMatrix=pool.map(MCMC_Cone,MCMC_List)
    SampledMatrix=tuple(SampledMatrix)
    np.save('CH_C3_T2_R1',SampledMatrix) # add savedfilename here
    end_time=time.perf_counter()   
    print(end_time-start_time)    
   


  



