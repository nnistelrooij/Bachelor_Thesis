#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 12 12:04:28 2018

@author: Luc
"""

import numpy as np
from scipy.stats import vonmises
import matplotlib.pyplot as plt


# Create parameter space and initialize prior and likelihood
class PSI_RiF:

    def __init__(self, rods, frames, kappa_oto, kappa_ver, kappa_hor, tau):

        self.rods = rods
        self.frames = frames
        self.kappa_oto = kappa_oto
        self.kappa_ver = kappa_ver
        self.kappa_hor = kappa_hor
        self.tau = tau

        # dimensions of the 2D stimulus space
        self.rod_num = len(self.rods);
        self.frame_num = len(self.frames);

        # dimensions of the parameter space
        kappa_oto_num = len(self.kappa_oto);
        kappa_ver_num = len(self.kappa_ver);
        kappa_hor_num = len(self.kappa_hor);
        tau_num = len(self.tau);
        
        # the rods I need for the cumulative density function
        theta_rod=np.linspace(-np.pi,np.pi,10000)
        
        # allocate memory for the lookup table (P)
        P = np.zeros([kappa_oto_num,kappa_ver_num,kappa_hor_num,tau_num,self.rod_num,self.frame_num])
        
        self.kappa_oto_mtx=np.zeros([kappa_oto_num,kappa_ver_num,kappa_hor_num,tau_num])
        self.kappa_ver_mtx=np.zeros([kappa_oto_num,kappa_ver_num,kappa_hor_num,tau_num])
        self.kappa_hor_mtx=np.zeros([kappa_oto_num,kappa_ver_num,kappa_hor_num,tau_num])
        self.tau_mtx=np.zeros([kappa_oto_num,kappa_ver_num,kappa_hor_num,tau_num])
        
        
        for k in range(0,kappa_oto_num):
            for l in range(0,kappa_ver_num):
                for m in range(0,kappa_hor_num):
                    for n in range(0,tau_num):
                        kappa_oto2 = kappa_oto[k]
                        kappa_ver2 = kappa_ver[l]
                        kappa_hor2 = kappa_hor[m]
                        tau2 = tau[n]
                        kappa1 = kappa_ver2-(1-np.cos(np.abs(2*self.frames)))*tau2*(kappa_ver2-kappa_hor2)
                        kappa2 = kappa_hor2+(1-np.cos(np.abs(2*self.frames)))*(1-tau2)*(kappa_ver2-kappa_hor2)

                        for i in range(0,self.frame_num):
                    
                            # the context provided by the frame
                            P_frame1 = vonmises.pdf(theta_rod-self.frames[i],kappa1[i])
                            P_frame2 = vonmises.pdf(theta_rod-np.pi/2-self.frames[i],kappa2[i])
                            P_frame3 = vonmises.pdf(theta_rod-np.pi-self.frames[i],kappa1[i])
                            P_frame4 = vonmises.pdf(theta_rod-3*np.pi/2-self.frames[i],kappa2[i])
                
                            P_frame = (P_frame1+P_frame2+P_frame3+P_frame4)
                            P_frame = P_frame/np.sum(P_frame)

                    
                            # the otoliths
                            P_oto = vonmises.pdf(theta_rod,kappa_oto2)
                            
                            # the upright prior
                        
                            # compute the cumulative density of all distributions convolved
                            cdf=np.cumsum(np.multiply(P_oto, P_frame))/np.sum(np.multiply(P_oto, P_frame))
                            cdf=np.nan_to_num(cdf)
                            cdf[cdf==0]=1e-10    
                            cdf[cdf>1.0]=1.0 
                            for j in range(0,self.rod_num):
                                index = np.argmax(theta_rod>=rods[j])
                                P[k][l][m][n][j][i]=cdf[index]
                            
                        self.kappa_oto_mtx[k][l][m][n]=kappa_oto2
                        self.kappa_ver_mtx[k][l][m][n]=kappa_ver2
                        self.kappa_hor_mtx[k][l][m][n]=kappa_hor2
                        self.tau_mtx[k][l][m][n]=tau2
                 

        self.lookup = np.reshape(P,(kappa_oto_num*kappa_ver_num*kappa_hor_num*tau_num,self.rod_num,self.frame_num),order="F")
        self.prior = np.ones(kappa_oto_num*kappa_ver_num*kappa_hor_num*tau_num)/(kappa_oto_num*kappa_ver_num*kappa_hor_num*tau_num)
        self.calcNextStim()

        
    def calcNextStim(self):
         
           
        # Compute posterior
        self.paxs = np.empty([self.lookup.shape[0], self.lookup.shape[1], self.lookup.shape[2]])
        self.paxf = np.empty([self.lookup.shape[0], self.lookup.shape[1], self.lookup.shape[2]])
        h = np.empty([self.frame_num, self.rod_num])
        
        self.paxs = np.einsum('i,ijk->ijk', self.prior, self.lookup)
        self.paxf = np.einsum('i,ijk->ijk', self.prior, 1 - self.lookup)
        self.paxs[self.paxs==0]=1e-10;
        self.paxf[self.paxf==0]=1e-10;
        
        ps = np.sum(self.paxs,0)                   
        pf = np.sum(self.paxf,0)
        

        self.paxs = np.einsum('jk,ijk->ijk', 1/ps, self.paxs)
        self.paxf = np.einsum('jk,ijk->ijk', 1/pf, self.paxf)
        
        self.paxs[self.paxs==0]=1e-10;
        self.paxf[self.paxf==0]=1e-10;

        hs = np.einsum('ijk,ijk->jk', -self.paxs, np.log(self.paxs))
        hf = np.einsum('ijk,ijk->jk', -self.paxf, np.log(self.paxf))



        # Compute entropy
        #hs = np.sum(-self.paxs * np.log(self.paxs),0)
        #hf = np.sum(-self.paxf * np.log(self.paxf),0)
        
      
        # Compute expected entropy
        h = ps * hs + pf * hf
        
        plt.pcolormesh(h)
        plt.show(block=False)
        
        ind = np.unravel_index(h.argmin(), h.shape)  # index of smallest expected entropy

        
        #x_f = np.expand_dims(self.rods,axis=1) 
        #x_f = np.tile(x_f,(1,self.frame_num))
        #x_f = x_f.flatten('F')
        #y_f = np.expand_dims(self.frames,axis=0) 
        #y_f = np.tile(y_f,(self.rod_num,1))
        #y_f = y_f.flatten('F')


        # Find stimulus that minimizes expected entropy
        self.stim = ([self.rods[ind[0]],self.frames[ind[1]]])
        #self.stim1_index = np.argmin(np.abs(self.rods - self.stim[0]))
        #self.stim2_index = np.argmin(np.abs(self.frames - self.stim[1]))
        self.stim1_index = ind[0]
        self.stim2_index = ind[1]
        
        
    def addData(self,response):

        self.stim = None

        # Update prior based on response
        if response == 1:
            self.prior = self.paxs[:,self.stim1_index,self.stim2_index]
        elif response == 0:
            self.prior = self.paxf[:,self.stim1_index,self.stim2_index]
        else:
            self.prior = self.prior

        ## WARNING: solution for value,index is not unique!
        ## take MAP instead of Expected Value
       
        #self.theta = np.array([self.kappa_oto_mtx[:,:,:,:].flatten('F'), self.kappa_ver_mtx[:,:,:,:].flatten('F'),self.kappa_hor_mtx[:,:,:,:].flatten('F'),self.tau_mtx[:,:,:,:].flatten('F')])
           # dimensions of the parameter space
        kappa_oto_num = len(self.kappa_oto);
        kappa_ver_num = len(self.kappa_ver);
        kappa_hor_num = len(self.kappa_hor);
        tau_num = len(self.tau);
        self.theta =np.array([np.reshape(self.kappa_oto_mtx[:,:,:,:],kappa_oto_num*kappa_ver_num*kappa_hor_num*tau_num), np.reshape(self.kappa_ver_mtx[:,:,:,:],kappa_oto_num*kappa_ver_num*kappa_hor_num*tau_num),np.reshape(self.kappa_hor_mtx[:,:,:,:],kappa_oto_num*kappa_ver_num*kappa_hor_num*tau_num),np.reshape(self.tau_mtx[:,:,:,:],kappa_oto_num*kappa_ver_num*kappa_hor_num*tau_num)])
        
        self.parms  = np.matmul(self.theta,self.prior)
        
        diff = (self.theta.transpose()-self.parms).transpose()
        self.var_parms = np.matmul(np.power(diff,2), self.prior)
        
        self.stim = None
        self.calcNextStim()
        #self.stim1_index = np.random.randint(25)
        #self.stim2_index = np.random.randint(8)
        #self.stim = ([self.rods[self.stim1_index],self.frames[self.stim2_index]])
       # print('Variance', self.var_parms)
        return self.parms, self.var_parms

  


