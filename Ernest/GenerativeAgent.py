import numpy as np
from scipy.stats import vonmises
from random import random


class GenerativeAgent:

    def __init__(self, kappa_ver, kappa_hor, tau, kappa_oto, theta_frame, theta_rod):
        self.kappa_ver=kappa_ver
        self.kappa_hor=kappa_hor
        self.tau=tau
        self.kappa_oto=kappa_oto
        self.theta_frame=theta_frame
        self.theta_rod=theta_rod

        self.makeProbTable()


    def makeProbTable(self):
        nFrames = np.size(self.theta_frame,0)
        nRods = np.size(self.theta_rod,0)
        cdf=np.zeros((nFrames,nRods))

        #compute kappas
        kappa1 = self.kappa_ver-(1-np.cos(np.abs(2*self.theta_frame)))*self.tau*(self.kappa_verself.kappa_hor)
        kappa2 = self.kappa_hor+(1-np.cos(np.abs(2*self.theta_frame)))*(1-self.tau)*(self.kappa_verself.kappa_hor)

        #for every frame orientation, compute:
        for i in range(0,np.size(self.theta_frame,0)):

            # the context provided by the frame
            P_frame1 = vonmises.pdf(self.theta_rod-self.theta_frame[i],kappa1[i])
            P_frame2 = vonmises.pdf(self.theta_rod-np.pi/2-self.theta_frame[i],kappa2[i])
            P_frame3 = vonmises.pdf(self.theta_rod-np.pi-self.theta_frame[i],kappa1[i])
            P_frame4 = vonmises.pdf(self.theta_rod-3*np.pi/2-self.theta_frame[i],kappa2[i])

            P_frame = (P_frame1+P_frame2+P_frame3+P_frame4)/4

            # the otoliths
            P_oto = vonmises.pdf(self.theta_rod,self.kappa_oto)

            # cumulatitve response distribution per frame
            cdf[i,:]=np.cumsum(np.multiply(P_oto, P_frame))/np.sum(np.multiply(P_oto, P_frame))

        #save cdf as lookup table
        self.prob_table=cdf

    #Determine the response of agent on particular frame and rod combination
    def getResponse(self,stim_frame,stim_rod):

        #Find index of stimulus
        idx_frame=np.where(self.theta_frame==stim_frame)[0]
        idx_rod=np.where(self.theta_rod==stim_rod)[0]

        #lookup probability of responding 1
        PCW=self.prob_table[idx_frame, idx_rod][0]

        #Determine response
        if random()<=PCW:
            return 1
        else:
            return 0