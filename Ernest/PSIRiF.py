import numpy as np
from scipy.stats import vonmises
from GenerativeAgent import GenerativeAgent


class PSIfor:

    def __init__(self, kappa_ver, kappa_hor, tau, kappa_oto, theta_frame, theta_rod):
        self.kappa_ver=kappa_ver
        self.kappa_hor=kappa_hor
        self.tau=tau
        self.kappa_oto=kappa_oto
        self.theta_frame=theta_frame
        self.theta_rod=theta_rod
        self.stim1_index=-1
        self.stim2_index=-1


        #dimensions of 2Dstimulus space
        self.nframes=len(self.theta_frame)
        self.nrods=len(self.theta_rod)

        #dimensions of 2D parameter space
        nkappa_ver=len(self.kappa_ver)
        nkappa_hor=len(self.kappa_hor)
        nkappa_oto=len(self.kappa_oto)
        ntau=len(self.tau)

        #initialize and compute the two kappas
        kappa1=np.zeros((nkappa_ver,nkappa_hor,ntau,self.nframes))
        kappa2=np.zeros((nkappa_ver,nkappa_hor,ntau,self.nframes))

        for kv in range(0,nkappa_ver):
            for kh in range(0,nkappa_hor):
                for t in range(0,ntau):
                    kappa1[kv,kh,t,:] = kappa_ver[kv]-(1-np.cos(np.abs(2*self.theta_frame)))*tau[t]*(kappa_ver[kv]-kappa_hor[kh])
                    kappa2[kv,kh,t,:] = kappa_hor[kh]+(1-np.cos(np.abs(2*self.theta_frame)))*(1-tau[t])*(kappa_ver[kv]-kappa_hor[kh])

        #initialize cumulitative distribution for every kappa_ver,kappa_hor,tau,sigma_oto combinati on per frame and rod orientation.
        cdf=np.zeros((nkappa_ver,nkappa_hor,ntau,nkappa_oto,self.nframes,self.nrods))

        for kv in range(0,nkappa_ver):
            for kh in range(0,nkappa_hor):
                for t in range(0,ntau):
                    # for all frames compute the contextual prior (four von mises), the otolith distribution (and the head-in-space prior)
                    for f in range(0,self.nframes):

                        # the context provided by the frame
                        p_frame1 = vonmises.pdf(self.theta_rodself.theta_frame[f],kappa1[kv,kh,t,f])
                        p_frame2 = vonmises.pdf(self.theta_rod-np.pi/2-self.theta_frame[f],kappa2[kv,kh,t,f])
                        p_frame3 = vonmises.pdf(self.theta_rod-np.piself.theta_frame[f],kappa1[kv,kh,t,f])
                        p_frame4 = vonmises.pdf(self.theta_rod-3*np.pi/2-self.theta_frame[f],kappa2[kv,kh,t,f])

                        p_frame = (p_frame1+p_frame2+p_frame3+p_frame4)/4.0

                        # the otoliths
                        for so in range(0,nkappa_oto):
                            ko=kappa_oto[so]
                            p_oto = vonmises.pdf(theta_rod,ko)
                            # the upright prior
                            #p_hsp = vonmises.pdf(theta_rod,kappa_hsp)

                            # compute the cumulative density of all distributions convolved
                            cdf[kv,kh,t,so,f,:]=np.cumsum(np.multiply(p_oto, p_frame))/np.sum(np.multiply(p_oto, p_frame))

        cdf=np.nan_to_num(cdf)
        cdf[cdf==0]=1e-10
        cdf[cdf>1.0]=1.0

        self.lookup=np.reshape(cdf,(nkappa_ver*nkappa_hor*nkappa_oto*ntau,self.nframes,self.nrods),order="F")
        # self.lookup=np.load('lookup.npy')
        self.prior=np.ones(nkappa_hor*nkappa_ver*nkappa_oto*ntau)/(nkappa_hor*nkappa_ver*nkappa_oto*ntau)

        self.makeG2()

        self.calcNextStim()

    def calcNextStim(self):
        # Compute posterior
        self.paxs = np.zeros([self.lookup.shape[0], self.lookup.shape[1], self.lookup.shape[2]])
        self.paxf = np.zeros([self.lookup.shape[0], self.lookup.shape[1], self.lookup.shape[2]])
        #h = np.zeros([self.nframes, self.nrods])

        self.paxs = np.einsum('i,ijk->ijk', self.prior, self.lookup)
        self.paxf = np.einsum('i,ijk->ijk', self.prior, 1.0 - self.lookup)

        ps = np.sum(self.paxs,0)
        pf = np.sum(self.paxf,0)

        self.paxs = np.einsum('jk,ijk->ijk', 1/ps, self.paxs)
        self.paxf = np.einsum('jk,ijk->ijk', 1/pf, self.paxf)

        # Compute entropy
        hs = np.sum(-self.paxs * np.log(self.paxs + 1e-10),0)
        hf = np.sum(-self.paxf * np.log(self.paxf + 1e-10),0)

        # Compute expected entropy
        h = hs*ps + hf*pf
        h = h.flatten('F')

        # Find stimulus with smallest expected entropy
        idx=np.argmin(h)

        frame_f = np.expand_dims(self.theta_frame,axis=1)
        frame_f = np.tile(frame_f,(1,self.nrods))
        frame_f = frame_f.flatten('F')
        rod_f = np.expand_dims(self.theta_rod,axis=0)
        rod_f = np.tile(rod_f,(self.nframes,1))
        rod_f = rod_f.flatten('F')

        # Find stimulus that minimizes expected entropy
        self.stim = ([frame_f[idx],rod_f[idx]])
        self.stim1_index = np.argmin(np.abs(self.theta_frame - self.stim[0]))
        self.stim2_index = np.argmin(np.abs(self.theta_rod - self.stim[1]))

    def addData(self,response):
        self.stim=None

        # Update prior based on response
        if response == 1:
            self.prior = self.paxs[:,self.stim1_index,self.stim2_index]
        elif response == 0:
            self.prior = self.paxf[:,self.stim1_index,self.stim2_index]
        else:
            self.prior = self.prior

        self.theta=np.array([self.kappa_ver_g2.flatten('F'),self.kappa_hor_g2.flatten('F'),self.tau_g2.flatten('F'),self.kappa_oto_g2.flatten('F')])
        self.params=np.matmul(self.theta,self.prior)

        self.calcNextStim()

        return self.params

    def makeG2(self):
        nkappa_ver=len(self.kappa_ver)
        nkappa_hor=len(self.kappa_hor)
        nkappa_oto=len(self.kappa_oto)
        ntau=len(self.tau)

        kappa_ver_g2 = np.expand_dims(self.kappa_ver,axis=1)
        kappa_ver_g2 = np.expand_dims(kappa_ver_g2,axis=2)
        kappa_ver_g2 = np.expand_dims(kappa_ver_g2,axis=3)
        self.kappa_ver_g2 = np.tile(kappa_ver_g2,(1,nkappa_hor,ntau,nkappa_oto))

        kappa_hor_g2 = np.expand_dims(self.kappa_hor,axis=0)
        kappa_hor_g2 = np.expand_dims(kappa_hor_g2,axis=2)
        kappa_hor_g2 = np.expand_dims(kappa_hor_g2,axis=3)
        self.kappa_hor_g2 = np.tile(kappa_hor_g2,(nkappa_ver,1,ntau,nkappa_oto))

        tau_g2 = np.expand_dims(self.tau,axis=0)
        tau_g2 = np.expand_dims(tau_g2,axis=1)
        tau_g2 = np.expand_dims(tau_g2,axis=3)
        self.tau_g2 = np.tile(tau_g2,(nkappa_ver,nkappa_hor,1,nkappa_oto))

        kappa_oto_g2 = np.expand_dims(self.kappa_oto,axis=0)
        kappa_oto_g2 = np.expand_dims(kappa_oto_g2,axis=1)
        kappa_oto_g2 = np.expand_dims(kappa_oto_g2,axis=2)
        self.kappa_oto_g2 = np.tile(kappa_oto_g2,(nkappa_ver,nkappa_hor,ntau,1))