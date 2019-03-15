import numpy as np
from scipy.stats import vonmises
from random import random
from tqdm import trange


class GenerativeAgent:

    def __init__(self, kappa_ver, kappa_hor, tau, kappa_oto, lapse, rods, frames):
        # Initialize parameter values
        self.kappa_ver = kappa_ver
        self.kappa_hor = kappa_hor
        self.tau = tau
        self.kappa_oto = kappa_oto
        self.lapse = lapse

        # Initialize stimulus grids
        self.rods = rods
        self.frames = frames

        # dimensions of the 2D stimulus space
        self.rod_num = len(self.rods)
        self.frame_num = len(self.frames)

        # pre-compute likelihood table
        print 'computing generative distribution\n'
        self.makeProbTable()


    def makeProbTable(self):
        # the rods I need for the cumulative density function
        theta_rod = np.linspace(-np.pi, np.pi, 10000)
        self.prob_table = np.zeros([self.rod_num, self.frame_num])

        # compute kappas
        kappa1 = self.kappa_ver -\
                 (1 - np.cos(np.abs(2 * self.frames))) *\
                 self.tau *\
                 (self.kappa_ver - self.kappa_hor)
        kappa2 = self.kappa_hor +\
                 (1 - np.cos(np.abs(2 * self.frames))) *\
                 (1 - self.tau) *\
                 (self.kappa_ver - self.kappa_hor)

        # for every frame orientation, calculate frame influence
        for j in range(self.frame_num):
            # the context provided by the frame
            P_frame0 = vonmises.pdf(theta_rod - self.frames[j], kappa1[j])
            P_frame90 = vonmises.pdf(theta_rod - np.pi/2 - self.frames[j], kappa2[j])
            P_frame180 = vonmises.pdf(theta_rod - np.pi - self.frames[j], kappa1[j])
            P_frame270 = vonmises.pdf(theta_rod - np.pi*3/2 - self.frames[j], kappa2[j])

            # add convolved distributions to P_frame
            P_frame = P_frame0 + P_frame90 + P_frame180 + P_frame270

            # the otoliths
            P_oto = vonmises.pdf(theta_rod, self.kappa_oto)

            # cumulatitve response distribution per frame
            cdf = np.cumsum(P_frame * P_oto) / np.sum(P_frame * P_oto)

            # add lapse probability to distribution
            cdf = self.lapse + (1 - 2 * self.lapse) * cdf

            for i in range(self.rod_num):
                # add distribution to lookup table
                idx = np.argmax(theta_rod >= self.rods[i])
                self.prob_table[i, j] = cdf[idx]

    # determine the response of agent on particular frame and rod combination
    def getResponse(self, stim_rod, stim_frame):
        # find index of stimulus
        idx_rod = np.where(self.rods == stim_rod)[0]
        idx_frame = np.where(self.frames == stim_frame)[0]

        # lookup probability of responding clockwise
        PCW = self.prob_table[idx_rod, idx_frame][0]

        # determine response
        return np.random.binomial(1, PCW)