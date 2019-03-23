#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 12 12:04:28 2018

@author: Luc
"""

import numpy as np
from scipy.stats import vonmises
from scipy.interpolate import splev, splrep
from sklearn.utils.extmath import cartesian
from tqdm import trange


# Create parameter space and initialize prior and likelihood
class PSI_RiF:

    def __init__(self, params, stimuli, stim_selection='adaptive'):
        # initialize parameter grids
        self.kappa_ver = params['kappa_ver']
        self.kappa_hor = params['kappa_hor']
        self.tau = params['tau']
        self.kappa_oto = params['kappa_oto']
        self.lapse = params['lapse']

        # Initialize stimulus grids
        self.rods = stimuli['rods']
        self.frames = stimuli['frames']

        # dimensions of the parameter space
        self.kappa_ver_num = len(self.kappa_ver)
        self.kappa_hor_num = len(self.kappa_hor)
        self.tau_num = len(self.tau)
        self.kappa_oto_num = len(self.kappa_oto)
        self.lapse_num = len(self.lapse)

        # dimensions of the 2D stimulus space
        self.rod_num = len(self.rods)
        self.frame_num = len(self.frames)

        # set stimulus selection mode
        self.stim_selection = stim_selection

        # pre-compute likelihood and parameter prior
        print 'computing likelihood'
        self.computeLikelihood()
        print 'computing prior'
        self.computePrior()

        # compute easier-to-use parameter data-structure
        print "computing parameter values cartesian product"
        self.computeCartesian()

        # calculate best next stimulus with lowest entropy
        self.calcNextStim()


    def computeLikelihood(self):
        # the rods I need for the cumulative density function
        theta_rod_num = 10000
        theta_rod = np.linspace(-np.pi, np.pi, theta_rod_num)

        # allocate memory for the lookup table (P)
        P = np.zeros([self.kappa_ver_num, self.kappa_hor_num, self.tau_num, self.kappa_oto_num, self.lapse_num,
                      self.rod_num, self.frame_num])

        # initialize otolith distributions before for-loops
        P_oto = [self.__calcPOto(self.kappa_oto[i], theta_rod) for i in range(self.kappa_oto_num)]

        for i in trange(self.kappa_ver_num):
            for j in range(self.kappa_hor_num):
                for k in range(self.tau_num):
                    # compute the 2D rod-frame distribution for the given kappas, tau and rods
                    P_frame = self.__calcPFrame(self.kappa_ver[i], self.kappa_hor[j], self.tau[k], theta_rod)

                    for l in range(self.kappa_oto_num):
                        # compute the cumulative density of all distributions convolved
                        cdf = np.cumsum(P_frame * P_oto[l], 0) / np.sum(P_frame * P_oto[l], 0)

                        # reduce cdf to |rods|, |frames| by using spline interpolation
                        cdf = self.__reduceCDF(cdf, theta_rod)

                        for m in range(self.lapse_num):
                            # add lapse probability to distribution
                            PCW = self.lapse[m] + (1 - 2 * self.lapse[m]) * cdf

                            # add distribution to look-up table
                            P[i, j, k, l, m] = PCW

        # reshape to |param_space|, |rods|, |frames|
        self.lookup = np.reshape(P,
                                 [self.kappa_ver_num * self.kappa_hor_num * self.tau_num * self.kappa_oto_num * self.lapse_num,
                                  self.rod_num, self.frame_num],
                                 order="F")


    def __calcPFrame(self, kappa_ver, kappa_hor, tau, theta_rod):
        # computes kappas
        kappa1 = kappa_ver -\
                 (1 - np.cos(np.abs(2 * self.frames))) *\
                 tau *\
                 (kappa_ver - kappa_hor)
        kappa2 = kappa_hor +\
                 (1 - np.cos(np.abs(2 * self.frames))) *\
                 (1 - tau) *\
                 (kappa_ver - kappa_hor)

        # for every frame orientation, calculate frame influence
        P_frame = np.empty([len(theta_rod), self.frame_num])
        for i in range(self.frame_num):
            # the context provided by the frame
            P_frame0 = vonmises.pdf(theta_rod - self.frames[i], kappa1[i])
            P_frame90 = vonmises.pdf(theta_rod - np.pi/2 - self.frames[i], kappa2[i])
            P_frame180 = vonmises.pdf(theta_rod - np.pi - self.frames[i], kappa1[i])
            P_frame270 = vonmises.pdf(theta_rod - np.pi*3/2 - self.frames[i], kappa2[i])

            # add convolved distributions to P_frame
            P_frame[:, i] = P_frame0 + P_frame90 + P_frame180 + P_frame270

        return P_frame


    def __calcPOto(self, kappa_oto, theta_rod):
        # a simple von Mises distribution centered at 0 degrees
        return vonmises.pdf(theta_rod, kappa_oto).reshape(len(theta_rod), 1)


    def __reduceCDF(self, cdf, theta_rod):
        # initialize reduced cdf with dimensions |rods|, |frames|
        cdf_reduced = np.zeros([self.rod_num, self.frame_num])

        # for every frame orientation, calculate cumulative prob for rods in self.rods
        for i in range(self.frame_num):
            # use spline interpolation to get a continuous cdf
            cdf_continuous = splrep(theta_rod, cdf[:, i], s=0)

            # select cumulative probs of rods in self.rods from continuous cdf
            cdf_reduced[:, i] = splev(self.rods, cdf_continuous, der=0)

        return cdf_reduced


    def computePrior(self):
        # uniform discrete prior
        self.prior = np.ones(self.kappa_ver_num * self.kappa_hor_num * self.tau_num * self.kappa_oto_num * self.lapse_num) /\
                        (self.kappa_ver_num * self.kappa_hor_num * self.tau_num * self.kappa_oto_num * self.lapse_num)


    def computeCartesian(self):
        # all the combinations of all parameter values
        self.theta = cartesian([self.kappa_ver, self.kappa_hor, self.tau, self.kappa_oto, self.lapse]).transpose()

        
    def calcNextStim(self):
        # compute posterior
        self.paxs = np.einsum('i,ijk->ijk', self.prior, self.lookup)
        self.paxf = np.einsum('i,ijk->ijk', self.prior, 1.0 - self.lookup)


        # probabilities of rod and frame orientations
        ps = np.sum(self.paxs, 0)
        pf = np.sum(self.paxf, 0)


        # normalize posterior
        self.paxs = np.einsum('jk,ijk->ijk', 1.0 / ps, self.paxs)
        self.paxf = np.einsum('jk,ijk->ijk', 1.0 / pf, self.paxf)


        # determine next stimulus adaptively or randomly
        if self.stim_selection == 'adaptive':
            self.__calcAdaptiveStim(ps, pf)
        elif self.stim_selection == 'random':
            self.__calcRandomStim()
        else:
            raise Exception, 'undefined stimulus selection mode: ' + self.stim_selection

        self.stim = (self.rods[self.stim1_index], self.frames[self.stim2_index])

    def __calcAdaptiveStim(self, ps, pf):
        # cannot take the log of 0
        self.paxs[self.paxs == 0.0] = 1.0e-10
        self.paxf[self.paxf == 0.0] = 1.0e-10


        # compute expected entropy
        hs = np.einsum('ijk,ijk->jk', -self.paxs, np.log(self.paxs))
        hf = np.einsum('ijk,ijk->jk', -self.paxf, np.log(self.paxf))

        h = ps * hs + pf * hf


        # determine stimulus with smallest expected entropy
        idx = np.unravel_index(h.argmin(), h.shape)
        self.stim1_index, self.stim2_index = idx


    def __calcRandomStim(self):
        # randomly select next stimulus
        self.stim1_index = np.random.randint(self.rod_num)
        self.stim2_index = np.random.randint(self.frame_num)


    def addData(self, response):
        # update prior based on response
        if response == 1:
            self.prior = self.paxs[:, self.stim1_index, self.stim2_index]
        elif response == 0:
            self.prior = self.paxf[:, self.stim1_index, self.stim2_index]
        else:
            self.prior = self.prior

        # update stimulus based on posterior
        self.calcNextStim()


    def calcParameterValues(self, mode='mean'):
        if mode == 'MAP':
            param_values = self.__calcParameterValuesMAP()
        elif mode == 'mean':
            param_values = self.__calcParameterValuesMean()
        else:
            raise Exception, 'undefined parameter value calculation mode: ' + mode

        # put parameter values in dictionary
        param_values_dict = {'kappa_ver': param_values[0],
                             'kappa_hor': param_values[1],
                             'tau': param_values[2],
                             'kappa_oto': param_values[3],
                             'lapse': param_values[4]}

        return param_values_dict


    # calculate posterior parameter values based on MAP
    def __calcParameterValuesMAP(self):
        return self.theta[:, np.argmax(self.prior)]


    # calculate expected posterior parameter values
    def __calcParameterValuesMean(self):
        return np.matmul(self.theta, self.prior)


    def calcParameterDistributions(self):
        # get posterior in right shape
        posterior = self.prior.reshape([self.kappa_ver_num, self.kappa_hor_num, self.tau_num, self.kappa_oto_num, self.lapse_num])

        param_distributions = []
        for i in range(posterior.ndim):
            # all axes except the axis of the current parameter
            axes = tuple(k for k in range(posterior.ndim) if k != i)

            # calculate parameter distribution
            param_distribution = posterior.sum(axes)

            # add parameter distribution to param_distributions
            param_distributions.append(param_distribution)

        # put parameter distributions in dictionary
        param_distributions_dict = {'kappa_ver': param_distributions[0],
                                    'kappa_hor': param_distributions[1],
                                    'tau': param_distributions[2],
                                    'kappa_oto': param_distributions[3],
                                    'lapse': param_distributions[4]}

        return param_distributions_dict

