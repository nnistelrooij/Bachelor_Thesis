import numpy as np
from scipy.stats import vonmises, beta
from scipy.interpolate import splev, splrep
from sklearn.utils.extmath import cartesian
from tqdm import trange


class PSI_RiF:
    # Create parameter space and initialize prior, likelihood and stimulus
    def __init__(self, params, stimuli, stim_selection='adaptive'):
        # initialize parameter grids
        self.kappa_ver = params['kappa_ver']
        self.kappa_hor = params['kappa_hor']
        self.tau = params['tau']
        self.kappa_oto = params['kappa_oto']
        self.lapse = params['lapse']
        self.params = params

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

        # pre-compute likelihood
        print 'computing likelihood'
        self.__computeLikelihood()

        # compute easier-to-use parameter data-structure
        print "computing parameter values cartesian product"
        self.__computeTheta()

        # reset psi object to initial values
        self.reset(stim_selection)


    def __computeLikelihood(self):
        # the rods I need for the cumulative density function
        theta_rod_num = 10000
        theta_rod = np.linspace(-np.pi, np.pi, theta_rod_num)

        # allocate memory for the lookup table (P)
        P = np.zeros([self.kappa_ver_num, self.kappa_hor_num, self.tau_num, self.kappa_oto_num, self.lapse_num,
                      self.rod_num, self.frame_num])

        # initialize otolith distributions before for-loops
        P_oto = [self.__calcPOto(kappa_oto, theta_rod) for kappa_oto in self.kappa_oto]

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


    def __computeTheta(self):
        # all the combinations of all parameter values
        self.theta = cartesian([self.kappa_ver, self.kappa_hor, self.tau, self.kappa_oto, self.lapse]).transpose()


    def reset(self, stim_selection='adaptive'):
        # compute initial prior
        print 'computing prior'
        self.__computePrior()

        # calculate best next stimulus with lowest entropy or a random stimulus based on self.stim_selection
        self.stim_selection = stim_selection
        self.__calcNextStim()


    def __computePrior(self):
        # compute parameter priors
        kappa_ver_prior = self.__computeUniformPrior(self.kappa_ver)
        kappa_hor_prior = self.__computeUniformPrior(self.kappa_hor)
        tau_prior = self.__computeBetaPrior(self.tau, 10, 1.6, 0.6467)
        kappa_oto_prior = self.__computeUniformPrior(self.kappa_oto)
        lapse_prior = self.__computeBetaPrior(self.lapse, 2.0, 35, 0.1322)

        # all the combinations of all parameter prior probabilities
        theta_prior = cartesian([kappa_ver_prior, kappa_hor_prior, tau_prior, kappa_oto_prior, lapse_prior])

        # turn combinations in 1D array of size |param_space| which sums to 1
        self.prior = np.prod(theta_prior, 1)


    # uniform discrete prior
    def __computeUniformPrior(self, param):
        return np.ones(len(param)) / len(param)


    # beta prior with a and b as shape parameters
    def __computeBetaPrior(self, param, a, b, param_threshold):
        # prior threshold to recover extreme values
        prior_threshold = beta.pdf(param_threshold, a, b)

        # compute beta prior
        prior = beta.pdf(param, a, b)
        prior[prior < prior_threshold] = prior_threshold

        # return normalized prior
        return prior / np.sum(prior)


    def __calcNextStim(self):
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
            self.stim1_index, self.stim2_index = self.__calcAdaptiveStim(ps, pf)
        elif self.stim_selection == 'random':
            self.stim1_index, self.stim2_index = self.__calcRandomStim()
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
        return np.unravel_index(h.argmin(), h.shape)


    # randomly select next stimulus
    def __calcRandomStim(self):
        return np.random.randint(self.rod_num), np.random.randint(self.frame_num)


    def addData(self, response):
        # update prior based on response
        if response == 1:
            self.prior = self.paxs[:, self.stim1_index, self.stim2_index]
        elif response == 0:
            self.prior = self.paxf[:, self.stim1_index, self.stim2_index]
        else:
            raise Exception, 'response is ' + str(response) + ', but must be 1 or 0'

        # update stimulus based on posterior
        self.__calcNextStim()


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
        for axis in 'ijklm':
            # calculate marginalized posterior for one parameter
            param_distribution = np.einsum('ijklm->' + axis, posterior)

            # add parameter distribution to param_distributions
            param_distributions.append(param_distribution)

        # put parameter distributions in dictionary
        param_distributions_dict = {'kappa_ver': param_distributions[0],
                                    'kappa_hor': param_distributions[1],
                                    'tau': param_distributions[2],
                                    'kappa_oto': param_distributions[3],
                                    'lapse': param_distributions[4]}

        return param_distributions_dict


    def calcParameterVariances(self):
        # calculate parameter means and parameter distributions
        param_means = self.calcParameterValues(mode='mean')
        param_distributions = self.calcParameterDistributions()

        # return dictionary with each parameter values distribution variance
        return {param: self.__calcParameterVariance(param, param_means, param_distributions) for param in self.params.keys()}


    # calculate parameter values distribution variance
    def __calcParameterVariance(self, param, param_means, param_distributions):
        return np.sqrt(np.sum(param_distributions[param] * (self.params[param] - param_means[param])**2))


    def calcNegLogLikelihood(self, data):
        if isinstance(data, np.ndarray):
            # compute negative log likelihood for all right, respectively left responses
            neg_log_likelihood_right_responses = np.einsum('ijk,jkl->i', -np.log(self.lookup), data)
            neg_log_likelihood_left_responses = np.einsum('ijk,jkl->i', -np.log(1.0 - self.lookup), 1.0 - data)

            # compute negative log likelihood for all responses
            neg_log_likelihood = neg_log_likelihood_right_responses + neg_log_likelihood_left_responses

            return neg_log_likelihood
        else:
            # compute negative log likelihood for one response
            if data == 1:
                return -np.log(self.lookup[:, self.stim1_index, self.stim2_index])
            elif data == 0:
                return -np.log(1.0 - self.lookup[:, self.stim1_index, self.stim2_index])
            else:
                raise Exception, 'response is ' + str(data) + ', but must be 1 or 0'
