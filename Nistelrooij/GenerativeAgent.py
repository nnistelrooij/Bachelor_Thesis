import numpy as np
from scipy.stats import vonmises
from scipy.interpolate import splev, splrep


# transforms kappa values into sigma values
def kap2sig(kap):
    return np.sqrt((3.9945e3 / kap) - 0.0226e3)


class GenerativeAgent:

    def __init__(self, params, stimuli):
        # Initialize parameter values
        self.kappa_ver = params['kappa_ver']
        self.kappa_hor = params['kappa_hor']
        self.tau = params['tau']
        self.kappa_oto = params['kappa_oto']
        self.lapse = params['lapse']

        # Initialize stimulus grids
        self.rods = stimuli['rods']
        self.frames = stimuli['frames']

        # dimensions of the 2D stimulus space
        self.rod_num = len(self.rods)
        self.frame_num = len(self.frames)

        # pre-compute likelihood table
        print 'computing generative distribution\n'
        self.makeProbTable()


    def makeProbTable(self):
        # the rods I need for the cumulative density function
        theta_rod = np.linspace(-np.pi, np.pi, 10000)

        # make space for look-up table
        self.prob_table = np.zeros([self.rod_num, self.frame_num])

        # the otoliths
        P_oto = vonmises.pdf(theta_rod, self.kappa_oto)

        # compute kappas
        kappa1, kappa2 = self.__computeKappas()

        # for every frame orientation, calculate frame influence
        for i in range(self.frame_num):
            # the context provided by the frame
            P_frame0 = vonmises.pdf(theta_rod - self.frames[i], kappa1[i])
            P_frame90 = vonmises.pdf(theta_rod - np.pi/2 - self.frames[i], kappa2[i])
            P_frame180 = vonmises.pdf(theta_rod - np.pi - self.frames[i], kappa1[i])
            P_frame270 = vonmises.pdf(theta_rod - np.pi*3/2 - self.frames[i], kappa2[i])

            # add convolved distributions to P_frame
            P_frame = P_frame0 + P_frame90 + P_frame180 + P_frame270

            # cumulative response distribution per frame
            cdf = np.cumsum(P_frame * P_oto) / np.sum(P_frame * P_oto)

            # reduce cdf to |rods| using spline interpolation
            cdf_continuous = splrep(theta_rod, cdf, s=0)
            cdf = splev(self.rods, cdf_continuous, der=0)

            # add lapse probability to distribution
            PCW = self.lapse + (1 - 2 * self.lapse) * cdf

            # add probabilities to look-up table
            self.prob_table[:, i] = PCW


    def __computeKappas(self):
        kappa1 = self.kappa_ver -\
                 (1 - np.cos(np.abs(2 * self.frames))) *\
                 self.tau *\
                 (self.kappa_ver - self.kappa_hor)
        kappa2 = self.kappa_hor +\
                 (1 - np.cos(np.abs(2 * self.frames))) *\
                 (1 - self.tau) *\
                 (self.kappa_ver - self.kappa_hor)

        return kappa1, kappa2


    # determine response_num responses of the generative agent on a given rod and frame orientation
    def getResponse(self, stim_rod, stim_frame, response_num=1):
        # find index of stimulus
        idx_rod = np.where(self.rods == stim_rod)[0]
        idx_frame = np.where(self.frames == stim_frame)[0]

        # lookup probability of responding clockwise
        PCW = self.prob_table[idx_rod, idx_frame][0]

        # determine response
        return np.random.binomial(1, PCW, response_num)


    # determine response_num responses for each rod-frame pair
    def getResponses(self, response_num):
        # initialize responses array
        responses = np.empty([self.rod_num, self.frame_num, response_num])

        # determine response_num responses for each given rod and frame
        for i in range(self.rod_num):
            for j in range(self.frame_num):
                responses[i, j] = self.getResponse(self.rods[i], self.frames[j], response_num)

        return responses


    # calculate prior and visual context weights
    def calcWeights(self):
        # compute prior variance
        prior_variance = np.repeat(kap2sig(self.kappa_oto), self.frame_num)

        # compute visual context variance for each frame orientation
        kappa1, _ = self.__computeKappas()
        context_variance = kap2sig(kappa1)

        # compute weights with equation given in Alberts et al. (2018), displayed in Alberts et al. (2017)
        weights = {'prior': (1 / prior_variance) / ((1 / prior_variance) + (1 / context_variance)),
                   'context': (1 / context_variance) / ((1 / prior_variance) + (1 / context_variance))}

        return weights
