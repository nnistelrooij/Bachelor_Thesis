"""
Copyright 2018, J. Cooke, Radboud University Nijmegen

AdaptiveModelSelection is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

AdaptiveModelSelection is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with AdaptiveModelSelection. If not, see <http://www.gnu.org/licenses/>.

---

AdaptiveModelSelection procedure for dissociating psychometric models.

References:
Cooke, J. R. H., van Beers, R. J., Selen, L. P. J., & Medendorp, W. P.
Bayesian adaptive stimuli selection for dissociating psychophysical models.

"""

import collections
import sys
from sklearn.utils.extmath import cartesian
import scipy
from scipy.stats import norm, beta, vonmises
from scipy.misc import logsumexp
import numpy as np
import string as st


def genDat(parameters, psyfun='cGauss'):
    """Generate synthetic data from psychometric function.

    Arguments
    ---------
        parameters (dict): dictionary containing parameters to generate data from. For example,
            stim1   : reference speed

            stim2    : probe speed

            alpha  : bias in psychometric curve

            beta   : slope of psychometric curve

            lapse : lapse rate of psychometric function

        psyfun (str): psychometric function to generate data from.
            'fixGauss': cumulative gaussian with fixed noise

            'WeberGauss' cumulative Gaussian with weber scaling

            'genwebGauss' cumulative Gaussian with combination of weber scaling and fixed noise

    Returns
    -------
    scalar bernoulli variable sampled with success probability determined by psyfunction and parameters
    """
    pf = Lik(parameters, psyfun=psyfun)
    y = pf

    r = np.random.binomial(1, y)

    return r


def Lik(parameters, psyfun='cGauss'):
    """Generate conditional probabilities from psychometric function.

    Arguments
    ---------
        parameters: dictionary containing parameters necessary to simulate psychometric model- psyfun
        (the model can be recoded to your purpose, you just need the build model parameters to match)
            Currently used
            alpha   : bias

            beta    : weber fraction

            lambda  : lapse rate (upper asymptote & lower asymptote)

            gamma   : fixed noise

            stim1    : first stimulus

            stim2      : second stimulus (set to 0 if not 2 ifc)

        psyfun (str): type of psychometric function.
            (again modified as desired, requires a matching model name in self.models)
            'fixGauss': cumulative gaussian with fixed noise (using relative 0.2*beta for slope)

            'WeberGauss' cumulative Gaussian with weber scaling

            'genwebGauss' cumulative Gaussian with combination of weber scaling and fixed noise

            'flatbias' cumulative Gaussian with a fixed bias term

            'sinbias' cumulative Gaussian with a sinusoidal modulation of bias

    Returns
    -------
    1D-array of conditional probabilities p(response | theta , x)
    """

    if psyfun == 'fixGauss':
        pf = norm.cdf(parameters['stim1'] - parameters['stim2'], parameters['alpha'],
                      np.sqrt((parameters['beta']) ** 2 + (parameters['beta']) ** 2))
        y = parameters['lapse'] + (1 - 2 * parameters['lapse']) * pf

    elif psyfun == 'weberGauss':
        pf = norm.cdf(parameters['stim1'] - parameters['stim2'], parameters['alpha'],
                      np.sqrt((parameters['beta'] * parameters['stim1']) ** 2 + (
                          parameters['beta'] * parameters['stim2']) ** 2))
        y = parameters['lapse'] + (1 - 2 * parameters['lapse']) * pf

    elif psyfun == 'genWebGauss':
        pf = norm.cdf(parameters['stim1'] - parameters['stim2'], parameters['alpha'],
                      np.sqrt(((parameters['beta'] * parameters['stim1']) ** 2 + parameters['gamma'] ** 2) + (
                          (parameters['beta'] * parameters['stim2']) ** 2 + parameters['gamma'] ** 2)))
        y = parameters['lapse'] + (1 - 2 * parameters['lapse']) * pf

    elif psyfun == 'flatbias':
        mu = parameters['alpha']
        pf = norm.cdf(parameters['stim1'], mu, parameters['sigma'])
        y = pf
    elif psyfun == 'sinbias':
        mu = parameters['amp'] * np.sin(parameters['stim2'] + parameters['offset']) + parameters['alpha']
        pf = norm.cdf(parameters['stim1'], mu, parameters['sigma'])
        y = pf

    elif psyfun == 'funcVestibular' or psyfun == 'dysfuncVestibular':
        # the frame in retinal coordinates
        frameRetinal = -parameters['stim1']

        # compute how the kappa's changes with frame angle
        kappa1 = parameters['kappaVer'] -\
                 (1 - np.cos(np.abs(2 * frameRetinal))) *\
                 parameters['tau'] *\
                 (parameters['kappaVer'] - parameters['kappaHor'])
        kappa2 = parameters['kappaHor'] +\
                 (1 - np.cos(np.abs(2 * frameRetinal))) *\
                 (1 - parameters['tau']) *\
                 (parameters['kappaVer'] - parameters['kappaHor'])

        # probability distributions for the four von-mises
        PFrame0 = vonmises.pdf(-parameters['stim2'] + frameRetinal, kappa1)
        PFrame90 = vonmises.pdf(-parameters['stim2'] + np.pi / 2 + frameRetinal, kappa2)
        PFrame180 = vonmises.pdf(-parameters['stim2'] + np.pi + frameRetinal, kappa1)
        PFrame270 = vonmises.pdf(-parameters['stim2'] + 3 * np.pi / 2 + frameRetinal, kappa2)

        # add the probability distributions
        PFrame = PFrame0 + PFrame90 + PFrame180 + PFrame270

        # the prior is always oriented with gravity
        POto = norm.pdf(parameters['stim2'], 0, parameters['sigmaOto'])

        # normalize each rod distribution to 1
        if PFrame.size > 1:
            rodGridSize = 9
            for i in range(0, PFrame.size, rodGridSize):
                PFrame[i:i+rodGridSize] /= np.sum(PFrame[i:i+rodGridSize])
                POto[i:i+rodGridSize] /= np.sum(POto[i:i+rodGridSize])

        # compute likelihood distribution
        cdf = np.multiply(PFrame, POto)

        # compute cumulative distribution and normalize each rod distribution to 1
        if PFrame.size > 1:
            for i in range(0, PFrame.size, rodGridSize):
                cdf[i:i+rodGridSize] = np.cumsum(cdf[i:i+rodGridSize]) / np.sum(cdf[i:i+rodGridSize])

        # compute output with lapse rate
        y = parameters['lapse'] + (1 - 2 * parameters['lapse']) * cdf

    else:
        raise ValueError('function ' + psyfun + ' is invalid')

    return y


def build_model(modelName, modelPrior, parameters, parameterPriors, parameterGrid):
    """build a dictionary with relevant model information.

    Arguments
    ---------
        modelName :
            str:name of the model.

        modelPrior :
            float: prior probability of this model

        parameters:
            tuple: tuple of parameter names, one for each parameter of the model.

        parameterPriors:
            tuple: tuple of priors for each parameter (provided in order of parameters)
                    Supported priors
                    ('uniform',a,b): discrete prior over parameter grid, a,b would be needed for continous uniform (not yet implemented)
                    ('beta',a,b): beta distribution with shape parameters a and b (beta requires a,b >0)

        parameterGrid:
            tuple: tuple of parameter grids (numpy array) for each parameters (provided in order of parameters)

    Returns
    -------
    a dictionary with all the keys and values necessary to be used in our main psimethod.

    Example
    ----------
    fixgauss = MS.build_model('fixgauss',mprior, ('beta','lapse'),(slopePrior,lapsePrior),(slopeGrid,lapseRateGrid))

    """

    model = {'name': modelName, 'prior': modelPrior}
    par = {}
    for i in range(0, len(parameters)):
        par[parameters[i]] = {'prior': parameterPriors[i], 'searchGrid': parameterGrid[i]}
    model['parameters'] = par
    return model


class Psi:
    """Find the stimulus intensity which maximises the cross entropy of the model probabiltiy distribution.

    Arguments
    ---------
        stimRange :
            range of possible stimulus intensities for first stimulus.

        stimRange2 :
            range of possible stimulus intensities for second stimulus.

        models:
            dictionary of models and their associated parameters (see buid_model)

        sampleMethod:
            'AMS': Active Model Selection (select stimuli adaptively)

            'Rand': Random (select stimuli randomly fom stimRange and stimRange2)

        prior:
            prior attribute from previous session, useful for multisession recording

        likelihood:
            precomputed likelihood attribute, useful for simulating so we don't need to recompute likelihood each time


    How to use
    ----------
        Create a dictionary for each model you would like to compare (see build model) and stimuli range (1 and 2).
        Pass these into this function with the sample method you would like to use. This will initialize our method
        by computing the corresponding joint priors, likelihood functions and the first stimuli values xCurrent and
        xCurrent2 (first and second stimuli). Following using these stimuli call add data to update our method and
        create the next stimuli

        Full example:
            fixgauss = MS.build_model('fixgauss',mprior, ('beta','lapse'),(slopePrior,lapsePrior),(slopeGrid,lapseRateGrid))
            webergauss = MS.build_model('webergauss',mprior, ('beta','lapse'),(slopePrior,lapsePrior),(slopeGrid,lapseRateGrid))
            genWebGauss = MS.build_model('genWebGauss',mprior, ('beta','lapse','gamma'),(slopePrior,lapsePrior,gammaPrior),
                                                                                        (slopeGrid,lapseRateGrid,gammaGrid))

            models={'fixGauss':fixgauss,'weberGauss':webergauss,'genWebGauss':genWebGauss}

            obj=MS.Psi(stimRange1, stimRange2, models, sampleMethod='Rand') #set up object
            stimuli=obj.xCurrent
            stimuli2=obj.xCurrent2

            obj.addData(response)  # update Psi with response
    """

    def __init__(self, stimRange, stimRange2, models, sampleMethod, prior=None, likelihood=None):

        ## Psychometric function parameters
        self.stimRange = stimRange  # range of stimulus intensities
        self.stimRange2 = stimRange2  # range of stimulus intensities for second interval (2-afc)
        self.sampleMethod = sampleMethod
        self.models = models

        if prior is None:
            print 'Computing initial prior'
            self.prior = self.genprior()
            print 'Initial prior computed'
        else:
            self.prior = prior

        if likelihood is None:
            print 'Precomputing likelihood grid'
            self.likelihood = self.genlik()
            self.calcEinsumString()
            print 'Likelihood grid computed'
        else:
            self.likelihood = likelihood

        # settings
        self.response = []
        self.build_data_struct()

        ## Generate the first stimulus intensity
        self.minEntropyStim()

        print 'Initialization complete'

    def calcEinsumString(self):
        """Create einstein summation string for each model to return the marginal likelihood p(r=1/S1,S2,M).
            p(r=1/S1,S2,M)=sum(p(r=1/S1,S2,theta,M).
            I.E 2 parameter prior is NxM, Likelihood is however NxMxS1xS2, marginal likelihood is S1xS2
            where S are stimuli we want to compute. Done using einstein summation notation for tensors.
        """
        letters = st.ascii_lowercase  # import letters to make string
        for mod in self.likelihood:  # go over all the models
            ind = len(self.likelihood[mod]['likelihoodGridSize'])  # find number of parameters
            lettersNeeded = letters[0:ind]
            s1 = lettersNeeded[int(self.likelihood[mod]['stimInd'][0])]  # stimuli 1 letter
            s2 = lettersNeeded[int(self.likelihood[mod]['stimInd'][1])]  # stimuli 2 letter
            endInd = s1 + s2  # end dim is just the stimuli dimensions
            parInd = ''.join(e for e in lettersNeeded if e.lower() not in endInd)
            Einstring = lettersNeeded + ',' + parInd + '->' + endInd
            self.likelihood[mod]['einString'] = Einstring

    def genprior(self):  # prior probability distribution
        """create model and parameter priors for each model"""

        modelpriors = {}
        for mod in self.models:
            priors = collections.OrderedDict()
            priorGridSize = collections.OrderedDict()
            for par in self.models[mod]['parameters']:
                if self.models[mod]['parameters'][par]['prior'][0] == 'beta':
                    # beta prior
                    priors[par] = beta.pdf(self.models[mod]['parameters'][par]['searchGrid'],
                                           self.models[mod]['parameters'][par]['prior'][1],
                                           self.models[mod]['parameters'][par]['prior'][2])

                if self.models[mod]['parameters'][par]['prior'][0] == 'uniform':
                    # discrete uniform prior not continous
                    p = 1 / float(len(self.models[mod]['parameters'][par]['searchGrid']))
                    priors[par] = np.repeat(p, len(self.models[mod]['parameters'][par]['searchGrid']))

                priors[par] = priors[par] / np.sum(priors[par])
                priorGridSize[par] = np.size(self.models[mod]['parameters'][par]['searchGrid'])
            allPrior = np.prod(cartesian(priors.values()), 1)
            jointPrior = np.reshape(allPrior, priorGridSize.values())

            modelpriors[mod] = {'modelPrior': self.models[mod]['prior'], 'parameterPrior': priors,
                                'jointPrior': jointPrior, 'jointPriorNreshape': allPrior}
        return modelpriors

    def genlik(self):  # create likelihood table
        """create response likelihood for all stimuli and parameter combinations for each model"""
        modelLikeli = {}
        for mod in self.models:
            LikelihoodGrid = collections.OrderedDict()
            LikelihoodGridSize = collections.OrderedDict()
            for par in self.models[mod]['parameters']:
                LikelihoodGrid[par] = self.models[mod]['parameters'][par]['searchGrid']
                LikelihoodGridSize[par] = np.size(self.models[mod]['parameters'][par]['searchGrid'])

            LikelihoodGrid['stim1'] = self.stimRange
            LikelihoodGridSize['stim1'] = np.size(self.stimRange)
            LikelihoodGrid['stim2'] = self.stimRange2
            LikelihoodGridSize['stim2'] = np.size(self.stimRange2)
            a = cartesian((LikelihoodGrid.values()))
            keys = LikelihoodGrid.keys()

            LikelihoodComb = {}
            LikelihoodCombInd = np.empty(2)
            for i in range(0, len(keys)):
                LikelihoodComb[keys[i]] = a[:, i]
                if (keys[i] == 'stim1'):
                    LikelihoodCombInd[0] = int(i)
                elif (keys[i] == 'stim2'):
                    LikelihoodCombInd[1] = int(i)
            LikReshape = np.reshape(Lik(LikelihoodComb, psyfun=mod), LikelihoodGridSize.values())
            modelLikeli[mod] = {'likelihoodGrid': LikReshape, 'stimInd': LikelihoodCombInd,
                                'likelihoodGridSize': LikelihoodGridSize, 'likelihoodParams': LikelihoodGrid,
                                'likCombs': LikelihoodComb}
            del a
        return modelLikeli

    def __expectcrossentropy(self):
        """compute expected cross entropy over responses"""
        self.entropyYes = np.zeros((len(self.stimRange), len(self.stimRange2)))
        self.entropyNo = np.zeros((len(self.stimRange), len(self.stimRange2)))
        self.entropy = np.zeros((len(self.stimRange), len(self.stimRange2)))
        for mod in self.prior:
            self.entropyYes = self.entropyYes + self.prior[mod]['modelPrior'] * np.log(self.prior[mod]['posteriorYes'])
            self.entropyNo = self.entropyNo + self.prior[mod]['modelPrior'] * np.log(self.prior[mod]['posteriorNo'])
        self.entropy = -(
            self.likelihood['probRespYes'] * self.entropyYes + self.likelihood['probRespNo'] * self.entropyNo)

    def __expectentropy(self):
        """compute expected entropy of posterior over responses"""
        self.entropyYes = np.zeros((len(self.stimRange), len(self.stimRange2)))
        self.entropyNo = np.zeros((len(self.stimRange), len(self.stimRange2)))
        self.entropy = np.zeros((len(self.stimRange), len(self.stimRange2)))
        for mod in self.prior:
            # calculate expected information of posterior for each stim (mathematically max information is equal to min entropy)
            self.entropyYes = self.entropyYes + self.prior[mod]['posteriorYes'] * np.log(
                self.prior[mod]['posteriorYes'])
            self.entropyNo = self.entropyNo + self.prior[mod]['posteriorNo'] * np.log(self.prior[mod]['posteriorNo'])
        # E_r[H(x,r)] = p(r==1/x)sum(p(m/r==1)log(p(m/r==1)) +   p(r==0/x)sum(p(m/r==0)log(p(m/r==0))
        self.entropy = -(
            self.likelihood['probRespYes'] * self.entropyYes + self.likelihood['probRespNo'] * self.entropyNo)

    def minEntropyStim(self):
        """Find stimuli combination based on entropy of posterior

        Minimizing entropy is used as selection criterion for stimuli on the next trial.
        """

        # compute marginal likelihood
        if (self.sampleMethod == 'Rand'):
            self.minEntropyInd = (
                np.random.randint(0, np.size(self.stimRange)), np.random.randint(0, np.size(self.stimRange2)))
            self.likelihood['probRespYes'] = 0  # Normalizing constants over models
            self.likelihood['probRespNo'] = 0  # Normalizing constants
            for mod in self.prior:
                self.likelihood[mod]['MarginalLikelihoodYes'] = np.sum(np.multiply(
                    self.likelihood[mod]['likelihoodGrid'][Ellipsis, self.minEntropyInd[0], self.minEntropyInd[1]],
                    self.prior[mod]['jointPrior']))
                self.likelihood[mod]['MarginalLikelihoodNo'] = np.sum(np.multiply((1 - self.likelihood[mod][
                    'likelihoodGrid'][Ellipsis, self.minEntropyInd[0], self.minEntropyInd[1]]),
                                                                                  self.prior[mod]['jointPrior']))
                self.prior[mod]['probYes'] = self.prior[mod]['modelPrior'] * self.likelihood[mod][
                    'MarginalLikelihoodYes']
                self.prior[mod]['probNo'] = self.prior[mod]['modelPrior'] * self.likelihood[mod]['MarginalLikelihoodNo']
                self.likelihood['probRespYes'] = self.likelihood['probRespYes'] + self.prior[mod]['probYes']
                self.likelihood['probRespNo'] = self.likelihood['probRespNo'] + self.prior[mod]['probNo']
            for mod in self.prior:
                self.prior[mod]['posteriorYes'] = self.prior[mod]['probYes'] / self.likelihood['probRespYes']
                self.prior[mod]['posteriorNo'] = self.prior[mod]['probNo'] / self.likelihood['probRespNo']

        if (self.sampleMethod == 'AMS'):
            self.likelihood['probRespYes'] = 0  # Normalizing constants
            self.likelihood['probRespNo'] = 0  # Normalizing constants
            for mod in self.prior:
                einString = self.likelihood[mod]['einString']

                # compute marginal likelihood  \sum_\theta(p(r==1/ \theta,m,x)p(\theta/m,x))
                self.likelihood[mod]['MarginalLikelihoodYes'] = np.einsum(einString,
                                                                          self.likelihood[mod]['likelihoodGrid'],
                                                                          self.prior[mod]['jointPrior'])
                # compute marginal likelihood  \sum_\theta(p(r==0/ \theta,m,x)p(\theta/m,x))
                self.likelihood[mod]['MarginalLikelihoodNo'] = np.einsum(einString,
                                                                         (1 - self.likelihood[mod]['likelihoodGrid']),
                                                                         # compute marginal likelihood
                                                                         self.prior[mod]['jointPrior'])
                # compute unnormalized model posterior (p(m/r ==1,x) ~= p(m) * \sum_\theta(p(r==1/ \theta,m,x)p(\theta/m,x))
                self.prior[mod]['probYes'] = self.prior[mod]['modelPrior'] * self.likelihood[mod][
                    'MarginalLikelihoodYes']

                # compute  unnormalized model posterior (p(m/r ==0,x) ~= p(m) * \sum_\theta(p(r==0/ \theta,m,x)p(\theta/m,x))
                self.prior[mod]['probNo'] = self.prior[mod]['modelPrior'] * self.likelihood[mod]['MarginalLikelihoodNo']

                # compute normalizing constant p(r==1/x) = sum_m(p(r=1/x,m)p(m))
                self.likelihood['probRespYes'] = self.likelihood['probRespYes'] + self.prior[mod]['probYes']

                # compute normalizing constant p(r==0/x) = sum_m(p(r=0/x,m)p(m))
                self.likelihood['probRespNo'] = self.likelihood['probRespNo'] + self.prior[mod]['probNo']

            for mod in self.prior:
                # compute normalized model posterior (p(m/r ==1,x) = p(m) * \sum_\theta(p(r==1/ \theta,m,x)p(\theta/m,x)) / p(r==1/x)
                self.prior[mod]['posteriorYes'] = self.prior[mod]['probYes'] / self.likelihood['probRespYes']

                # compute normalized model posterior (p(m/r ==0,x) = p(m) * \sum_\theta(p(r==0/ \theta,m,x)p(\theta/m,x)) / p(r==0/x)
                self.prior[mod]['posteriorNo'] = self.prior[mod]['probNo'] / self.likelihood['probRespNo']

                ## Expected entropy for the next trial at intensity x, producing response r
            self.__expectentropy()
            self.minEntropyInd = np.unravel_index(self.entropy.argmin(),
                                                  self.entropy.shape)  # index of smallest expected entropy

        self.xCurrent = self.stimRange[self.minEntropyInd[0]]  # stim intensity at minimum expected entropy
        self.xCurrent2 = self.stimRange2[self.minEntropyInd[1]]  # stim intensity at minimum expected entropy

    def addData(self, response):
        """
        Add the most recent response to start calculating the next stimuli

        Arguments
        ---------
            response:
                currently restricted to binary (0,1) meaning is dependent on likelihood function
                1:
                0:
        """
        self.response.append(response)
        self.xCurrent = None
        self.xCurrent2 = None

        #  Keep the posterior probability distribution that corresponds to the recorded response

        if response == 1:
            # select the posterior that corresponds to the stimulus intensity of lowest entropy
            for mod in self.prior:
                if self.sampleMethod == 'AMS':
                    self.prior[mod]['modelPrior'] = self.prior[mod]['posteriorYes'][
                        self.minEntropyInd[0], self.minEntropyInd[1]]  # update model prob
                    self.prior[mod]['jointPrior'] = self.prior[mod]['jointPrior'] * \
                                                    self.likelihood[mod]['likelihoodGrid'][
                                                        Ellipsis, self.minEntropyInd[0], self.minEntropyInd[1]]
                    self.prior[mod]['jointPrior'] = self.prior[mod]['jointPrior'] / np.sum(
                        self.prior[mod]['jointPrior'])
                elif self.sampleMethod == 'Rand':
                    self.prior[mod]['modelPrior'] = self.prior[mod]['posteriorYes']  # update model prob
                    self.prior[mod]['jointPrior'] = self.prior[mod]['jointPrior'] * \
                                                    self.likelihood[mod]['likelihoodGrid'][
                                                        Ellipsis, self.minEntropyInd[0], self.minEntropyInd[1]]
                    self.prior[mod]['jointPrior'] = self.prior[mod]['jointPrior'] / np.sum(
                        self.prior[mod]['jointPrior'])


        elif response == 0:
            for mod in self.prior:
                if self.sampleMethod == 'AMS':
                    self.prior[mod]['modelPrior'] = self.prior[mod]['posteriorNo'][
                        self.minEntropyInd[0], self.minEntropyInd[1]]  # update model prob
                    self.prior[mod]['jointPrior'] = self.prior[mod]['jointPrior'] * (
                        1 - self.likelihood[mod]['likelihoodGrid'][
                            Ellipsis, self.minEntropyInd[0], self.minEntropyInd[1]])
                    self.prior[mod]['jointPrior'] = self.prior[mod]['jointPrior'] / np.sum(
                        self.prior[mod]['jointPrior'])
                if self.sampleMethod == 'Rand':
                    self.prior[mod]['modelPrior'] = self.prior[mod]['posteriorNo']  # update model prob
                    self.prior[mod]['jointPrior'] = self.prior[mod]['jointPrior'] * (
                        1 - self.likelihood[mod]['likelihoodGrid'][
                        Ellipsis, self.minEntropyInd[0], self.minEntropyInd[1]])
                    self.prior[mod]['jointPrior'] = self.prior[mod]['jointPrior'] / np.sum(
                        self.prior[mod]['jointPrior'])
        self.updateParEstimate()
        self.minEntropyStim()

    def meta_data(self):
        import time
        import copy
        metadata = {}
        date = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(time.time()))
        metadata['date'] = date
        metadata['AMS Version'] = 1.0
        metadata['Python Version'] = sys.version
        metadata['Numpy Version'] = np.__version__
        metadata['Scipy Version '] = scipy.__version__
        jsonModel = copy.deepcopy(self.models)
        for mod in jsonModel:
            for par in jsonModel[mod]['parameters']:
                jsonModel[mod]['parameters'][par]['searchGrid'] = \
                    jsonModel[mod]['parameters'][par]['searchGrid'].tolist()
        metadata['models'] = jsonModel
        return metadata

    def build_data_struct(self):
        summarydata = {}
        for mod in self.prior:
            parameters = {}
            i = 0
            idx = np.arange(0, len(self.prior[mod]['parameterPrior']))
            for par in self.prior[mod]['parameterPrior']:
                sumaxes = np.delete(idx, i)
                # compute parameter mean
                parameters[par] = np.sum(
                    np.sum(self.prior[mod]['jointPrior'], axis=tuple(sumaxes)) * self.models[mod]['parameters'][par][
                        'searchGrid'])
                i = i + 1
            summarydata[mod] = {'modelProb': self.prior[mod]['modelPrior'], 'parameterProb': parameters}
        self.summarydata = summarydata

    def updateParEstimate(self, stat='Mean'):
        for mod in self.prior:
            parameters = {}
            i = 0
            idx = np.arange(0, len(self.prior[mod]['parameterPrior']))
            if stat == 'Mean':
                for par in self.prior[mod]['parameterPrior']:
                    sumaxes = np.delete(idx, i)
                    # compute parameter mean
                    parameters[par] = np.sum(np.sum(self.prior[mod]['jointPrior'], axis=tuple(sumaxes)) *
                                             self.models[mod]['parameters'][par]['searchGrid'])
                    i = i + 1
                self.summarydata[mod] = {'modelProb': self.prior[mod]['modelPrior'], 'parameterProb': parameters}
            elif stat == 'MAP':  # todo get MAP to work
                for par in self.prior[mod]['parameterPrior']:
                    print 'todo'
                self.summarydata[mod] = {'modelProb': self.prior[mod]['modelPrior'], 'parameterProb': parameters}

    def batchmodelComp(self, stim1, stim2, response, returnLog=False):
        import pymc
        infoffset = -10000  # replace infinity with very very low value
        data = {}
        modelRes = {}
        bestFitParms = {}
        parms = {}
        output = {}
        data['stim1'] = stim1
        data['stim2'] = stim2
        ML = {}
        for mod in self.prior:
            for par in self.models[mod]['parameters']:
                parms[par] = self.models[mod]['parameters'][par]['searchGrid']
            parCombs = cartesian(parms.values())
            keys = parms.keys()
            tmp = np.empty(parCombs.shape[0])
            for j in range(0, parCombs.shape[0]):
                for i in range(0, len(keys)):
                    data[keys[i]] = parCombs[:, i][j]
                likeli = Lik(data, psyfun=mod)
                likeli = likeli
                tmp[j] = pymc.bernoulli_like(response, likeli)
            modelRes[mod] = tmp
            tmp[~np.isfinite(tmp)] = infoffset
            maxInd = np.argmax(tmp)
            bestFitParms[mod] = {}  # dict to store most likely parameters for this model
            for i in range(0, len(keys)):
                bestFitParms[mod][keys[i]] = parCombs[:, i][maxInd]  # most likely parameters out of grid

            logPos = modelRes[mod] + np.log(self.prior[mod]['jointPriorNreshape'])
            ML[mod] = logsumexp(logPos)
        for mod in self.prior:
            output[mod] = {}
            output[mod]['LogL'] = ML[mod]
            output[mod]['ModelProbability'] = np.exp(ML[mod] - max(ML.values())) / float(
                np.sum(np.exp(ML.values() - max(ML.values()))))
            output[mod]['BestFitParms'] = bestFitParms[mod]
        return output
