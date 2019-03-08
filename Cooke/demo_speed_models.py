
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

Demonstration of adaptive procedure for the noise model example.

References:
Cooke, J. R. H., van Beers, R. J., Selen, L. P. J., & Medendorp, W. P.
Bayesian adaptive stimuli selection for dissociating psychophysical models.

"""
import numpy as np
from mpl_toolkits.mplot3d.axes3d import Axes3D
import matplotlib.pyplot as plt
import AdaptiveModelSelection as Am

np.random.seed(42)  # set seed
nTrials = 750

psyFunctions = 'genWebGauss'  # model to generate data from
sampleMethod = 'AMS'
conFunctions = ('fixGauss', 'weberGauss', 'genWebGauss')  # models to compare
stimuliRange1 = np.linspace(0.3, 9, 20)  # stimuli space : probe
stimuliRange2 = np.linspace(0.6, 9, 10)  # stimuli space : ref

# grids to approximate marginal likelihood
thresholdGrid = np.linspace(-0.6, 0.6, 17)  # alpha
slopeGrid = np.linspace(0.01, 0.5, 25)  # beta
lapseRateGrid = np.linspace(0, 0.1, 10)  # lambda
gammaGrid = np.linspace(0, 2, 20)  # gamma

# priors to use for method and to generate data
# note: uniform is a discrete uniform (ie each grid value equally likely),
# the second 2 numbers determine the samples ranges used to test the algorithm eg min-max
thresholdPrior = ('uniform', -0.6, 0.6)  # alpha
slopePrior = ('uniform', 0.01, 0.5)  # beta
lapsePrior = ('uniform', 0, 0.1)  # lapse
gammaPrior = ('uniform', 0, 2)  # gamma

# set parameters of simulated subject
thresholdGen = 0.016
slopeGen = 0.197
lapseGen = 0
gammaGen = 0.122

modelPrior = 1 / float(len(conFunctions))  # uniform prior over models

# set up each model
fixgauss = Am.build_model(conFunctions[0], modelPrior, ('alpha', 'beta', 'lapse'),
                          (thresholdPrior, slopePrior, lapsePrior),
                          (thresholdGrid, slopeGrid * 5, lapseRateGrid))

webergauss = Am.build_model(conFunctions[1], modelPrior, ('alpha', 'beta', 'lapse'),
                            (thresholdPrior, slopePrior, lapsePrior),
                            (thresholdGrid, slopeGrid, lapseRateGrid))

genWebGauss = Am.build_model(conFunctions[2], modelPrior, ('alpha', 'beta', 'lapse', 'gamma'),
                             (thresholdPrior, slopePrior, lapsePrior, gammaPrior),
                             (thresholdGrid, slopeGrid, lapseRateGrid, gammaGrid))

models = {'fixGauss': fixgauss, 'weberGauss': webergauss, 'genWebGauss': genWebGauss}

obj = Am.Psi(stimuliRange1, stimuliRange2, models, sampleMethod=sampleMethod)  # set up object

# fetch first stimulus
stimuli = obj.xCurrent
stimuli2 = obj.xCurrent2

# set parameters to generate simulated subjects response
generativeParams = {'stim1': obj.xCurrent, 'stim2': obj.xCurrent2,
                    'alpha': thresholdGen, 'beta': slopeGen, 'lapse': lapseGen, 'gamma': gammaGen}

# set up variables for storage
response = np.full([nTrials], np.nan)
probe = np.full([nTrials], np.nan)
ref = np.full([nTrials], np.nan)
trial = np.full([nTrials], np.nan)

##############################

print "\n##############################\n"
print "Simulating data set\n"
print "Data generated from with following parameters\n"
print "Psychometric function: %s " % psyFunctions
print "Alpha: %.3f " % thresholdGen
print "Beta: %.3f " % slopeGen
print "Lapse Rate: %.3f " % lapseGen
print "Gamma: %.3f " % gammaGen
print "\n##############################\n"

# set up plotting
plt.ion()
f = plt.figure(figsize=(15, 10))
ax1 = f.add_subplot(2, 2, 1)
ax2 = f.add_subplot(2, 2, 2)
ax3 = f.add_subplot(2, 2, 3, projection='3d')
X = np.arange(0.1, 10, 0.8)
Y = np.arange(0.1, 10, 0.8)
X, Y = np.meshgrid(X, Y)
likelihoodTrue = {'stim1': X,
                  'stim2': Y,
                  'alpha': thresholdGen,
                  'beta': slopeGen,
                  'lapse': lapseGen,
                  'gamma': gammaGen}

# True psychometric surface of subject
liktrue = Am.Lik(likelihoodTrue, psyfun=psyFunctions)

for i in range(0, nTrials):  # run for length of trials

    trial[i] = i
    probe[i] = stimuli
    ref[i] = stimuli2

    # plotting
    modIter = 0
    ax1.clear()
    for mod in obj.prior:
        ax1.bar(modIter, obj.prior[mod]['modelPrior'], width=0.1)
        modIter += 0.1
    ax1.set_ylim(-0.1, 1.1)
    ax1.set_xlim(-0.1, 0.4)
    ax1.set_xticks([0, 0.1, 0.2])
    ax1.set_xticklabels(['Fixed', 'Weber', 'Generalized'])
    ax1.set_ylabel('Model Probability')
    ax1.set_title('Model Probabilities')

    ax2.clear()
    ax2.plot(trial, probe, 'b.', label=r'$s_2$')
    ax2.plot(trial, ref, 'r.', label=r'$s_1$')
    ax2.set_xlim(0, nTrials)
    ax2.set_ylim(0, np.max(stimuliRange1))
    ax2.set_xlabel('Trial')
    ax2.set_ylabel('Speed (deg/s)')
    ax2.set_title('Selected Stimuli')
    ax2.legend()

    # set stimuli subject receives
    generativeParams['stim1'] = stimuli
    generativeParams['stim2'] = stimuli2

    r = Am.genDat(parameters=generativeParams, psyfun=psyFunctions)  # generate simulated response
    response[i] = r  # store response
    obj.addData(r)  # update Psi with response

    # Use mean estimate of parameters to estimate the subject psychometric response surface
    likelihoodEst = {'stim1': X,
                     'stim2': Y,
                     'alpha': obj.summarydata[mod]['parameterProb']['alpha'],
                     'beta': obj.summarydata[mod]['parameterProb']['beta'],
                     'lapse': obj.summarydata[mod]['parameterProb']['lapse'],
                     'gamma': obj.summarydata[mod]['parameterProb']['gamma']}
    likest = Am.Lik(likelihoodEst, psyfun=psyFunctions)

    # plotting
    ax3.clear()
    ax3.scatter(probe, ref, response, c='r', alpha=0.2, label='Response')
    ax3.set_xlim(0, 10)
    ax3.set_ylim(0, 10)
    ax3.set_zlim(0, 1)
    ax3.plot_surface(X, Y, likest, rstride=1, cstride=1,
                     linewidth=0, antialiased=False, alpha=0.3, label='Estimated')
    ax3.plot_surface(X, Y, liktrue, rstride=1, cstride=1,
                     linewidth=0, antialiased=False, alpha=0.1, color='r', label='True')
    ax3.scatter(probe[i], ref[i], response[i], c='k', label='Current trial')
    ax3.view_init(30, 220)
    ax3.set_title('Predicted vs actual response probability')
    ax3.set_xlabel(r'$s_1$ (deg/s)')
    ax3.set_ylabel(r'$s_2$ (deg/s)')
    ax3.set_zlabel('Response Probability')
    plt.pause(0.001)

    stimuli = obj.xCurrent  # fetch new stimuli
    stimuli2 = obj.xCurrent2

print "\n##############################\n"
print "Data estimated with following parameters\n"
for mod in obj.prior:
    print "Psychometric function: %s : probability %.2f " % (mod, obj.prior[mod]['modelPrior'])
    print "Parameters %s: " % (obj.summarydata[mod])
print "\n##############################\n"
