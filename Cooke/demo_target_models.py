
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

Demonstration of adaptive procedure for target selection example

References:
Cooke, J. R. H., van Beers, R. J., Selen, L. P. J., & Medendorp, W. P.
Bayesian adaptive stimuli selection for dissociating psychophysical models.

"""
import time as ti
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D
import sys
sys.path.insert(0, '../packages/ActiveModelSelection')
import AdaptiveModelSelection as Am

np.random.seed(42)  # set seed
nTrials = 250

# set constants to generate data
psyFunctions = 'sinbias'  # functions to generate data from (only a sinusoid)
sampleMethod = 'AMS'  # algorithms to test
conFunctions = ('flatbias', 'sinbias')  # models to compare

SOA = np.linspace(-250, 250, 25)  # SOA
phase = np.linspace(0, 360, 8) * (np.pi / float(180))  # phase of sled

# Stimulus grids to use
ampGrid = np.linspace(0, 60, 15)
offsetGrid = np.linspace(-3, 3, 15)
alphaGrid = np.linspace(-70, 70, 15)
sigmaGrid = np.linspace(50, 190, 15)

# priors to use for method and to generate data
# note: uniform is a discrete uniform (ie each grid value equally likely),
# the second 2 numbers determine the samples ranges used to test the algorithm eg min-max
ampPrior = ('uniform', 0, 60)
sigmaPrior = ('uniform', 50, 190)
offsetPrior = ('uniform', -3, 3)
alphaPrior = ('uniform', -70, 70)

# set parameters of simulated subject

alphaGen = 0
sigmaGen = 100
offsetGen = 1
ampGen = 40

mprior = 1 / float(len(conFunctions))  # uniform model prior over models
# set up models to use in the algorithm
flatbias = Am.build_model(conFunctions[0], mprior, ('alpha', 'sigma'),
                          (alphaPrior, sigmaPrior),
                          (alphaGrid, sigmaGrid))
sinbias = Am.build_model(conFunctions[1], mprior, ('alpha', 'sigma', 'offset', 'amp'),
                         (alphaPrior, sigmaPrior, offsetPrior, ampPrior),
                         (alphaGrid, sigmaGrid, offsetGrid, ampGrid))

models = {'flatbias': flatbias, 'sinbias': sinbias}  # dictionary of models

# set up data storage
response = np.full([nTrials], np.nan)
stim1 = np.full([nTrials], np.nan)
stim2 = np.full([nTrials], np.nan)
trial = np.full([nTrials], np.nan)

obj = Am.Psi(SOA, phase, models, sampleMethod='AMS')  # set up object

stimuli = obj.xCurrent
stimuli2 = obj.xCurrent2
generativeParams = {'stim1': obj.xCurrent, 'stim2': obj.xCurrent2,
                    'alpha': alphaGen, 'sigma': sigmaGen, 'amp': ampGen, 'offset': offsetGen}

##############################

print "\n##############################\n"
print "Simulating dataset\n"
print "Data generated with following parameters\n"
print "Psychometric function: %s " % (psyFunctions)
print "alpha Gen: %.3f " % (alphaGen)
print "sigma Gen: %.3f " % (sigmaGen)
print "Offset Gen : %.3f " % (offsetGen)
print "amp Gen: %.3f " % (ampGen)

## set up plotting
plt.ion()
f = plt.figure(figsize=(15, 10))
ax1 = f.add_subplot(2, 2, 1)
ax2 = f.add_subplot(2, 2, 2)
ax3 = f.add_subplot(2, 2, 3, projection='3d')
ax4 = f.add_subplot(2, 2, 4)

X = np.arange(-250, 250, 10)
Y = np.arange(0, 360 * (np.pi / float(180)), 0.5)
X, Y = np.meshgrid(X, Y)
likelihoodTrue = {'stim1': X,
                  'stim2': Y,
                  'alpha': alphaGen,
                  'sigma': sigmaGen,
                  'amp': ampGen,
                  'offset': offsetGen}
liktrue = Am.Lik(likelihoodTrue, psyfun=psyFunctions)

timeStart = ti.time()
for i in range(1, nTrials):  # run for length of trials
    trial[i] = i
    stim1[i] = stimuli
    stim2[i] = stimuli2
    # plot stuff
    modIter = 0
    ax1.clear()
    for mod in obj.prior:
        ax1.bar(modIter, obj.prior[mod]['modelPrior'], width=0.1)
        modIter += 0.1
    ax1.set_ylim(-0.1, 1.1)
    ax1.set_xlim(-0.1, 0.3)
    ax1.set_xticks([0, 0.1])
    ax1.set_xticklabels(['Constant', 'Sinusoid'])
    ax1.set_ylabel('Model Probability')
    ax1.set_title('Model Probabilities')

    ax2.clear()
    ax2.plot(trial, stim1, 'b.')
    ax2.set_xlim(0, nTrials)
    ax2.set_ylim(np.min(SOA), np.max(SOA))
    ax2.set_xlabel('Trial')
    ax2.set_ylabel('SOA (ms)')
    ax2.set_title('Selected Stimuli')
    ax2.legend()

    ax4.clear()
    ax4.plot(trial, stim2, 'b.')
    ax4.set_xlim(0, nTrials)
    ax4.set_ylim(0, np.max(phase))
    ax4.set_xlabel('Trial')
    ax4.set_ylabel('Phase (rad)')
    ax4.legend()

    generativeParams['stim1'] = stim1[i]
    generativeParams['stim2'] = stim2[i]

    r = int(Am.genDat(parameters=generativeParams, psyfun=psyFunctions))  # generate simulated response
    response[i] = r

    obj.addData(r)  # update Psi with response

    likelihoodEst = {'stim1': X,
                     'stim2': Y,
                     'alpha': obj.summarydata[mod]['parameterProb']['alpha'],
                     'sigma': obj.summarydata[mod]['parameterProb']['sigma'],
                     'offset': obj.summarydata[mod]['parameterProb']['offset'],
                     'amp': obj.summarydata[mod]['parameterProb']['amp']}
    likest = Am.Lik(likelihoodEst, psyfun=psyFunctions)

    ax3.clear()
    ax3.scatter(stim1, stim2, response, c='g', alpha=0.2, label='Response')
    ax3.set_xlim(np.min(SOA), np.max(SOA))
    ax3.set_ylim(np.min(phase), np.max(phase))
    ax3.set_zlim(0, 1)
    ax3.plot_surface(X, Y, likest, rstride=1, cstride=1,
                     linewidth=0, antialiased=False, alpha=0.3, label='Estimated')
    ax3.plot_surface(X, Y, liktrue, rstride=1, cstride=1,
                     linewidth=0, antialiased=False, alpha=0.1, color='r', label='True')
    ax3.scatter(stim1[i], stim2[i], response[i], c='r', label='Response')
    ax3.view_init(10, 240 - 180)
    ax3.set_title('Predicted vs actual response probability')
    # ax3.legend()
    ax3.set_xlabel('SOA (ms)')
    ax3.set_ylabel('Phase (rad)')
    ax3.set_zlabel('Response Probability')
    plt.pause(0.0001)

    stimuli = obj.xCurrent  # fetch new stimuli
    stimuli2 = obj.xCurrent2

print "Data estimated with following parameters\n"
for mod in obj.prior:
    print "Psychometric function: %s : probability %.2f " % (mod, obj.prior[mod]['modelPrior'])
    print "Parameters %s: " % (obj.summarydata[mod])
print "\n##############################\n"
