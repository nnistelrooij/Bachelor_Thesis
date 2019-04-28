"""
Demonstration of adaptive procedure for the Bayesian vestibular model

References:
Cooke, J. R. H., van Beers, R. J., Selen, L. P. J., & Medendorp, W. P.
Bayesian adaptive stimuli selection for dissociating psychophysical models.

Alberts, B. B. G. T., de Brouwer, A. J., Selen, L. P. J., & Medendorp, W. P.
A Bayesian Account of Visual-Vestibular Interactions in the Rod-and-Frame task.

"""
import numpy as np
from mpl_toolkits.mplot3d.axes3d import Axes3D
from sklearn.utils.extmath import cartesian
import matplotlib.pyplot as plt
import AdaptiveModelSelection as Am

np.random.seed(42)  # set seed
nTrials = 750

psyFunctions = 'funcVestibular'  # model to generate data from
sampleMethod = 'AMS'
conFunctions = ('funcVestibular', 'dysfuncVestibular')  # models to compare
stimuliRange1 = np.linspace(-45, 40, 18) * np.pi / 180  # stimuli space : frames
stimuliRange2 = np.array([-7, -4, -2, -1, 0, 1, 2, 4, 7]) * np.pi / 180  # stimuli space : rods

# grids to approximate marginal likelihood
funcSigmaOtoGrid = np.linspace(8.12, 21.48, 8)  # upright prior
funcKappaVerGrid = np.linspace(59.95, 493.18, 8)  # vertical frame precision
funcKappaHorGrid = np.linspace(0.2, 65.86, 8)  # horizontal frame precision
funcTauGrid = np.linspace(0.77, 0.97, 8)  # increase/decrease scalar for kappaVer and kappaHor
funcLapseGrid = np.linspace(0, 0.13, 8) # lapse rate

# priors to use for method and to generate data
# note: uniform is a discrete uniform (ie each grid value equally likely),
# the second 2 numbers determine the samples ranges used to test the algorithm eg min-max
funcSigmaOtoPrior = ('uniform', 8.12, 21.48)
funcKappaVerPrior = ('uniform', 59.95, 493.18)
funcKappaHorPrior = ('uniform', 0.2, 65.86)
funcTauPrior = ('uniform', 0.77, 0.97)
funcLapsePrior = ('uniform', 0, 0.13)

# grids to approximate marginal likelihood
dysfuncSigmaOtoGrid = np.linspace(8.2, 18.04, 8)  # upright prior
dysfuncKappaVerGrid = np.linspace(26.26, 131.34, 8)  # vertical frame precision
dysfuncKappaHorGrid = np.linspace(1.38, 3.72, 8)  # horizontal frame precision
dysfuncTauGrid = np.linspace(0.91, 1, 8)  # increase/decrease scalar for kappaVer and kappaHor
dysfuncLapseGrid = np.linspace(0, 0.04, 8) # lapse rate

# priors to use for method and to generate data
# note: uniform is a discrete uniform (ie each grid value equally likely),
# the second 2 numbers determine the samples ranges used to test the algorithm eg min-max
dysfuncSigmaOtoPrior = ('uniform', 8.2, 18.04)
dysfuncKappaVerPrior = ('uniform', 26.26, 131.34)
dysfuncKappaHorPrior = ('uniform', 1.38, 3.72)
dysfuncTauPrior = ('uniform', 0.91, 1)
dysfuncLapsePrior = ('uniform', 0, 0.04)

# set parameters of simulated subject
sigmaOtoGen = 15
kappaVerGen = 138
kappaHorGen = 3
tauGen = 0.87
lapseGen = 0.01

sigmaOtoGen = 13.3
kappaVerGen = 50
kappaHorGen = 2.13
tauGen = 0.97
lapseGen = 0.02

modelPrior = 1 / float(len(conFunctions))  # uniform prior over models

# set up each model
funcVestibular = Am.build_model(conFunctions[0], modelPrior, ('sigmaOto', 'kappaVer', 'kappaHor', 'tau', 'lapse'),
                                (funcSigmaOtoPrior, funcKappaVerPrior, funcKappaHorPrior, funcTauPrior, funcLapsePrior),
                                (funcSigmaOtoGrid, funcKappaVerGrid, funcKappaHorGrid, funcTauGrid, funcLapseGrid))
dysfuncVestibular = Am.build_model(conFunctions[1], modelPrior, ('sigmaOto', 'kappaVer', 'kappaHor', 'tau', 'lapse'),
                                   (dysfuncSigmaOtoPrior, dysfuncKappaVerPrior, dysfuncKappaHorPrior, dysfuncTauPrior, dysfuncLapsePrior),
                                   (dysfuncSigmaOtoGrid, dysfuncKappaVerGrid, dysfuncKappaHorGrid, dysfuncTauGrid, dysfuncLapseGrid))

models = {'funcVestibular': funcVestibular, 'dysfuncVestibular': dysfuncVestibular}

obj = Am.Psi(stimuliRange1, stimuliRange2, models, sampleMethod=sampleMethod)  # set up object

# fetch first stimulus
stimuli = obj.xCurrent
stimuli2 = obj.xCurrent2

# set parameters to generate simulated subjects response
generativeParams = {'stim1': obj.xCurrent, 'stim2': obj.xCurrent2,
                    'sigmaOto': sigmaOtoGen,
                    'kappaVer': kappaVerGen, 'kappaHor': kappaHorGen, 'tau': tauGen,
                    'lapse': lapseGen}

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
print "Sigma_oto: %.3f " % sigmaOtoGen
print "Kappa_ver: %.3f " % kappaVerGen
print "Kappa_hor Rate: %.3f " % kappaHorGen
print "Tau: %.3f " % tauGen
print "Lapse rate: %.3f " % lapseGen
print "\n##############################\n"

# set up plotting
plt.ion()
f = plt.figure(figsize=(15, 10))
ax1 = f.add_subplot(2, 2, 1)
ax2 = f.add_subplot(2, 2, 2)
# ax3 = f.add_subplot(2, 2, 3, projection='3d')
X = stimuliRange1
Y = stimuliRange2
X1, Y1 = np.meshgrid(X, Y)
Xprime = np.repeat(X, Y.size)
Yprime = np.tile(Y, X.size)
likelihoodTrue = {'stim1': Xprime,
                  'stim2': Yprime,
                  'sigmaOto': sigmaOtoGen,
                  'kappaVer': kappaVerGen,
                  'kappaHor': kappaHorGen,
                  'tau': tauGen,
                  'lapse': lapseGen}

# True psychometric surface of subject
liktrue = np.reshape(Am.Lik(likelihoodTrue, psyfun=psyFunctions), (X.size, Y.size))

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

    # ax2.clear()
    # ax2.plot(trial, probe, 'b.', label=r'$s_2$')
    # ax2.plot(trial, ref, 'r.', label=r'$s_1$')
    # ax2.set_xlim(0, nTrials)
    # ax2.set_ylim(0, np.max(stimuliRange1))
    # ax2.set_xlabel('Trial')
    # ax2.set_ylabel('Speed (deg/s)')
    # ax2.set_title('Selected Stimuli')
    # ax2.legend()

    # set stimuli subject receives
    generativeParams['stim1'] = stimuli
    generativeParams['stim2'] = stimuli2

    r = Am.genDat(parameters=generativeParams, psyfun=psyFunctions)  # generate simulated response
    response[i] = r  # store response
    obj.addData(r)  # update Psi with response

    # Use mean estimate of parameters to estimate the subject psychometric response surface
    likelihoodEst = {'stim1': Xprime,
                     'stim2': Yprime,
                     'sigmaOto': obj.summarydata[mod]['parameterProb']['sigmaOto'],
                     'kappaVer': obj.summarydata[mod]['parameterProb']['kappaVer'],
                     'kappaHor': obj.summarydata[mod]['parameterProb']['kappaHor'],
                     'tau': obj.summarydata[mod]['parameterProb']['tau'],
                     'lapse': obj.summarydata[mod]['parameterProb']['lapse']}
    likest = np.reshape(Am.Lik(likelihoodEst, psyfun=psyFunctions), (X.size, Y.size))

    # plotting
    # ax3.clear()
    # ax3.scatter(probe, ref, response, c='r', alpha=0.2, label='Response')
    # ax3.set_xlim(0, 10)
    # ax3.set_ylim(0, 10)
    # ax3.set_zlim(0, 1)
    # ax3.plot_surface(X, Y, likest, rstride=1, cstride=1,
    #                  linewidth=0, antialiased=False, alpha=0.3, label='Estimated')
    # ax3.plot_surface(X, Y, liktrue, rstride=1, cstride=1,
    #                  linewidth=0, antialiased=False, alpha=0.1, color='r', label='True')
    # ax3.scatter(probe[i], ref[i], response[i], c='k', label='Current trial')
    # ax3.view_init(30, 220)
    # ax3.set_title('Predicted vs actual response probability')
    # ax3.set_xlabel(r'$s_1$ (deg/s)')
    # ax3.set_ylabel(r'$s_2$ (deg/s)')
    # ax3.set_zlabel('Response Probability')
    # plt.pause(0.001)

    stimuli = obj.xCurrent  # fetch new stimuli
    stimuli2 = obj.xCurrent2

print "\n##############################\n"
print "Data estimated with following parameters\n"
for mod in obj.prior:
    print "Psychometric function: %s : probability %.2f " % (mod, obj.prior[mod]['modelPrior'])
    print "Parameters %s: " % (obj.summarydata[mod])
print "\n##############################\n"

plt.show()
