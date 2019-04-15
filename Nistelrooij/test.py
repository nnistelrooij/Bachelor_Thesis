import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange

from GenerativeAgent import GenerativeAgent
from PSI_RiF import PSI_RiF
from plots import Plotter


# transforms sigma values into kappa values
def sig2kap(sig):  # in degrees
    sig2 = np.square(sig)
    return 3994.5 / (sig2 + 22.6)


kappa_ver = sig2kap(np.linspace(10.01, 0.0, 25))
# kappa_ver = [sig2kap(4.87)]
kappa_hor = sig2kap(np.linspace(99.12, 5.4, 25))
# kappa_hor = [sig2kap(52.26)]
# tau = np.linspace(0.58, 1, 25)
tau = np.array([0.8])
# kappa_oto = sig2kap(np.linspace(2.71, 1.71, 25))
kappa_oto = [sig2kap(2.21)]
# lapse = np.linspace(0.0, 0.06, 25)
lapse = [0.02]

params = {'kappa_ver': kappa_ver,
          'kappa_hor': kappa_hor,
          'tau': tau,
          'kappa_oto': kappa_oto,
          'lapse': lapse}


# kappa_ver_gen = sig2kap(4.3)
# kappa_hor_gen = sig2kap(37)
# tau_gen = 0.8
# kappa_oto_gen = sig2kap(2.2)
# lapse_gen = 0.0

# control
kappa_ver_gen = sig2kap(4.87)
kappa_hor_gen = sig2kap(52.26)
tau_gen = 0.8
kappa_oto_gen = sig2kap(2.21)
lapse_gen = 0.02

# dysfunctional
# kappa_ver_gen = sig2kap(4.99)
# kappa_hor_gen = sig2kap(67.51)
# tau_gen = 0.87
# kappa_oto_gen = sig2kap(5.92)
# lapse_gen = 0.05

# functional
# kappa_ver_gen = sig2kap(8.09)
# kappa_hor_gen = sig2kap(39.28)
# tau_gen = 0.97
# kappa_oto_gen = sig2kap(5.75)
# lapse_gen = 0.02

params_gen = {'kappa_ver': kappa_ver_gen,
              'kappa_hor': kappa_hor_gen,
              'tau': tau_gen,
              'kappa_oto': kappa_oto_gen,
              'lapse': lapse_gen}


rods = np.array([-7, -4, -2, -1, 0, 1, 2, 4, 7]) * np.pi / 180
frames = np.linspace(-45, 40, 18) * np.pi / 180

# rods = np.linspace(-9, 9, 1000) * np.pi / 180
# frames = np.linspace(-45, 45, 1000) * np.pi / 180

stimuli = {'rods': rods, 'frames': frames}


# initialize generative agent
genAgent = GenerativeAgent(params_gen, stimuli)

# initialize psi object
psi = PSI_RiF(params, stimuli)

# number of iterations of the experiment
iterations_num = 500

# initialize plotter and plot generative distribution, generative weights and the negative log likelihood
plotter = Plotter(params, params_gen, stimuli, genAgent, psi, iterations_num, plot_period=500)
# plotter.plotGenProbTable()
# plotter.plotGenVariances()
# plotter.plotGenWeights()
# plotter.plotGenPSE()
plotter.plotNegLogLikelihood(responses_num=500)
plotter.plot()

data = {}
data['adaptive'] = []
data['random'] = []

for stim_selection in ['adaptive', 'random']*10:
    # set stimulus selection mode and reset psi object to initial values
    psi.reset(stim_selection)

    # reset plotter to plot new figures
    plotter.reset()

    # run model for given number of iterations
    print 'inferring model ' + stim_selection + 'ly'

    for _ in range(iterations_num):
        # get stimulus from psi object
        rod, frame = psi.stim

        # get response from the generative model
        response = genAgent.getResponses(rod, frame, 1)


        # plot selected stimuli
        # plotter.plotStimuli()

        # plot updated parameter values based on mean and MAP
        # plotter.plotParameterValues()

        # the parameter distributions may be plotted at most once (so comment out at least one)

        # plot parameter distributions of current trial
        # plotter.plotParameterDistributions()

        # plot parameter distributions of each trial as surfaces
        # plotter.plotParameterDistributions(projection='3d')

        # the negative log likelihood may be plotted at most once (so comment out at least one)

        # plot negative log likelihood of responses thus far as a contour plot
        plotter.plotNegLogLikelihood()

        # plot negative log likelihood of responses thus far as a surface
        # plotter.plotNegLogLikelihood(projection='3d')

        # actually plot all the figures
        # plotter.plot()


        # add data to psi object
        psi.addData(response)

    data[stim_selection].append(plotter.neg_log_likelihood)

data['adaptive'] = np.array(data['adaptive'])
data['random'] = np.array(data['random'])

# for i in range(len(lapse)):
#     mean = np.mean(data['adaptive'][:, i])
#     std = np.std(data['adaptive'][:, i])
#     print (i + 1), lapse[i], mean, max(mean - std, 0), min(mean + std, 1)
#
# print '\n\n\n\n\n\n'
# for i in range(len(lapse)):
#     mean = np.mean(data['random'][:, i])
#     std = np.std(data['random'][:, i])
#     print (i + 1), lapse[i], mean, max(mean - std, 0), min(mean + std, 1)


for i in range(25):
    for j in range(25):
        mean = np.mean(data['adaptive'][:, i, j])
        std = np.std(data['adaptive'][:, i, j])
        print (j + 1), kappa_ver[j], (i + 1), kappa_hor[i], mean, std
print '\n\n\n\n\n'
for i in range(25):
    for j in range(25):
        mean = np.mean(data['random'][:, i, j])
        std = np.std(data['random'][:, i, j])
        print (j + 1), kappa_ver[j], (i + 1), kappa_hor[i], mean, std

# do not close plots when program finishes
plt.show()
