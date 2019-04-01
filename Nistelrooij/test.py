import numpy as np
from tqdm import trange
import matplotlib.pyplot as plt

from GenerativeAgent import GenerativeAgent
from PSI_RiF import PSI_RiF
from plots import Plotter


# transforms sigma values into kappa values
def sig2kap(sig):  # in degrees
    sig2 = np.square(sig)
    return 3.9945e3 / (sig2 + 0.0226e3)


kappa_ver = np.linspace(sig2kap(2.3), sig2kap(7.4), 25)
# kappa_ver = [sig2kap(4.3)]
kappa_hor = np.linspace(sig2kap(28), sig2kap(76), 25)
# kappa_hor = [sig2kap(37)]
# tau = np.linspace(0.6, 1.0, 25)
tau = np.array([0.8])
# kappa_oto = np.linspace(sig2kap(1.4), sig2kap(3.0), 8)
kappa_oto = [sig2kap(2.2)]
# lapse = np.linspace(0.0, 0.1, 8)
lapse = [0.0]

params = {'kappa_ver': kappa_ver,
          'kappa_hor': kappa_hor,
          'tau': tau,
          'kappa_oto': kappa_oto,
          'lapse': lapse}


kappa_ver_gen = sig2kap(4.3)
kappa_hor_gen = sig2kap(37)
tau_gen = 0.8
kappa_oto_gen = sig2kap(2.2)
lapse_gen = 0.0

params_gen = {'kappa_ver': kappa_ver_gen,
              'kappa_hor': kappa_hor_gen,
              'tau': tau_gen,
              'kappa_oto': kappa_oto_gen,
              'lapse': lapse_gen}


rods = np.array([-7, -4, -2, -1, 0, 1, 2, 4, 7]) * np.pi / 180
frames = np.linspace(-45, 40, 18) * np.pi / 180

stimuli = {'rods': rods, 'frames': frames}


# initialize generative agent
genAgent = GenerativeAgent(params_gen, stimuli)

# initialize psi object
psi = PSI_RiF(params, stimuli)

# number of iterations of the experiment
iterations_num = 500

# initialize plotter and plot generative distribution, generative weights and the negative log likelihood
plotter = Plotter(iterations_num, params, params_gen, stimuli, genAgent, psi)
plotter.plotGenProbTable()
plotter.plotGenWeights()
plotter.plotNegLogLikelihood('kappa_ver', 'kappa_hor')
plotter.plot()

for stim_selection in ['adaptive', 'random']:
    # set stimulus selection mode and reset psi object to initial values
    psi.reset(stim_selection)

    # reset plotter to plot new figures
    plotter.reset()

    # run model for given number of iterations
    print 'inferring model ' + stim_selection + 'ly'

    responses = []
    for _ in trange(iterations_num):
        # plot selected stimuli
        plotter.plotStimuli()

        # plot updated parameter values based on mean and MAP
        plotter.plotParameterValues()

        # the parameter distributions may be plotted at most once (so comment out at least one)

        # plot parameter distributions of current trial
        plotter.plotParameterDistributions()

        # plot parameter distributions of each trial as surfaces
        # plotter.plotParameterDistributions('3d')


        # get stimulus from psi object
        rod, frame = psi.stim

        # get response from the generative model
        response = genAgent.getResponse(rod, frame)

        # add data to psi object
        psi.addData(response)


        # the negative log likelihood may be plotted at most once (so comment out at least one)

        # plot negative log likelihood of responses thus far as a contour plot
        plotter.plotNegLogLikelihood('kappa_ver', 'kappa_hor', response_num=1)

        # plot negative log likelihood of responses thus far as a surface
        # plotter.plotNegLogLikelihood('kappa_ver', 'kappa_hor', projection='3d', response_num=1)

        # actually plot all the figures
        plotter.plot()

    # print results
    print 'Parameters of generative model'
    print params_gen
    print 'Parameter values based on MAP'
    print psi.calcParameterValues('MAP')
    print 'Expected parameter values'
    print psi.calcParameterValues('mean')

# do not close plots when program finishes
plt.show()
