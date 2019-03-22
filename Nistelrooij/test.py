import numpy as np
from tqdm import trange

from GenerativeAgent import GenerativeAgent
from PSI_RiF import PSI_RiF
from plots import *


# transforms sigma values into kappa values
def sig2kap(sig):  # in degrees
    sig2 = np.square(sig)
    return 3.9945e3 / (sig2 + 0.0226e3)


kappa_ver = np.linspace(sig2kap(2.3), sig2kap(7.4), 8)
# kappa_ver = [sig2kap(4.9)]
kappa_hor = np.linspace(sig2kap(28), sig2kap(76), 8)
# kappa_hor = [sig2kap(52.0)]
tau = np.linspace(0.6, 1.0, 8)
# tau = [0.8]
kappa_oto = np.linspace(sig2kap(1.4), sig2kap(3.0), 8)
# kappa_oto = [sig2kap(2.2)]
lapse = np.linspace(0.0, 0.1, 8)
# lapse = [0.02]

params = {'kappa_ver': kappa_ver,
          'kappa_hor': kappa_hor,
          'tau': tau,
          'kappa_oto': kappa_oto,
          'lapse': lapse}


kappa_ver_gen = sig2kap(4.9)
kappa_hor_gen = sig2kap(52.0)
tau_gen = 0.8
kappa_oto_gen = sig2kap(2.2)
lapse_gen = 0.02

params_gen = {'kappa_ver': kappa_ver_gen,
              'kappa_hor': kappa_hor_gen,
              'tau': tau_gen,
              'kappa_oto': kappa_oto_gen,
              'lapse': lapse_gen}


rods = np.array([-7, -4, -2, -1, 0, 1, 2, 4, 7]) * np.pi / 180
frames = np.linspace(-45, 40, 18) * np.pi / 180

stimuli = {'rods': rods, 'frames': frames}


# initialize generative agent and show rod distribution plot for each frame orientation
genAgent = GenerativeAgent(params_gen, stimuli)
plotProbTable(genAgent)

# test for adaptive and random stimulus selection
iterations_num = 500
for stim_selection in ['adaptive', 'random']:
    # initialize psi object
    psi = PSI_RiF(params, stimuli, stim_selection)

    # initialize plotter object
    plotter = Plotter(iterations_num, params, params_gen, stimuli)

    # run model for given number of iterations
    print 'inferring model'

    responses = []
    for _ in trange(iterations_num):
        # get stimulus from psi object
        rod, frame = psi.stim

        # plot selected stimuli
        plotter.plotStimuli(psi)

        # plot updated parameter values based on mean
        plotter.plotParameterValues(psi)

        # plot parameter distributions of current trial
        plotter.plotParameterDistributions(psi)

        # get response from the generative model
        response = genAgent.getResponse(rod, frame)

        # add data to psi object
        psi.addData(response)

        # bookkeeping
        responses.append(response)

    # print results
    print 'Parameters of generative model'
    print params_gen
    print 'Parameter values based on MAP'
    print psi.calcParameterValues('MAP')
    print 'Expected parameter values'
    print psi.calcParameterValues('mean')
    print 'First 50 responses'
    print responses[:50], '\n'
