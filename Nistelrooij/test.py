import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange
from collections import OrderedDict

from GenerativeAgent import GenerativeAgent
from PSI_RiF import PSI_RiF
from plots import Plotter
from prints import Printer


# transforms sigma values into kappa values
def sig2kap(sig):  # in degrees
    sig2 = np.square(sig)
    return 3994.5 / (sig2 + 22.6)


kappa_ver = sig2kap(np.linspace(10.01, 0.0, 25))
# kappa_ver = [sig2kap(4.87)]
kappa_hor = sig2kap(np.linspace(99.12, 5.4, 25))
# kappa_hor = [sig2kap(52.26)]
# tau = np.linspace(0.58, 1, 25)
tau = [0.8]
# kappa_oto = sig2kap(np.linspace(2.71, 1.71, 25))
kappa_oto = [sig2kap(2.21)]
# lapse = np.linspace(0.0, 0.06, 25)
lapse = [0.02]

params = OrderedDict()
params['kappa_ver'] = kappa_ver
params['kappa_hor'] = kappa_hor
params['tau'] = tau
params['kappa_oto'] = kappa_oto
params['lapse'] = lapse


# control
kappa_ver_gen = sig2kap(4.87)
kappa_hor_gen = sig2kap(52.26)
tau_gen = 0.8
kappa_oto_gen = sig2kap(2.21)
lapse_gen = 0.02

# functional
# kappa_ver_gen = sig2kap(8.09)
# kappa_hor_gen = sig2kap(39.28)
# tau_gen = 0.97
# kappa_oto_gen = sig2kap(5.75)
# lapse_gen = 0.02

# dysfunctional
# kappa_ver_gen = sig2kap(4.99)
# kappa_hor_gen = sig2kap(67.51)
# tau_gen = 0.87
# kappa_oto_gen = sig2kap(5.92)
# lapse_gen = 0.05

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

# number of iterations of the experiment and number of experiments
iterations_num = 500
experiments_num = 20

# initialize plotter and plot generative distribution, weights, variances and bias and the negative log likelihood
plotter = Plotter(params, params_gen, stimuli, genAgent, psi, iterations_num, plot_period=25)
plotter.plotGenProbTable()
plotter.plotGenVariances()
plotter.plotGenWeights()
plotter.plotGenPSE()
plotter.plotNegLogLikelihood(responses_num=500)
plotter.plot()


# initialize printer and print generative variances, weights and bias
printer = Printer(params, stimuli, genAgent, psi, iterations_num, experiments_num)
printer.printGenVariances()
printer.printGenWeights()
printer.printGenPSE()

for stim_selection in ['adaptive', 'random']*(experiments_num / 2):
    # set stimulus selection mode and reset psi object to initial values
    psi.reset(stim_selection)

    # reset plotter to plot new figures
    plotter.reset()

    # reset printer for new experiment
    printer.reset()

    # run model for given number of iterations
    print 'inferring model ' + stim_selection + 'ly'

    for _ in trange(iterations_num):
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
        # plotter.plotNegLogLikelihood()

        # plot negative log likelihood of responses thus far as a surface
        # plotter.plotNegLogLikelihood(projection='3d')

        # plot parameter value distribution variances of each trial
        plotter.plotParameterVariances()

        # actually plot all the figures
        # plotter.plot()


        # print selected stimuli data
        # printer.printStimuli()

        # print parameter distributions data
        printer.printParameterDistributions()

        # print negative log likelihood data
        printer.printNegLogLikelihood()

        # progress the printer to the next trial
        printer.nextTrial()



        # add data to psi object
        psi.addData(response)


# do not close plots when program finishes
plt.show()
