import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange
from collections import OrderedDict

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

# number of iterations of the experiment
iterations_num = 500

# initialize plotter and plot generative distribution, weights, variances and bias and the negative log likelihood
plotter = Plotter(params, params_gen, stimuli, genAgent, psi, iterations_num, plot_period=iterations_num)
plotter.plotGenProbTable()
plotter.plotGenVariances(print_data=True)
plotter.plotGenWeights(print_data=True)
plotter.plotGenPSE(print_data=True)
plotter.plotNegLogLikelihood(responses_num=500)
plotter.plot()


print_param_distribution_data = False
param_distribution_data = {'param': 'lapse', 'adaptive': [], 'random': []}

print_neg_log_likelihood_data = True
neg_log_likelihood_data = {'adaptive': [], 'random': []}


for stim_selection in ['adaptive', 'random']*10:
    # set stimulus selection mode and reset psi object to initial values
    psi.reset(stim_selection)

    # reset plotter to plot new figures
    plotter.reset()

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
        plotter.plotNegLogLikelihood()

        # plot negative log likelihood of responses thus far as a surface
        # plotter.plotNegLogLikelihood(projection='3d')

        # actually plot all the figures
        plotter.plot()


        # add data to psi object
        psi.addData(response)

    # save final distribution/negative log likelihood of current experiment
    if print_neg_log_likelihood_data:
        neg_log_likelihood_data[stim_selection].append(plotter.neg_log_likelihood)
    if print_param_distribution_data:
        param_distribution_data[stim_selection].append(plotter.param_distributions[param_distribution_data['param']][-1])


if print_param_distribution_data:
    param_distribution_data['adaptive'] = np.array(param_distribution_data['adaptive'])
    param_distribution_data['random'] = np.array(param_distribution_data['random'])

    for stim_selection in ['adaptive', 'random']:
        print '\n\n\n%s %s Distribution:\n\n\n' % (param_distribution_data['param'], stim_selection)
        print 'sample %s mean mean_minus_std mean_plus_std' % param_distribution_data['param']

        for i in range(len(params[param_distribution_data['param']])):
            mean = np.mean(param_distribution_data[stim_selection][:, i])
            std = np.std(param_distribution_data[stim_selection][:, i])
            print (i + 1), params[param_distribution_data['param']][i], mean, max(mean - std, 0), min(mean + std, 1)


if print_neg_log_likelihood_data:
    neg_log_likelihood_data['adaptive'] = np.array(neg_log_likelihood_data['adaptive'])
    neg_log_likelihood_data['random'] = np.array(neg_log_likelihood_data['random'])

    for stim_selection in ['adaptive', 'random']:
        print '\n\n\n%s-%s %s Negative log Likelihood:\n\n\n' % (plotter.free_param1, plotter.free_param2, stim_selection)
        print '%s_sample %s %s_sample %s mean std' % (plotter.free_param1, plotter.free_param1,
                                                      plotter.free_param2, plotter.free_param2)

        neg_log_likelihood_min = float('inf')
        neg_log_likelihood_max = float('-inf')

        for i in range(len(params[plotter.free_param2])):
            for j in range(len(params[plotter.free_param1])):
                mean = np.mean(neg_log_likelihood_data[stim_selection][:, i, j])
                std = np.std(neg_log_likelihood_data[stim_selection][:, i, j])
                print (j + 1), params[plotter.free_param1][j], (i + 1), params[plotter.free_param2][i], mean, std

                neg_log_likelihood_min = min(neg_log_likelihood_min, mean)
                neg_log_likelihood_max = max(neg_log_likelihood_max, mean)

        print '\n\n\nMinimum and Maximum values of Mean Negative log Likelihood'
        print neg_log_likelihood_min, neg_log_likelihood_max


        param_values_at_minimum = np.empty([10, 2])
        for i in range(10):
            min_flat_index = np.argmin(neg_log_likelihood_data[stim_selection][i])
            min_indices = np.unravel_index(min_flat_index, neg_log_likelihood_data[stim_selection][i].shape)
            param_values_at_minimum[i] = [params[plotter.free_param1][min_indices[1]], params[plotter.free_param2][min_indices[0]]]

        print '\n\n\n%s Mean Values at Minimum of Negative log Likelihood' % stim_selection
        mean_param1 = np.mean(param_values_at_minimum[:, 0])
        mean_param2 = np.mean(param_values_at_minimum[:, 1])
        print mean_param1, mean_param2

        print '\n\n\n%s Standard Deviations at Minimum of Negative log Likelihood' % stim_selection
        std_param1 = np.std(param_values_at_minimum[:, 0])
        std_param2 = np.std(param_values_at_minimum[:, 1])
        print std_param1, std_param2


# do not close plots when program finishes
plt.show()
