import numpy as np
import matplotlib.pyplot as plt

def plotProbTable(rods, prob_table):
    plt.plot(rods * 180 / np.pi, prob_table)
    plt.title('Generative Rod Distribution for Each Frame Orientation')
    plt.xlabel('rod [deg]')
    plt.ylabel('P(right)')
    plt.pause(0.001)


class Plotter:

    def __init__(self, iterations_num, params, params_gen):
        self.iterations_num = iterations_num

        # set the parameter values and the shape when combining the parameter values
        self.params = params
        self.params_shape = [len(self.params[param]) for param in ['kappa_ver', 'kappa_hor', 'tau', 'kappa_oto', 'lapse']]

        # set the generative parameter values
        self.params_gen = params_gen

        # initialize stimuli and params as Nones
        self.stimuli = None
        self.param_values = None
        self.param_distributions = None


    def initStimuliFigure(self):
        # initialize selected stimuli plot
        figure_stimuli = plt.figure()
        self.plot_stimuli = figure_stimuli.add_subplot(1, 1, 1)

        # initialize stimuli
        self.stimuli = ([], [])


    def plotStimuli(self, psi):
        if self.stimuli is None:
            self.initStimuliFigure()

        # add stimuli to self.stimuli as degrees
        rod, frame = psi.stim
        self.stimuli[0].append(rod * 180.0 / np.pi)
        self.stimuli[1].append(frame * 180.0 / np.pi)

        # compute current trial number
        trial_num = len(self.stimuli[0])

        # plot selected stimuli
        self.plot_stimuli.clear()
        self.plot_stimuli.scatter(np.arange(trial_num), self.stimuli[0], label='rod [deg]')
        self.plot_stimuli.scatter(np.arange(trial_num), self.stimuli[1], label='frame [deg]')
        self.plot_stimuli.set_xlabel('trial number')
        self.plot_stimuli.set_ylabel('selected stimulus [deg]')
        self.plot_stimuli.set_xlim(0, self.iterations_num)
        self.plot_stimuli.set_title('Selected Stimuli for Each Trial')
        self.plot_stimuli.legend()

        # pause to let pyplot draw plot
        plt.pause(0.0001)


    def initParameterValuesFigure(self):
        # initialize calculated parameter values plots
        self.figure_param_values = plt.figure(figsize=(15, 8))
        plot_kappa_ver = self.figure_param_values.add_subplot(2, 3, 1)
        plot_kappa_hor = self.figure_param_values.add_subplot(2, 3, 2)
        plot_tau = self.figure_param_values.add_subplot(2, 3, 4)
        plot_kappa_oto = self.figure_param_values.add_subplot(2, 3, 5)
        plot_lapse = self.figure_param_values.add_subplot(2, 3, 6)

        # size of points in parameter values graph
        self.point_size = 5

        # initialize parameter values and parameter plots dictionaries
        self.param_values = {'MAP': {'kappa_ver': [], 'kappa_hor': [], 'tau': [], 'kappa_oto': [], 'lapse': []},
                             'mean': {'kappa_ver': [], 'kappa_hor': [], 'tau': [], 'kappa_oto': [], 'lapse': []}}
        self.param_values_plots = {'kappa_ver': plot_kappa_ver, 'kappa_hor': plot_kappa_hor, 'tau': plot_tau,
                            'kappa_oto': plot_kappa_oto, 'lapse': plot_lapse}


    def plotParameterValues(self, psi):
        if self.param_values is None:
            self.initParameterValuesFigure()

        param_values_MAP = psi.calcParameterValues('MAP')
        param_values_mean = psi.calcParameterValues('mean')

        # draw each parameter's values plot
        for param in self.params.keys():
            # add parameter value to self.param_values
            self.param_values['MAP'][param].append(param_values_MAP[param])
            self.param_values['mean'][param].append(param_values_mean[param])

            # plot specific parameter's values graph
            self.__plotParemeterValues(param)

        # add a single legend to the figure
        handles, labels = self.param_values_plots['kappa_ver'].get_legend_handles_labels()
        self.figure_param_values.legend(handles, labels, loc='upper right')

        # fit all the plots to the screen with no overlapping text
        plt.tight_layout()

        # pause to let pyplot draw graphs
        plt.pause(0.0001)


    def __plotParemeterValues(self, param):
        # compute current trial number
        trial_num = len(self.param_values['MAP'][param])

        # retrieve specific parameter plot from self.param_values_plots
        plot = self.param_values_plots[param]

        # plot specific parameter values
        plot.clear()
        plot.hlines(self.params_gen[param], 1, self.iterations_num, label='generative parameter value')
        plot.scatter(np.arange(trial_num), self.param_values['MAP'][param], s=self.point_size, label='parameter value based on MAP')
        plot.scatter(np.arange(trial_num), self.param_values['mean'][param], s=self.point_size, label='expected parameter value')
        plot.set_xlabel('trial number')
        plot.set_ylabel(param)
        plot.set_xlim(0, self.iterations_num)
        plot.set_title('Calculated %s Values for Each Trial' % param)


    def initParemeterDistributionsFigure(self):
        # initialize calculated parameter values plots
        self.figure_param_distributions = plt.figure(figsize=(15, 8))
        plot_kappa_ver = self.figure_param_distributions.add_subplot(2, 3, 1)
        plot_kappa_hor = self.figure_param_distributions.add_subplot(2, 3, 2)
        plot_tau = self.figure_param_distributions.add_subplot(2, 3, 4)
        plot_kappa_oto = self.figure_param_distributions.add_subplot(2, 3, 5)
        plot_lapse = self.figure_param_distributions.add_subplot(2, 3, 6)

        # initialize parameter plot dictionary
        self.param_distributions_plots = {'kappa_ver': plot_kappa_ver, 'kappa_hor': plot_kappa_hor, 'tau': plot_tau,
                            'kappa_oto': plot_kappa_oto, 'lapse': plot_lapse}


    def plotParameterDistributions(self, psi):
        if self.param_distributions is None:
            self.initParemeterDistributionsFigure()

        # get posterior from psi object in right shape
        posterior = psi.prior.reshape(self.params_shape)

        # compute the distributions of the parameters
        kappa_ver = posterior.sum(4).sum(3).sum(2).sum(1)
        kappa_hor = posterior.sum(4).sum(3).sum(2).sum(0)
        tau = posterior.sum(4).sum(3).sum(1).sum(0)
        kappa_oto = posterior.sum(4).sum(2).sum(1).sum(0)
        lapse = posterior.sum(3).sum(2).sum(1).sum(0)

        # put distributions in dictionary
        self.param_distributions = {'kappa_ver': kappa_ver, 'kappa_hor': kappa_hor, 'tau': tau, 'kappa_oto': kappa_oto,
                                    'lapse': lapse}

        # plot the plot for each parameter
        for param in self.params.keys():
            # retrieve specific parameter plot from self.param_distributions_plots
            plot = self.param_distributions_plots[param]

            # plot specific parameter distribution
            plot.clear()
            plot.vlines(self.params_gen[param], 0, 1, label='generative parameter value')
            plot.plot(self.params[param], self.param_distributions[param], label='parameter value distribution')
            plot.set_xlabel(param)
            plot.set_ylabel('probability')
            plot.set_title('%s Distribution for Current Trial' % param)

        # add a single legend to the figure
        handles, labels = self.param_distributions_plots['kappa_ver'].get_legend_handles_labels()
        self.figure_param_distributions.legend(handles, labels, loc='upper right')

        # fit all the plots to the screen with no overlapping text
        plt.tight_layout()

        # pause to let pyplot draw graphs
        plt.pause(0.0001)