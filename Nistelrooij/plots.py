import numpy as np
import matplotlib.pyplot as plt


def plotProbTable(genAgent):
    plt.plot(genAgent.rods * 180 / np.pi, genAgent.prob_table)
    plt.title('Generative Rod Distribution for Each Frame Orientation')
    plt.xlabel('rod [deg]')
    plt.ylabel('P(right)')
    plt.pause(0.0001)


class Plotter:

    def __init__(self, iterations_num, params, params_gen, stimuli):
        self.iterations_num = iterations_num

        # set the parameter values and the shape when combining the parameter values
        self.params = params
        self.params_shape = [len(self.params[param]) for param in ['kappa_ver', 'kappa_hor', 'tau', 'kappa_oto', 'lapse']]

        # set the generative parameter values
        self.params_gen = params_gen

        self.stimuli = stimuli

        # initialize selected stimuli and params values and distributions as Nones
        self.selected_stimuli = None
        self.param_values = None
        self.param_distributions = None


    def initStimuliFigure(self):
        # initialize selected stimuli plot
        stimuli_figure = plt.figure()
        self.stimuli_plot = stimuli_figure.add_subplot(1, 1, 1)

        # initialize selected stimuli
        self.selected_stimuli = {'rods': [], 'frames': []}

        # initialize minimum and maximum stimulus values in degrees
        self.stim_min = np.amin(self.stimuli['frames']) * 180 / np.pi
        self.stim_max = np.amax(self.stimuli['frames']) * 180 / np.pi


    def plotStimuli(self, psi):
        if self.selected_stimuli is None:
            self.initStimuliFigure()

        # add stimuli to self.stimuli in degrees
        rod, frame = psi.stim
        self.selected_stimuli['rods'].append(rod * 180.0 / np.pi)
        self.selected_stimuli['frames'].append(frame * 180.0 / np.pi)

        # compute current trial number
        trial_num = len(self.selected_stimuli['rods'])

        # only plot every 10 trials
        if trial_num == 1 or (trial_num % 10) == 0:
            # use shorter name
            plot = self.stimuli_plot

            # plot selected stimuli
            plot.clear()
            plot.scatter(np.arange(trial_num), self.selected_stimuli['rods'], label='rod [deg]')
            plot.scatter(np.arange(trial_num), self.selected_stimuli['frames'], label='frame [deg]')
            plot.set_xlabel('trial number')
            plot.set_ylabel('selected stimulus [deg]')
            plot.set_xlim(0, self.iterations_num)
            plot.set_ylim(self.stim_min, self.stim_max)
            plot.set_title('Selected Stimuli for Each Trial')
            plot.legend(title='Legend')

            # pause to let pyplot draw plot
            plt.pause(0.0001)


    def initParameterValuesFigure(self):
        # initialize calculated parameter values figure and plots
        self.param_values_figure = plt.figure(figsize=(15, 8))
        plots = [self.param_values_figure.add_subplot(2, 3, i) for i in [1, 2, 4, 5, 6]]

        # size of points in parameter values figure
        self.point_size = 5

        # initialize parameter values and parameter values plots dictionaries
        self.param_values = {'MAP': {param: [] for param in self.params.keys()},
                             'mean': {param: [] for param in self.params.keys()}}
        self.param_values_plots = {param: plots[i] for param, i in zip(self.params.keys(), range(len(self.params)))}


    def plotParameterValues(self, psi):
        if self.param_values is None:
            self.initParameterValuesFigure()

        param_values_MAP = psi.calcParameterValues('MAP')
        param_values_mean = psi.calcParameterValues('mean')

        # add parameter values to self.param_values
        for param in self.params.keys():
            self.param_values['MAP'][param].append(param_values_MAP[param])
            self.param_values['mean'][param].append(param_values_mean[param])

        # compute current trial number
        trial_num = len(self.param_values['MAP']['kappa_ver'])

        # only draw plots every 10 trials
        if trial_num == 1 or (trial_num % 10) == 0:
            # draw each parameter's values plot
            for param in self.params.keys():
                self.__plotParemeterValues(param, trial_num)

            # add a single legend to the figure
            handles, labels = self.param_values_plots['kappa_ver'].get_legend_handles_labels()
            self.param_values_figure.legend(handles, labels, loc='upper right', title='Legend')

            # fit all the plots to the screen with no overlapping text
            plt.tight_layout()

            # pause to let pyplot draw graphs
            plt.pause(0.0001)


    def __plotParemeterValues(self, param, trial_num):
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
        self.param_distributions_figure = plt.figure(figsize=(15, 8))
        plots = [self.param_distributions_figure.add_subplot(2, 3, i) for i in [1, 2, 4, 5, 6]]

        # initialize parameter plot dictionary
        self.param_distributions_plots = {param: plots[i] for param, i in zip(self.params.keys(), range(len(self.params)))}

        # distributions are not stored, so trial_num member is needed for this figure.
        self.trial_num = 0


    def plotParameterDistributions(self, psi):
        if self.param_distributions is None:
            self.initParemeterDistributionsFigure()

        self.trial_num += 1

        # only draw plots every 10 trials
        if self.trial_num == 1 or (self.trial_num % 10) == 0:
            # get posterior from psi object in right shape
            posterior = psi.prior.reshape(self.params_shape)

            # compute the distributions of the parameters
            kappa_ver = posterior.sum(4).sum(3).sum(2).sum(1)
            kappa_hor = posterior.sum(4).sum(3).sum(2).sum(0)
            tau = posterior.sum(4).sum(3).sum(1).sum(0)
            kappa_oto = posterior.sum(4).sum(2).sum(1).sum(0)
            lapse = posterior.sum(3).sum(2).sum(1).sum(0)

            # put distributions in dictionary
            self.param_distributions = {'kappa_ver': kappa_ver, 'kappa_hor': kappa_hor, 'tau': tau,
                                        'kappa_oto': kappa_oto, 'lapse': lapse}

            # plot the plot for each parameter
            for param in self.params.keys():
                self.__plotParameterDistributions(param, self.trial_num)

            # add a single legend to the figure with the number of trials
            handles, labels = self.param_distributions_plots['kappa_ver'].get_legend_handles_labels()
            self.param_distributions_figure.legend(handles, labels, loc='upper right', title='Legend')

            # fit all the plots to the screen with no overlapping text
            plt.tight_layout()

            # pause to let pyplot draw graphs
            plt.pause(0.0001)


    def __plotParameterDistributions(self, param, trial_num):
        # retrieve specific parameter plot from self.param_distributions_plots
        plot = self.param_distributions_plots[param]

        # plot specific parameter distribution
        plot.clear()
        plot.vlines(self.params_gen[param], 0.0, 1.0, label='generative parameter value')
        plot.plot(self.params[param], self.param_distributions[param], label='parameter value distribution')
        plot.set_xlabel(param)
        plot.set_ylabel('P(%s = x)' % param)
        plot.set_ylim(0.0, 1.0)
        plot.set_title('%s Distribution for Trial %d' % (param, trial_num))