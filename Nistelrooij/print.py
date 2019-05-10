import numpy as np


class Printer:
    def __init__(self, params, stimuli, genAgent, psi, iterations_num, experiments_mun):
        # set the members of the Printer object
        self.params = params
        self.stimuli = stimuli
        self.iterations_num = iterations_num
        self.experiments_num = experiments_mun

        # save generative agent and psi object in Plotter object
        self.genAgent = genAgent
        self.psi = psi

        self.selected_stimuli = None
        self.param_distributions = None

        # initialize current trial and experiment numbers
        self.trial_num = 1
        self.current_experiment_num = 1


    def reset(self):
        # reset the selected stimuli for new experiment
        self.selected_stimuli = None

        # reset current trial number
        self.trial_num = 1


    def nextTrial(self):
        # increase current trial number
        self.trial_num += 1

        # increase current experiment number at end of each experiment
        if self.trial_num == self.iterations_num:
            self.current_experiment_num += 1


    def printGenVariances(self):
        # get variances from generative agent
        variances = self.genAgent.calcVariances()

        # print data as space separated values
        print '\n\n\nGenerative Variances:\n\n\n'
        print 'frame otoliths context'

        for i in range(len(self.stimuli['frames'])):
            print self.stimuli['frames'][i] * 180 / np.pi, variances['otoliths'][i], variances['context'][i]


    def printGenWeights(self):
        # get weights from generative agent
        weights = self.genAgent.calcWeights()

        # print data as space separated values
        print '\n\n\nGenerative Weights:\n\n\n'
        print 'frame otoliths context'

        for i in range(len(self.stimuli['frames'])):
            print self.stimuli['frames'][i] * 180 / np.pi, weights['otoliths'][i], weights['context'][i]


    def printGenPSE(self):
        # get PSEs from generative agent in degrees
        PSE = self.genAgent.calcPSE()

        # print data as space separated values
        print '\n\n\nGenerative PSE:\n\n\n'
        print 'frame bias'

        for i in range(len(self.stimuli['frames'])):
            print self.stimuli['frames'][i] * 180 / np.pi, PSE[i]


    def printStimuli(self):
        if self.selected_stimuli is None:
            # initialize selected stimuli
            self.selected_stimuli = {'rods': [], 'frames': []}

        # add stimuli to self.selected_stimuli in degrees
        rod, frame = self.psi.stim
        self.selected_stimuli['rods'].append(rod * 180.0 / np.pi)
        self.selected_stimuli['frames'].append(frame * 180.0 / np.pi)

        # print data as space separated values
        if self.trial_num == self.iterations_num:
            print '\n\n\nSelected Stimuli:\n\n\n'
            print 'trial rod frame'

            for i in range(self.iterations_num):
                print (i + 1), self.selected_stimuli['rods'][i] * 180 / np.pi, \
                            self.selected_stimuli['frames'][i] * 180 / np.pi


    def printParameterDistributions(self):
        if self.param_distributions is None:
            # initialize parameter distributions
            self.param_distributions = {param: {'adaptive': [], 'random': []} for param in self.params.keys()}

        # add parameter distributions at end of experiment
        if self.trial_num == self.iterations_num:
            param_distributions = self.psi.calcParameterDistributions()

            # add parameter distributions to self.param_distributions
            for param in self.params.keys():
                self.param_distributions[param][self.psi.stim_selection].append(param_distributions[param])

        if self.current_experiment_num == self.experiments_num:
            for param in self.params.keys():
                self.__printParameterDistribution(param)


    def __printParameterDistribution(self, param):
        distributions = {'adaptive': np.array(self.param_distributions[param]['adaptive']),
                         'random': np.array(self.param_distributions[param]['random'])}

        for stim_selection in ['adaptive', 'random']:
            print '\n\n\n%s %s Distribution:\n\n\n' % (param, stim_selection)
            print 'sample %s mean mean_minus_std mean_plus_std' % param

            for i in range(len(self.params[param])):
                mean = np.mean(distributions[stim_selection][:, i])
                std = np.std(distributions[stim_selection][:, i])
                print (i + 1), self.params[param][i], mean, max(mean - std, 0), min(mean + std, 1)

