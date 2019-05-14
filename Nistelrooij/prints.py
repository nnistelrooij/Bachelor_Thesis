import numpy as np


class Printer:
    def __init__(self, params, stimuli, genAgent, psi, iterations_num, experiments_mun):
        # set the members of the Printer object
        self.params = params
        self.stimuli = stimuli
        self.iterations_num = iterations_num
        self.experiments_num = experiments_mun

        # save generative agent and psi object in Printer object
        self.genAgent = genAgent
        self.psi = psi

        # initialize data members as Nones
        self.param_distributions = None
        self.param_sds = None
        self.neg_log_likelihood = None

        # initialize current experiment number
        self.current_experiment_num = 0

        # reset printer for new experiment
        self.reset()


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
                print (i + 1), self.selected_stimuli['rods'][i], self.selected_stimuli['frames'][i]


    def printParameterDistributions(self):
        if self.param_distributions is None:
            # initialize parameter distributions
            self.param_distributions = {param: {'adaptive': [], 'random': []} for param in self.params.keys()}

        # add parameter distributions at end of each experiment
        if self.trial_num == self.iterations_num:
            param_distributions = self.psi.calcParameterDistributions()

            # add parameter distributions to self.param_distributions
            for param in self.params.keys():
                self.param_distributions[param][self.psi.stim_selection].append(param_distributions[param])

        # print data at end of all experiments
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


    def printParameterStandardDeviations(self):
        if self.param_sds is None:
            # initialize parameter standard deviations dictionary
            self.param_sds = {param: {'adaptive': [], 'random': []} for param in self.params.keys()}

        # initialize parameter standard deviations at start of each experiment
        if self.trial_num == 1:
            for param in self.params.keys():
                self.param_sds[param][self.psi.stim_selection].append([])

        param_sds = self.psi.calcParameterStandardDeviations()

        # add parameter standard deviations to self.param_sds
        for param in self.params.keys():
            self.param_sds[param][self.psi.stim_selection][-1].append(param_sds[param])

        # print data at end of all experiments
        if self.current_experiment_num == self.experiments_num:
            for param in self.params.keys():
                self.__printParameterStandardDeviations(param)


    def __printParameterStandardDeviations(self, param):
        param_sds = {'adaptive': np.array(self.param_sds[param]['adaptive']),
                     'random': np.array(self.param_sds[param]['random'])}

        for stim_selection in ['adaptive', 'random']:
            print '\n\n\n%s %s Standard Deviations:\n\n\n' % (param, stim_selection)
            print 'trial mean mean_minus_std mean_plus_std'

            for i in range(self.iterations_num):
                mean = np.mean(param_sds[stim_selection][:, i])
                std = np.std(param_sds[stim_selection][:, i])
                print (i + 1), mean, max(mean - std, 0), min(mean + std, 0.31914)


    def printNegLogLikelihood(self):
        if self.neg_log_likelihood is None:
            # initialize negative log likelihood dictionary
            self.__initNegLogLikelihoodData()

        # initialize negative log likelihood at start of each experiment
        if self.trial_num == 1:
            self.neg_log_likelihood[self.psi.stim_selection].append(np.zeros(self.neg_log_likelihood_shape))

        # data is latest response from generative agent
        data = self.genAgent.last_response

        # get negative log likelihood from psi object given the data
        neg_log_likelihood = self.psi.calcNegLogLikelihood(data)
        self.neg_log_likelihood[self.psi.stim_selection][-1] += neg_log_likelihood.reshape(self.neg_log_likelihood_shape)

        if self.current_experiment_num == self.experiments_num:
            self.neg_log_likelihood['adaptive'] = np.array(self.neg_log_likelihood['adaptive'])
            self.neg_log_likelihood['random'] = np.array(self.neg_log_likelihood['random'])

            for stim_selection in ['adaptive', 'random']:
                self.__printNegLogLikelihood(stim_selection)
                self.__printNegLogLikelihoodMinMax(stim_selection)
                self.__printMinimumOfNegLogLikelihood(stim_selection)


    def __initNegLogLikelihoodData(self):
        # initialize the two free parameters, self.free_param1 and self.free_param2
        free_params = [param for param in self.params.keys() if len(self.params[param]) > 1]
        if len(free_params) == 2:
            self.free_param1, self.free_param2 = free_params
            self.neg_log_likelihood_shape = (len(self.params[self.free_param1]), len(self.params[self.free_param2]))
        else:
            raise Exception, 'model must have exactly two free parameters'

        # initialize negative log likelihood array
        self.neg_log_likelihood = {'adaptive': [], 'random': []}


    def __printNegLogLikelihood(self, stim_selection):
        print '\n\n\n%s-%s %s Negative log Likelihood:\n\n\n' % (self.free_param1, self.free_param2, stim_selection)
        print '%s_sample %s %s_sample %s mean std' % (self.free_param1, self.free_param1, self.free_param2, self.free_param2)

        for i in range(self.neg_log_likelihood_shape[0]):
            for j in range(self.neg_log_likelihood_shape[1]):
                mean = np.mean(self.neg_log_likelihood[stim_selection][:, i, j])
                std = np.std(self.neg_log_likelihood[stim_selection][:, i, j])
                print (i + 1), self.params[self.free_param1][i], (j + 1), self.params[self.free_param2][j], mean, std


    def __printNegLogLikelihoodMinMax(self, stim_selection):
        mean_neg_log_likelihood = np.mean(self.neg_log_likelihood[stim_selection], axis=0)
        neg_log_likelihood_min = np.amin(mean_neg_log_likelihood)
        neg_log_likelihood_max = np.amax(mean_neg_log_likelihood)

        print '\n\n\nMinimum and Maximum values of Mean Negative log Likelihood'
        print neg_log_likelihood_min, neg_log_likelihood_max


    def __printMinimumOfNegLogLikelihood(self, stim_selection):
        stim_selection_experiments_num = self.neg_log_likelihood[stim_selection].shape[0]
        param_values_at_minimum = np.empty([stim_selection_experiments_num, 2])

        for i in range(stim_selection_experiments_num):
            min_flat_index = np.argmin(self.neg_log_likelihood[stim_selection][i])
            min_param_indices = np.unravel_index(min_flat_index, self.neg_log_likelihood_shape)
            param_values_at_minimum[i] = [self.params[self.free_param1][min_param_indices[0]],
                                          self.params[self.free_param2][min_param_indices[1]]]

        print '\n\n\n%s Mean Values at Minimum of Negative log Likelihood' % stim_selection
        min_mean_free_param1 = np.mean(param_values_at_minimum[:, 0])
        min_mean_free_param2 = np.mean(param_values_at_minimum[:, 1])
        print min_mean_free_param1, min_mean_free_param2

        print '\n\n\n%s Standard Deviations at Minimum of Negative log Likelihood' % stim_selection
        min_std_free_param1 = np.std(param_values_at_minimum[:, 0])
        min_std_free_param2 = np.std(param_values_at_minimum[:, 1])
        print min_std_free_param1, min_std_free_param2

