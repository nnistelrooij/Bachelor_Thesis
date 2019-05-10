import numpy as np


class Printer:
    def __init__(self, params, stimuli, genAgent, psi, iterations_num):
        # set the members of the Printer object
        self.params = params
        self.stimuli = stimuli
        self.iterations_num = iterations_num

        # save generative agent and psi object in Plotter object
        self.genAgent = genAgent
        self.psi = psi

        self.selected_stimuli = None


    def reset(self):
        # reset the selected stimuli for new experiment
        self.selected_stimuli = None

        # reset current trial number
        self.trial_num = 1


    def nextTrial(self):
        # increase current trial number
        self.trial_num += 1


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
                print (i + 1), self.selected_stimuli['rods'][i] * 180 / np.pi,\
                            self.selected_stimuli['frames'][i] * 180 / np.pi
