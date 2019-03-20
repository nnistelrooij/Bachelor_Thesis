import numpy as np
import matplotlib.pyplot as plt

def plotProbTable(rods, prob_table):
    plt.plot(rods * 180 / np.pi, prob_table)
    plt.title('Generative Rod Distribution for each Frame Orientation')
    plt.xlabel('rod [deg]')
    plt.ylabel('P(right)')
    plt.pause(0.001)


class StimuliPlotter:
    def __init__(self, iterations_num, stim_min, stim_max):
        self.iterations_num = iterations_num
        self.stim_min = stim_min * 180 / np.pi
        self.stim_max = stim_max * 180 / np.pi

        plt.ion()
        figure = plt.figure()
        self.plot = figure.add_subplot(1, 1, 1)

        self.stimuli = ([], [])

    def plotStimuli(self, rod, frame):
        self.stimuli[0].append(rod * 180.0 / np.pi)
        self.stimuli[1].append(frame * 180.0 / np.pi)
        trial_num = len(self.stimuli[0])

        self.plot.clear()
        self.plot.plot(np.arange(trial_num), self.stimuli[0], 'b.', label='rod [deg]')
        self.plot.plot(np.arange(trial_num), self.stimuli[1], 'r.', label='frame [deg]')
        self.plot.set_xlabel('trial')
        self.plot.set_ylabel('selected stimulus [deg]')
        self.plot.set_xlim(0, self.iterations_num)
        self.plot.set_ylim(self.stim_min, self.stim_max)
        self.plot.set_title('Selected Stimuli')
        self.plot.legend()

        plt.pause(0.0001)



def plotP(rods, P, title):
    plt.figure()
    plt.plot(rods * 180 / np.pi, P)
    plt.title(title)
    plt.xlabel('rod [deg]')
    plt.ylabel('P(right)')

def show():
    plt.show()