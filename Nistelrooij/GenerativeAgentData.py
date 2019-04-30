import numpy as np
from itertools import product
import csv


class GenerativeAgentData:
    def __init__(self, path):
        # save data in expandable list
        data = []

        with open(path, 'r') as f:
            # read in comma-separated values from given path
            reader = csv.reader(f)
            for row in reader:
                # data is frame, rod, response; needs to become rod, frame, response
                data_row = [float(row[1]), float(row[0]), int(row[2] == '1.0')]
                data.append(data_row)

        # transform data list to NumPy array
        self.data = np.array(data)


        # Initialize stimulus grids
        rods = np.unique(self.data[:, 0])
        frames = np.unique(self.data[:, 1])

        # dimensions of the 2D stimulus space
        self.rod_num = len(rods)
        self.frame_num = len(frames)
        self.responses_per_stimulus_num = self.data.shape[0] / (self.rod_num * self.frame_num)


        # make 3D list structure for mutable responses
        self.responses = [[[] for _ in range(self.frame_num)] for _ in range(self.rod_num)]

        for data_row in self.data:
            # get rod, frame and response from the data array
            rod, frame, response = data_row[0], data_row[1], data_row[2]

            # find index of stimulus
            idx_rod = np.where(rods == rod)[0][0]
            idx_frame = np.where(frames == frame)[0][0]

            # add response to probability table
            self.responses[idx_rod][idx_frame].append(response)


        self.rods = rods * np.pi / 180
        self.frames = frames * np.pi / 180
        self.last_response = None

        self.deltas = [[0, 1], [1, 0], [0, -1], [-1, 0],
                       [1, 1], [1, -1], [-1, 1], [-1, -1]]


    # determine a response of the generative agent on a given rod and frame orientation
    def getResponse(self, stim_rod, stim_frame, recursion=1):
        # find index of stimulus
        idx_rod = np.where(self.rods == stim_rod)[0][0]
        idx_frame = np.where(self.frames == stim_frame)[0][0]

        response = None

        if not self.responses[idx_rod][idx_frame]:
            if recursion == 0:
                return (stim_rod, stim_frame), None
            else:
                for idx_rod_delta, idx_frame_delta in self.deltas:
                    if response is None and \
                       0 <= (idx_rod + idx_rod_delta) < self.rod_num and \
                       0 <= (idx_frame + idx_frame_delta) < self.frame_num:
                        _, response = self.getResponse(self.rods[idx_rod + idx_rod_delta],
                                                       self.frames[idx_frame + idx_frame_delta],
                                                       recursion - 1)
        else:
            # set response to be one from the data
            response = self.responses[idx_rod][idx_frame].pop()

        # save last response
        self.last_response = response

        return (stim_rod, stim_frame), response