import numpy as np


class Bird:
    def __init__(self, model, fitness):
        self.model = model
        self.fitness = fitness

    def jump(self, bird_height, pipe_height, pipe_distance):
        input = np.asarray([bird_height, pipe_height, pipe_distance])
        input = np.atleast_2d(input)
        output = self.model.predict(input)
        # print(output[0])
        return True if (output[0] > .5) else False
