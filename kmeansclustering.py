import numpy as np

class KMeansClustering: 
    def __init__(self, inputs, k): 
        self.inputs = inputs
        self.k = k
        choices = np.random.choice(inputs.shape[0], k, False)
        self.centroids = inputs[choices]

    def assignments(): 
        a