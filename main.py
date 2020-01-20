## Main code for GMM-EM

import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import numpy as np
from sklearn.datasets.samples_generator import make_blobs

def data_creation():
    X, y_true = make_blobs(n_samples=400, centers=4,
                       cluster_std=0.60, random_state=0)
    return X[:, ::-1]

class gaussians():

    def __init__(self, mean, covariance):
        self.mean = mean
        self.covariance = covariance

        
