## Main code for GMM-EM

import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import numpy as np
from sklearn.datasets.samples_generator import make_blobs

def data_creation():
    X, y_true = make_blobs(n_samples=400, centers=4,
                       cluster_std=0.60, random_state=0)
    return X[:, ::-1]

class GMM_EM():
    def __init__(self, num_means, data):
        self.num_functions = num_functions
        self.data = data
        self.functions = [gaussians(data.shape[1]) for i in range(int(num_means))]
        self.pi_k = np.random.random(num_functions)
        self.pi_k = (self.pi_k - max(self.pi_k))/(max(self.pi_k) - min(self.pi_k))
        for i in range(len(self.functions)):
            self.functions[i].pi_k = self.pi_k[i]
        self.rn_k = np.zeros((data.shape[0], data.num_functions))

    def E_step(self):
        self.sum_rn_k = 0
        for n in range(int(data.shape[0])):
            for k in range((data.num_functions)):
                self.functions[k].rn_k[n,k] = self.functions[k].pi_k * np.random.multivariate_normal(mean, cov)
                self.sum_rn_k += self.functions[k].rn_k[n,k]
        for i range(num_functions):
            self.functions[i].normalize_rn_k(self.sum_rn_k)
        return

    def M_step(self):
        

class gaussians():

    def __init__(self, num_features):
        self.means = np.random.random(int(self.num_features))
        self.rn_k = np.zeros((int(data.shape[0]), int(self.num_means)))
        self.cov_k = np.random.random((int(data.shape[0]),int(data.shape[0])))
        self.pi_k = 0

    def normalize_rn_k(self, norm_factor):
        self.rn_k /= norm_factor
        return
