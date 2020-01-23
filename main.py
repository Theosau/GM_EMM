## Main code for GMM-EM

import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import numpy as np
from sklearn.datasets.samples_generator import make_blobs
from scipy.stats import multivariate_normal

def data_creation():
    X, y_true = make_blobs(n_samples=400, centers=4,
                       cluster_std=0.60, random_state=0)
    return X #X[:, ::-1]

class GMM_EM():
    def __init__(self, num_functions, data):
        self.num_functions = num_functions
        self.data = data
        self.num_points = self.data.shape[0]
        self.num_features = self.data.shape[1]
        self.functions = [gaussians(self.num_features, self.num_points, self.data) \
                                    for i in range(int(num_functions))]
        self.pi_k = np.random.random(num_functions)
        self.pi_k /= self.pi_k.sum()
        proof = 0
        for k in range(len(self.functions)):
            self.functions[k].pi = self.pi_k[k]

    def E_step(self):
        self.Nk = 0
        for n in range(int(self.num_points)):
            self.sum_rn_k = 0
            self.sum_fcts_rn = 0
            self.sum_fcts_rn_prior = 0
            for k in range((self.num_functions)):
                self.functions[k].rn[n] = self.functions[k].pi * \
                        multivariate_normal(self.functions[k].means, self.functions[k].cov).pdf(self.data[n,:])
                self.sum_rn_k += self.functions[k].rn[n]
            for k in range(self.num_functions):
                self.sum_fcts_rn_prior += self.functions[k].rn[n]
                self.functions[k].rn[n] /= self.sum_rn_k
                self.sum_fcts_rn += self.functions[k].rn[n]
        return

    def M_step(self):
        for k in range(self.num_functions):
            self.Nk = self.functions[k].rn.sum()
            self.functions[k].means = (self.functions[k].rn @ self.data)/self.Nk
            self.weighted_cov = np.zeros((self.num_features, self.num_features))
            self.functions[k].cov = self.weighted_cov
            for n in range(int(self.num_points)):
                self.weighted_cov = self.functions[k].rn[n] * \
                            ((self.data[n,:] - self.functions[k].means).reshape(self.num_features,1)@\
                            (self.data[n,:] - self.functions[k].means).reshape(1,self.num_features))
                self.functions[k].cov += self.weighted_cov
            self.functions[k].cov /= self.Nk
            self.functions[k].pi = self.Nk/self.data.shape[0]
        return


class gaussians():

    def __init__(self, num_features, num_points, data):
        self.data = data
        self.cov = np.cov(self.data.T)
        self.means = np.mean(self.data, axis=0) + 5*np.random.random(int(num_features))
        self.rn = np.ones(int(num_points))
        self.pi = 0
