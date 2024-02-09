import pickle
from abc import abstractmethod

import numpy as np
from gurobipy import *
import math

class TwoClustersMIP(BaseModel):
    """Skeleton of MIP you have to write as the first exercise.
    You have to encapsulate your code within this class that will be called for evaluation.
    """

    def _init_(self, n_pieces, n_clusters):
        """Initialization of the MIP Variables

        Parameters
        ----------
        n_pieces: int
            Number of pieces for the utility function of each feature.
        n°clusters: int
            Number of clusters to implement in the MIP.
        """
        self.L=n_pieces
        self.K=n_clusters
        self.seed = 123
        # self.model = self.instantiate()
        self.model = Model("UTA model")
        self.criterions = 4

    def instantiate(self):
        """Instantiation of the MIP Variables - To be completed."""
        weights_1 = np.full(self.criterions, 1/self.criterions)
        weights_2 = np.full(self.criterions, 1/self.criterions)
        self.weights = [weights_1, weights_2]
        return

    def fit(self, X, Y):
        """Estimation of the parameters - To be completed.

        Parameters
        ----------
        X: np.ndarray
            (n_samples, n_features) features of elements preferred to Y elements
        Y: np.ndarray
            (n_samples, n_features) features of unchosen elements
        """

        self.breakpoints= [(1/(self.L))*i for i in range(self.L+1)]
       

        P = len(X)
        I={}
        for p in range(P):
            for k in range(self.K):
                I[p, k] = self.model.addVar(vtype=GRB.BINARY, name=f"I_{p}_{k}")
        M = 10
        e = 10**-3
        sigma = {}
        for p in range(P):
            for k in range(self.K):
                sigma[p, k] = self.model.addVar(lb=0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name=f"sigma_{p}_{k}")

        somme=0
        # Variables for utility at each breakpoint
        breakpoint_utils={}
        for k in range(self.K):
            for i in range(self.criterions):
                for b in range(self.L+1):
                    breakpoint_utils[k, i, b] = self.model.addVar(lb=0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name=f"breakpoint_utils_{k}{i}{b}")
        
        # Constraints for linear segments
        for k in range(self.K):
            for i in range(self.criterions):
                for b in range(self.L):
                    self.model.addConstr((breakpoint_utils[k, i, b+1] - breakpoint_utils[k, i, b]) >=0)

        # Function to calculate utility
        def calculate_utility(k, features):
            utility = 0
            for i, feature in enumerate(features):
                for b in range(self.L):
                    if self.breakpoints[b] <= feature < self.breakpoints[b + 1]:
                        utility += breakpoint_utils[k, i, b] + ((breakpoint_utils[k, i, b+1]-breakpoint_utils[k, i, b])/(self.breakpoints[b+1]-self.breakpoints[b])) * (feature - self.breakpoints[b])
                        break
            return utility

        # Utility difference constraint
        for p in range(P):
            self.model.addConstr(sum(I[p, k] for k in range(self.K)) >= 1)
            for k in range(self.K):
                self.model.addConstr(M * (1 - I[p, k]) + calculate_utility(k, X[p]) - calculate_utility(k, Y[p]) - e +sigma[p,k]>= 0)
        for p in range(P):
            for k in range(self.K):
                somme+= sigma[p,k]
        self.model.setObjective(somme, GRB.MINIMIZE)
        self.breakpoint_utils = breakpoint_utils
        self.model.optimize()

        return

    def predict_utility(self, X):
        utilities = np.zeros((len(X), self.K))  # Tableau 2D: lignes pour les échantillons, colonnes pour les clusters
        for p in range(len(X)):
            for k in range(self.K):
                utility = 0
                for i, feature in enumerate(X[p]):
                    for b in range(self.L):
                        if self.breakpoints[b] <= feature < self.breakpoints[b + 1]:
                            # Calculer l'utilité pour chaque cluster séparément
                            utility += self.breakpoint_utils[k, i, b].X + ((self.breakpoint_utils[k, i, b+1].X - self.breakpoint_utils[k, i, b].X) / (self.breakpoints[b+1] - self.breakpoints[b])) * (feature - self.breakpoints[b])
                            break
                utilities[p, k] = utility  # Stocker l'utilité de l'échantillon 'p' pour le cluster 'k'
        return utilities