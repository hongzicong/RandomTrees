# -*- coding: utf-8 -*-
"""
Created on Fri May 31 10:12:16 2019

@author: DELL-PC
"""

import numpy as np
from DecisionTree import DecisionTreeRegressor
from sklearn.externals.joblib import Parallel, delayed

class RandomTreesRegressor:
    
    def __init__(self, n_trees, n_processes, max_features_num, max_depth, min_samples_split=2):
        self.n_trees = n_trees
        self.n_processes = n_processes
        self.max_features_num = max_features_num
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
    
    
    def fit(self, X, y):
        '''
        Parameters
        ----------
        X : matrix of shape = [n_samples, n_features]
            The matrix consisting of the input data
            
        y : array of shape = [n_samples]
            The label of X
        '''
        self.trees_ = [DecisionTreeRegressor(self.max_features_num, 
                                             self.max_depth, 
                                             self.min_samples_split) for i in range(self.n_trees)]
        self.trees_ = Parallel(n_jobs=self.n_processes)(
                delayed(_parallel_build)(tree, X, y)
                for i, tree in enumerate(self.trees_))
    
    
    def predict(self, X):
        '''
        Parameters
        ----------
        X : matrix of shape = [n_samples, n_features]
            The matrix consisting of the input data
            
        Returns
        -------
        y : array of shape = [n_samples]
            The predicted data
        '''
        
        y = np.zeros((X.shape[0]), dtype=np.float64)
        
        for tree in self.trees_:
            y += tree.predict(X)
        
        y /= self.n_trees
        
        return y

def _parallel_build(tree, X, y):
    tree.build(X, y)
    return tree