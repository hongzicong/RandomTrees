# -*- coding: utf-8 -*-
"""
Created on Fri May 31 10:12:16 2019

@author: DELL-PC
"""

import numpy as np
from DecisionTree import DecisionTreeRegressor

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
        self.trees_ = [self._build_tree(X, y) for i in range(self.n_trees)]
    
    
    def _build_tree(self, X, y):
        tree = DecisionTreeRegressor(self.max_features_num, self.max_depth, self.min_samples_split)
        tree.build(X, y)
        return tree
    
    
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