# -*- coding: utf-8 -*-
"""
Created on Fri May 31 15:45:59 2019

@author: DELL-PC
"""

import numpy as np

class DecisionTreeRegressor:
    
    def __init__(self, max_features_num, max_depth, min_impurity_split=1e-7):
        self.max_features_num = max_features_num
        self.max_depth = max_depth
        self.min_impurity_split = min_impurity_split
    
    def build(self, X, y, height):
        '''
        Parameters
        ----------
        X : matrix of shape = [n_samples, n_features]
            The matrix consisting of the input data
            
        y : array of shape = [n_samples]
            The label of X
        '''
        bestfeature, bestValue = self.select_best_feature(X, y)
        if bestfeature == None:
            return bestValue
        
        tree = {}
        
        height -= 1
        if height < 0:
            return self.avg_data(y)
        
        tree['bestFeature'] = bestfeature
        tree['bestVal'] = bestValue
        
        left_X, left_y, right_X, right_y = self.split_data_set(X, y, bestfeature, bestValue)
        tree['right'] = self.build(right_X, right_y, height)
        tree['left'] = self.build(left_X, left_y, height)
        return tree

    
    def mse_data(self, label):
        return np.var(label) * np.shape(label)[0]
    
    
    def avg_data(self, label):
        return np.mean(label)
    
    
    def split_data_set(self, X, y, feature, value):
        left_index = np.nonzero(X[:, feature] > value)[0]
        right_index = np.nonzero(X[:, feature] < value)[0]
        return X[left_index, :], y[left_index, :], X[right_index, :], y[right_index, :]
    
    
    def select_best_feature(self, X, y):
        '''
        Parameters
        ----------
        X : matrix of shape = [n_samples, n_features]
            The matrix consisting of the input data
            
        y : array of shape = [n_samples]
            The label of X
        '''
        n_features = X.shape[1]
        features_index = []
        
        best_MSE = np.inf
        best_feature = 0
        best_value = 0
        
        MSE = self.mse_data(X, y)

        for i in range(self.max_features_num):
            features_index.append(np.random.randint(n_features))
        
        for feature in features_index:
            for value in set(X[:, feature]):
                left_X, left_y, right_X, right_y = self.split_data_set(X, y, feature, value)

                new_MSE = self.MSE_data(left_y) + self.MSE_data(right_y)
                
                if best_MSE > new_MSE:
                    best_feature = feature
                    best_value = value
                    best_MSE = new_MSE
        
        if (MSE - best_MSE) < self.min_impurity_split:
            return None, self.avg_data(y)
        
        return best_feature, best_value