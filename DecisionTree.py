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
        
        
    def build(self, X, y):
        self.tree = self.__build(X, y, self.max_depth)
        
    
    def __build(self, X, y, height):
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

    
    def predict(self, X):
        return self.__predict(self.tree, X)


    def __predict(self, tree, X):
        if not isinstance(tree, dict):
            return float(tree)
        if X[tree['bestFeature']] > tree['bestVal']:
            if type(tree['left']) == 'float':
                return tree['left']
            else:
                return self.__predict(tree['left'], X)
        else:
            if type(tree['right'])=='float':
                return tree['right']
            else:
                return self.__predict(tree['right'], X)

    
    def mse_data(self, label):
        return np.var(label) * np.shape(label)[0]
    
    
    def avg_data(self, label):
        return np.mean(label)
    
    
    def split_data_set(self, X, y, feature, value):
        left_index = np.nonzero(X.iloc[:, feature] > value)[0]
        right_index = np.nonzero(X.iloc[:, feature] < value)[0]
        return X.iloc[left_index, :], y.iloc[left_index, :], X.iloc[right_index, :], y.iloc[right_index, :]
    
    
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
        
        MSE = self.mse_data(y)

        for i in range(self.max_features_num):
            features_index.append(np.random.randint(n_features))
        
        for feature in features_index:
            for value in set(X.iloc[:, feature]):
                left_X, left_y, right_X, right_y = self.split_data_set(X, y, feature, value)

                new_MSE = self.mse_data(left_y)[0] + self.mse_data(right_y)[0]
                if best_MSE > new_MSE:
                    best_feature = feature
                    best_value = value
                    best_MSE = new_MSE
        
        if (MSE - best_MSE) < self.min_impurity_split:
            return None, self.avg_data(y)
        
        return best_feature, best_value