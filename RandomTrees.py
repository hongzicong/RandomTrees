# -*- coding: utf-8 -*-
"""
Created on Fri May 31 10:12:16 2019

@author: DELL-PC
"""

import numpy as np


class RandomTrees:
    
    def __init__(self, n_estimators, n_processes, max_features_num):
        self.n_estimators = n_estimators
        self.n_processes = n_processes
        self.max_features_num = max_features_num
    
    
    def fit(self, X, y):
        '''
        Parameters
        ----------
        X : matrix of shape = [n_samples, n_features]
            The matrix consisting of the input data
            
        Returns
        -------
        y : array of shape = [n_samples]
            The label of X
        '''
        pass
    
    
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
        
        
        
        y /= len(self.n_estimators)
        
        return y
    
    
    def mse_data(self, data):
        return np.var(data[:, -1]) * np.shape(data)[0]
    
    
    def avg_data(self, data):
        return np.mean(data[:,-1])
    
    
    def splitDataSet(self, dataSet, feature, value):
        return dataSet[np.nonzero(dataSet[:, feature] > value)[0], :], \
                dataSet[np.nonzero(dataSet[:, feature] < value)[0], :]
    
    
    def select_best_feature(self, data):
        '''
        Parameters
        ----------
        data : matrix of shape = [n_samples, n_features + 1]
            The matrix consisting of the train data and the respective label
        
        max_features : int
            The number of features to consider when looking for the best split
        '''
        n_features = data.shape[1]
        features_index = []
        
        best_MSE = np.inf
        best_feature = 0
        best_value = 0
        
        MSE = self.mse_data(data)

        for i in range(self.max_features_num):
            features_index.append(np.random.randint(n_features))
        
        for feature in features_index:
            for value in set(data[:, feature]):
                data_split0, data_split1 = self.splitDataSet(data, feature, value)
                
                new_MSE = self.MSE_data(data_split0) + self.MSE_data(data_split1)
                
                if best_MSE > new_MSE:
                    best_feature = feature
                    best_value = value
                    best_MSE = new_MSE
        
        if (MSE - best_MSE) < 0.001:
            return None, self.avg_data(data)
        
        return best_feature, best_value