# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 19:41:54 2019

@author: DELL-PC
"""

from RandomTrees import RandomTreesRegressor

import pandas as pd
import time as tm

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

time_begin = tm.time()

RANDOM_SEED = 106

data_train_1 = pd.read_csv("../dm2019springproj2/train1.csv", header=None)
#data_train_2 = pd.read_csv("../dm2019springproj2/train2.csv", header=None)
#data_train_3 = pd.read_csv("../dm2019springproj2/train3.csv", header=None)
#data_train_4 = pd.read_csv("../dm2019springproj2/train4.csv", header=None)
#data_train_5 = pd.read_csv("../dm2019springproj2/train5.csv", header=None)

data_test_1 = pd.read_csv("../dm2019springproj2/test1.csv", header=None)
#data_test_2 = pd.read_csv("../dm2019springproj2/test2.csv", header=None)
#data_test_3 = pd.read_csv("../dm2019springproj2/test3.csv", header=None)
#data_test_4 = pd.read_csv("../dm2019springproj2/test4.csv", header=None)
#data_test_5 = pd.read_csv("../dm2019springproj2/test5.csv", header=None)
#data_test_6 = pd.read_csv("../dm2019springproj2/test6.csv", header=None)

label_1 = pd.read_csv("../dm2019springproj2/label1.csv", header=None)
#label_2 = pd.read_csv("../dm2019springproj2/label2.csv", header=None)
#label_3 = pd.read_csv("../dm2019springproj2/label3.csv", header=None)
#label_4 = pd.read_csv("../dm2019springproj2/label4.csv", header=None)
#label_5 = pd.read_csv("../dm2019springproj2/label5.csv", header=None)

#x = pd.concat([data_train_1, data_train_2, data_train_3, data_train_4, data_train_5], ignore_index=True)
#y = pd.concat([label_1, label_2, label_3, label_4, label_5], ignore_index=True)

x = data_train_1[0:1000]
y = label_1[0:1000]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state = RANDOM_SEED)

model = RandomTreesRegressor(n_trees=2, n_processes=2, max_features_num=10, max_depth=5, min_samples_split=4)
model.fit(x_train, y_train)

y_pred = model.predict(x_test)
predictions = [round(value) for value in y_pred]

score = r2_score(y_test, predictions)
print("R2 Score: %.2f%%" % (score * 100.0))

print("Time used: %f" % (tm.time() - time_begin))

#data_test = pd.concat([data_test_1, data_test_2, data_test_3, data_test_4, data_test_5, data_test_6], ignore_index=True)
#result = model.predict(data_test)
#pd.DataFrame(result,columns=['Predicted'],index=list(range(1,len(result) + 1))).to_csv('../../dm2019springproj2/result_xgb.csv')