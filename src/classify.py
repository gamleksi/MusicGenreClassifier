#!/usr/bin/python

import numpy as np
# from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
# import matplotlib.pyplot as plt


##
# Import and investigate the data
##
test_data = np.genfromtxt('data/test_data.csv', delimiter=',')
train_data = np.genfromtxt('data/train_data.csv', delimiter=',')
train_labels = np.genfromtxt('data/train_labels.csv', delimiter=',')





##
# Prediction
##

scaler = preprocessing.StandardScaler().fit(train_data)

# test_data_scaled = scaler.transform(test_data)
# train_data_scaled = scaler.transform(train_data)

# clf = svm.LinearSVC()
clf = RandomForestClassifier(max_depth=100, random_state=0)
# clf.fit(train_data_scaled, train_labels)
clf.fit(train_data, train_labels)
predictions = clf.predict(test_data)
print ("Predictions shape:", predictions.shape)

f = open('test_labels.csv', 'w+')
f.write('Sample_id,Sample_label\n')

ind = np.arange(test_data.shape[0])

for i in range(predictions.shape[0]):
    row = "{},{}\n".format(ind[i]+1,int(predictions[i]))
    f.write(row)

f.close()