import numpy as np
from sklearn.decomposition import PCA
from sklearn.cross_validation import train_test_split
from sklearn import preprocessing
from keras.models import Sequential
from keras.layers import Dense


# Import the data
test_data = np.genfromtxt('data/test_data.csv', delimiter=',')
train_data = np.genfromtxt('data/train_data.csv', delimiter=',')
train_labels = np.genfromtxt('data/train_labels.csv', delimiter=',')


# Standardize data
scaler = preprocessing.StandardScaler()
scaler.fit(train_data)
X = scaler.transform(train_data)
y = train_labels

model = Sequential()

model.add(Dense(units=64, activation='relu' input_dim=100))
model.add(Dense(units=10, activation='softmax'))

model.compiler(loss='categorial_crossentropy', optimizer='sgd', metrics=['accuracy'])

model.fit(X, y, epocs=5, batch_size=32)

loss_and_metrics = model.evaluate(x_test, y_test, batch_size=128)

# best_score = 0.0

# for j in [10,20,40,80,150,270]:
#     # Use PCA to reduce dimensionality
#     print ('--------------------------')
#     if j == 270:
#         Xn = X
#         print ('Using entire original data')

#     else:
#         pca = PCA(n_components=j)
#         Xn = pca.fit_transform(X)
#         print ('Variance explained by {} PCA dimensions: {}'.format(j ,sum(pca.explained_variance_ratio_)))

#     # Split data into train and test sets
#     X_train, X_test, y_train, y_test = train_test_split(Xn, y, test_size=0.2, random_state=42)

#     for i in [3,4,5,6,7,10,20,40]:
#         mlp = MLPClassifier(hidden_layer_sizes=(i), max_iter=500)
#         mlp.fit(X_train, y_train)
#         if mlp.score(X_test,y_test) > best_score:
#             best_score= mlp.score(X_test,y_test)
#         print ('MLP accuracy for test set (hidden layer size: {}): {}'.format(i, mlp.score(X_test, y_test)))
#         mlp = MLPClassifier(hidden_layer_sizes=(i,i), max_iter=500)
#         mlp.fit(X_train, y_train)
#         if mlp.score(X_test,y_test) > best_score:
#             best_score= mlp.score(X_test,y_test)
#         print ('MLP accuracy for test set (hidden layer size: ({},{})): {}'.format(i,i, mlp.score(X_test, y_test)))
#         mlp = MLPClassifier(hidden_layer_sizes=(i,i,i), max_iter=500)
#         mlp.fit(X_train, y_train)
#         if mlp.score(X_test,y_test) > best_score:
#             best_score= mlp.score(X_test,y_test)
#         print ('MLP accuracy for test set (hidden layer size: ({},{},{})): {}'.format(i,i,i, mlp.score(X_test, y_test)))


# print ("Best score for MLP: {}".format(best_score))

# X_test = scaler.transform(test_data)
# clf = svm.NuSVC(kernel='rbf', nu=0.01)
# clf.fit(X, y)
# predictions = clf.predict(X_test)
# print ("Predictions shape:", predictions.shape)

# f = open('test_labels.csv', 'w+')
# f.write('Sample_id,Sample_label\n')

# ind = np.arange(test_data.shape[0])

# for i in range(predictions.shape[0]):
#     row = "{},{}\n".format(ind[i]+1,int(predictions[i]))
#     f.write(row)

# f.close()