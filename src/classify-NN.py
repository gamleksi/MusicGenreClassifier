import numpy as np
from sklearn.decomposition import PCA
from sklearn.cross_validation import train_test_split
from sklearn import neighbors, linear_model
from sklearn import preprocessing

# Import and investigate the data
test_data = np.genfromtxt('data/test_data.csv', delimiter=',')
train_data = np.genfromtxt('data/train_data.csv', delimiter=',')
train_labels = np.genfromtxt('data/train_labels.csv', delimiter=',')


# Standardize data
X = preprocessing.scale(train_data)
# X = train_data
y = train_labels


best_score = 0.0

for j in [2,5,10,20,40,80,150,250,270]:
    # Use PCA to reduce dimensionality
    if j == 270:
        Xn = X
    else:
        pca = PCA(n_components=j)
        Xn = pca.fit_transform(X)
        print ('Variance explained by {} PCA dimensions: %f'.format(j ,sum(pca.explained_variance_ratio_)))

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(Xn, y, test_size=0.2, random_state=42)


    for i in range(5,35,2):
        knn = neighbors.KNeighborsClassifier(n_neighbors = i)
        knn_model_1 = knn.fit(X_train, y_train)
        if knn_model_1.score(X_test,y_test) > best_score:
            best_score = knn_model_1.score(X_test,y_test)
        print ('{}-NN accuracy for test set: {}'.format(i, knn_model_1.score(X_test, y_test)))


print ("Best score: {}".format(best_score))
# print ("Predictions shape:", predictions.shape)

# f = open('test_labels.csv', 'w+')
# f.write('Sample_id,Sample_label\n')

# ind = np.arange(test_data.shape[0])

# for i in range(predictions.shape[0]):
#     row = "{},{}\n".format(ind[i]+1,int(predictions[i]))
#     f.write(row)

# f.close()