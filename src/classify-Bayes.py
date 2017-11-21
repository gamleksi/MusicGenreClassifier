import numpy as np
from sklearn.decomposition import PCA
from sklearn.cross_validation import train_test_split
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn import preprocessing


# Import the data
test_data = np.genfromtxt('data/test_data.csv', delimiter=',')
train_data = np.genfromtxt('data/train_data.csv', delimiter=',')
train_labels = np.genfromtxt('data/train_labels.csv', delimiter=',')


# Standardize data
X = preprocessing.scale(train_data)
y = train_labels


best_score_naiveNB = 0.0
best_score_bernoulliNB = 0.0

for j in [2,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,40,80,150,250,270]:
    # Use PCA to reduce dimensionality
    print ('--------------------------')
    if j == 270:
        Xn = X
        print ('Using entire original data')

    else:
        pca = PCA(n_components=j)
        Xn = pca.fit_transform(X)
        print ('Variance explained by {} PCA dimensions: {}'.format(j ,sum(pca.explained_variance_ratio_)))

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(Xn, y, test_size=0.2, random_state=42)


    clf = GaussianNB()
    clf.fit(X_train, y_train)
    if clf.score(X_test,y_test) > best_score_naiveNB:
        best_score_naiveNB = clf.score(X_test,y_test)
    print ('Naive Bayes accuracy for test set: {}'.format(clf.score(X_test, y_test)))

    clf = BernoulliNB()
    clf.fit(X_train, y_train)
    if clf.score(X_test,y_test) > best_score_bernoulliNB:
        best_score_bernoulliNB = clf.score(X_test,y_test)
    print ('Bernoulli Bayes accuracy for test set: {}'.format(clf.score(X_test, y_test)))

print ("Best score for Naive Bayes: {}".format(best_score_naiveNB))
# print ("Best score for Multinomial Bayes: {}".format(best_score_multinomialNB))
print ("Best score for Bernoulli Bayes: {}".format(best_score_bernoulliNB))

# X_test = preprocessing.scale(test_data)
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