import numpy as np
from sklearn.decomposition import PCA
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn import svm
from sklearn import preprocessing


# Import the data
test_data = np.genfromtxt('data/test_data.csv', delimiter=',')
train_data = np.genfromtxt('data/train_data.csv', delimiter=',')
train_labels = np.genfromtxt('data/train_labels.csv', delimiter=',')


# Standardize data
X = preprocessing.scale(train_data)
y = train_labels


# real_best_score = 0.0
discrete_best_score = 0.0
# nu_real_best_score = 0.0
nu_discrete_best_score = 0.0

for j in [270]:
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


    # clf = AdaBoostClassifier(
    #         svm.SVC(decision_function_shape='ovo'),
    #         n_estimators = 600,
    #         learning_rate=1.5)
    # clf.fit(X_train, y_train)
    # if clf.score(X_test,y_test) > real_best_score:
    #     real_best_score = clf.score(X_test,y_test)
    # print ('SVM + real AdaBoost accuracy for test set: {}'.format(clf.score(X_test, y_test)))
    svc = svm.SVC(decision_function_shape='ovo')
    nusvc = svm.NuSVC(kernel='rbf', nu=0.01)
    svc.fit(X_train, y_train)
    nusvc.fit(X_train, y_train)
    print ("Best score for SVM + discrete AdaBoost: {}".format(svc.score(X_test, y_test)))
    print ("Best score for NuSVM + discrete AdaBoost: {}".format(nusvc.score(X_test, y_test)))

    clf = AdaBoostClassifier(
            base_estimator = svc,
            n_estimators = 600,
            learning_rate=1.5,
            algorithm='SAMME')
    clf.fit(X_train, y_train)
    if clf.score(X_test,y_test) > discrete_best_score:
        discrete_best_score = clf.score(X_test,y_test)
    print ('SVM + discrete AdaBoost accuracy for test set: {}'.format(clf.score(X_test, y_test)))

    # clf = AdaBoostClassifier(
    #         svm.NuSVC(kernel='rbf', nu=0.01),
    #         n_estimators = 600,
    #         learning_rate=1.5)
    # clf.fit(X_train, y_train)
    # if clf.score(X_test,y_test) > nu_real_best_score:
    #     nu_real_best_score = clf.score(X_test,y_test)
    # print ('NuSVM + real AdaBoost accuracy for test set: {}'.format(clf.score(X_test, y_test)))

    clf = AdaBoostClassifier(
            base_estimator = nusvc,
            n_estimators = 600,
            learning_rate=1.5,
            algorithm='SAMME')
    clf.fit(X_train, y_train)
    if clf.score(X_test,y_test) > nu_discrete_best_score:
        nu_discrete_best_score = clf.score(X_test,y_test)
    print ('NuSVM + discrete Adaboost accuracy for test set: {}'.format(clf.score(X_test, y_test)))


# print ("Best score for SVM + real AdaBoost: {}".format(real_best_score))
print ("Best score for SVM + discrete AdaBoost: {}".format(discrete_best_score))
# print ("Best score for NuSVM + real AdaBoost: {}".format(nu_real_best_score))
print ("Best score for NuSVM + discrete AdaBoost: {}".format(nu_discrete_best_score))
# print ("Predictions shape:", predictions.shape)

# f = open('test_labels.csv', 'w+')
# f.write('Sample_id,Sample_label\n')

# ind = np.arange(test_data.shape[0])

# for i in range(predictions.shape[0]):
#     row = "{},{}\n".format(ind[i]+1,int(predictions[i]))
#     f.write(row)

# f.close()