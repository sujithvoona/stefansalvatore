# Part 1 Preprocessing
# Importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
data_train = pd.read_csv('Pacific_train.csv')
data_test = pd.read_csv('Pacific_test.csv')
data_X_train = data_train[["Date","Time","Maximum Wind","Minimum Pressure","Low Wind NE","Low Wind SE","Low Wind SW","Low Wind NW","Moderate Wind NE","Moderate Wind SE","Moderate Wind SW","Moderate Wind NW","High Wind NE","High Wind SE","High Wind NW","High Wind SW"]]
data_y_train = data_train["Status"]
data_X_train = data_X_train[:10455]
data_y_train = data_y_train[:10455]
data_X_test = data_test[["Date","Time","Maximum Wind","Minimum Pressure","Low Wind NE","Low Wind SE","Low Wind SW","Low Wind NW","Moderate Wind NE","Moderate Wind SE","Moderate Wind SW","Moderate Wind NW","High Wind NE","High Wind SE","High Wind NW","High Wind SW"]]
data_y_test = data_test["Status"]


# Part 2 Classification by different classifiers
# Fitting the DecisionTreeClassifier onto the data
from sklearn.tree import DecisionTreeClassifier
classifier_1 = DecisionTreeClassifier(criterion = 'entropy')
classifier_1.fit(data_X_train, data_y_train)
z_1 = classifier_1.feature_importances_

# Predicting the test values by DecisionTreeClassifier
y_pred_1 = classifier_1.predict(data_X_test)
y_pred_1 = pd.Series(y_pred_1)

# Evaluating by confusion matrix for DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
cm_1 = confusion_matrix(data_y_test, y_pred_1)
from sklearn.metrics import accuracy_score
acc_1 = accuracy_score(data_y_test, y_pred_1)
from sklearn.metrics import recall_score
rs_1 = recall_score(data_y_test, y_pred_1, average = 'macro')
from sklearn.metrics import precision_score
pr_1 = precision_score(data_y_test, y_pred_1, average = 'macro')

# Applying k-fold cross validation for DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
accuracies_1 = cross_val_score(estimator = classifier_1, X = data_X_train, y = data_y_train, cv = 10, n_jobs = -1)
accuracies_1.mean()
accuracies_1.std()


# Fitting RandomForestClassifier onto the data
from sklearn.ensemble import RandomForestClassifier
classifier_2 = RandomForestClassifier(n_estimators = 1000, criterion = 'entropy')
classifier_2.fit(data_X_train, data_y_train)
z_2 = classifier_2.feature_importances_

# Predicting the test values by RandomForestClassifier
y_pred_2 = classifier_2.predict(data_X_test)
y_pred_2 = pd.Series(y_pred_2)

# Evaluating by confusion matrix for RandomForestClassifier
from sklearn.metrics import confusion_matrix
cm_2 = confusion_matrix(data_y_test, y_pred_2)
from sklearn.metrics import accuracy_score
acc_2 = accuracy_score(data_y_test, y_pred_2)
from sklearn.metrics import recall_score
rs_2 = recall_score(data_y_test, y_pred_2, average = 'macro')
from sklearn.metrics import precision_score
pr_2 = precision_score(data_y_test, y_pred_2, average = 'macro')

# Applying k-fold cross validation for RandomForestClassifier
from sklearn.model_selection import cross_val_score
accuracies_2 = cross_val_score(estimator = classifier_2, X = data_X_train, y = data_y_train, cv = 10, n_jobs = -1)
accuracies_2.mean()
accuracies_2.std()


# Fitting the NaiveBayesClassifier onto the data
from sklearn.naive_bayes import GaussianNB
classifier_3 = GaussianNB()
classifier_3.fit(data_X_train, data_y_train)
#z_3 = classifier_3.feature_importances_

# Predicting the test values by NaiveBayesClassifier
y_pred_3 = classifier_3.predict(data_X_test)
y_pred_3 = pd.Series(y_pred_3)

# Evaluating by confusion matrix for NaiveBayesClassifier
from sklearn.metrics import confusion_matrix
cm_3 = confusion_matrix(data_y_test, y_pred_3)
from sklearn.metrics import accuracy_score
acc_3 = accuracy_score(data_y_test, y_pred_3)
from sklearn.metrics import recall_score
rs_3 = recall_score(data_y_test, y_pred_3, average = 'macro')
from sklearn.metrics import precision_score
pr_3 = precision_score(data_y_test, y_pred_3, average = 'macro')

# Applying k-fold cross validation for NaiveBayesClassifier
from sklearn.model_selection import cross_val_score
accuracies_3 = cross_val_score(estimator = classifier_3, X = data_X_train, y = data_y_train, cv = 10, n_jobs = -1)
accuracies_3.mean()
accuracies_3.std()


# Fitting SVM Gaussian kernel onto the data
from sklearn.svm import SVC
classifier_4 = SVC(kernel = 'rbf', gamma = 'auto', cache_size = 1000)
classifier_4.fit(data_X_train, data_y_train)
#z_4 = classifier_4.feature_importances_

# Predicting the test values by SVM Gaussian kernel
y_pred_4 = classifier_4.predict(data_X_test)
y_pred_4 = pd.Series(y_pred_4)

# Evaluating by confusion matrix for SVM Gaussian kernel
from sklearn.metrics import confusion_matrix
cm_4 = confusion_matrix(data_y_test, y_pred_4)
from sklearn.metrics import accuracy_score
acc_4 = accuracy_score(data_y_test, y_pred_4)
from sklearn.metrics import recall_score
rs_4 = recall_score(data_y_test, y_pred_4, average = 'macro')
from sklearn.metrics import precision_score
pr_4 = precision_score(data_y_test, y_pred_4, average = 'macro')

# Applying k-fold cross validation for SVM Gaussian kernel
from sklearn.model_selection import cross_val_score
accuracies_4 = cross_val_score(estimator = classifier_4, X = data_X_train, y = data_y_train, cv = 10, n_jobs = -1)
accuracies_4.mean()
accuracies_4.std()