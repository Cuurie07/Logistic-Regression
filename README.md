# Logistic-Regression
The data is present in the mnist.mat file. The preprocess() function in the script has the preprocessing steps (apply feature selection and feature normalization) and divide data set into 3 parts: training set, validation set and testing set.


The input of blrObjFunction.m includes 3 parameters:
• X is a data matrix where each row contains a feature vector in original coordinate (not including the bias 1 at the beginning of vector). 
• wk is a column vector representing the parameters of Logistic Regression. Size of wk is (D + 1) * 1.
• yk is a column vector representing the labels of corresponding feature vectors in data matrix X. Each entry in this vector is either 1 or 0 to represent whether the feature vector belongs to a class Ck or not (k=0,1,···,K 1). Size of yk is N*1 where N is the number of rows of X. 


Function blrObjFunction() has 2 outputs:
• error is a scalar value.
• error grad is a column vector of size (D + 1) * 1 which represents the gradient of error function.


The input of blrPredict() includes 2 parameters:
• Similar to function blrObjFunction(), X is also a data matrix where each row contains a feature vector in original coordinate (not including the bias 1 at the beginning of vector). In other words, X has size N * D.
• W is a matrix where each column is a weight vector (wk) of classifier for digit k. Concretely, W has size (D + 1) * K where K = 10 is the number of classifiers.

The output of function blrPredict() is a column vector label which has size N * 1.


For multi-class Logistic Regression, the posterior probabilities are given by a softmax transformation of linear functions of the feature variables.

The details for the SVM package is given in http://scikit-learn. org/stable/modules/generated/sklearn.svm.SVC.html.

