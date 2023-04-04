"""EECS 445 - Winter 2022.

Project 1
"""

# from msilib.schema import Binary
import pandas as pd
import numpy as np
import itertools
import string

from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics
from matplotlib import pyplot as plt

from helper import *

import warnings
from sklearn.exceptions import ConvergenceWarning

warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=ConvergenceWarning)

np.random.seed(445)

def extract_word(input_string):
    """Preprocess review into list of tokens.

    Convert input string to lowercase, replace punctuation with spaces, and split along whitespace.
    Return the resulting array.

    E.g.
    > extract_word("I love EECS 445. It's my favorite course!")
    > ["i", "love", "eecs", "445", "it", "s", "my", "favorite", "course"]

    Input:
        input_string: text for a single review
    Returns:
        a list of words, extracted and preprocessed according to the directions
        above.
    """
    #convert to lowercase
    input = input_string.lower()

    #replace punctuation with space
    for i in input:
        if i in string.punctuation:
            input = input.replace(i, " ")

    return input.split()

def extract_dictionary(df):
    """Map words to index.

    Reads a pandas dataframe, and returns a dictionary of distinct words
    mapping from each distinct word to its index (ordered by when it was
    found).

    E.g., with input:
        | text                          | label | ... |
        | It was the best of times.     |  1    | ... |
        | It was the blurst of times.   | -1    | ... |

    The output should be a dictionary of indices ordered by first occurence in
    the entire dataset:
        {
           it: 0,
           was: 1,
           the: 2,
           best: 3,
           of: 4,
           times: 5,
           blurst: 6
        }
    The index should be autoincrementing, starting at 0.

    Input:
        df: dataframe/output of load_data()
    Returns:
        a dictionary mapping words to an index
    """
    word_dict = {}
    index = 0

    for i in df['text']:
        for j in extract_word(i):
            if (j not in word_dict):
                word_dict[j] = index
                index = index + 1

    return word_dict

def generate_feature_matrix(df, word_dict):
    """Create matrix of feature vectors for dataset.

    Reads a dataframe and the dictionary of unique words to generate a matrix
    of {1, 0} feature vectors for each review.  Use the word_dict to find the
    correct index to set to 1 for each place in the feature vector. The
    resulting feature matrix should be of dimension (# of reviews, # of words
    in dictionary).

    Input:
        df: dataframe that has the text and labels
        word_dict: dictionary of words mapping to indices
    Returns:
        a numpy matrix of dimension (# of reviews, # of words in dictionary)
    """
    number_of_reviews = df.shape[0]
    number_of_words = len(word_dict)
    feature_matrix = np.zeros((number_of_reviews, number_of_words))
    
    index = 0
    for i in df['text']:
        for j in extract_word(i):
            if (j in word_dict):
                feature_matrix[index][word_dict[j]] = 1
        index += 1


    return feature_matrix

def performance(y_true, y_pred, metric="accuracy"):
    """Calculate performance metrics.

    Performance metrics are evaluated on the true labels y_true versus the
    predicted labels y_pred.

    Input:
        y_true: (n,) array containing known labels
        y_pred: (n,) array containing predicted scores
        metric: string specifying the performance metric (default='accuracy'
                 other options: 'f1-score', 'auroc', 'precision', 'sensitivity',
                 and 'specificity')
    Returns:
        the performance as an np.float64
    """
    # TODO: Implement this function
    # This is an optional but very useful function to implement.
    # See the sklearn.metrics documentation for pointers on how to implement
    # the requested metrics.
    if (metric == "accuracy"):
        return metrics.accuracy_score(y_true, y_pred)
    elif (metric == "f1-score"):
        return metrics.f1_score(y_true, y_pred)
    elif (metric == "auroc"):
        return metrics.roc_auc_score(y_true, y_pred)
    elif (metric == "precision"):
        return metrics.precision_score(y_true, y_pred)
    else:
        tn, fp, fn, tp = metrics.confusion_matrix(y_true, y_pred).ravel()
        if (metric == "sensitivity"):
            return (tp / (tp+fn))
        else:
            return (tn / (tn+fp))

def cv_performance(clf, X, y, k=5, metric="accuracy"):
    """Split data into k folds and run cross-validation.

    Splits the data X and the labels y into k-folds and runs k-fold
    cross-validation: for each fold i in 1...k, trains a classifier on
    all the data except the ith fold, and tests on the ith fold.
    Calculates and returns the k-fold cross-validation performance metric for
    classifier clf by averaging the performance across folds.
    Input:
        clf: an instance of SVC()
        X: (n,d) array of feature vectors, where n is the number of examples
           and d is the number of features
        y: (n,) array of binary labels {1,-1}
        k: an int specifying the number of folds (default=5)
        metric: string specifying the performance metric (default='accuracy'
             other options: 'f1-score', 'auroc', 'precision', 'sensitivity',
             and 'specificity')
    Returns:
        average 'test' performance across the k folds as np.float64
    """
    # TODO: Implement this function
    # HINT: You may find the StratifiedKFold from sklearn.model_selection
    # to be useful

    scores = []

    skf = StratifiedKFold(n_splits=k, shuffle=False)
    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        clf.fit(X_train, y_train)
        if (metric == "auroc"):
            y_pred = clf.decision_function(X_test)
        else:
            y_pred = clf.predict(X_test)
        # Put the performance of the model on each fold in the scores array
        scores.append(performance(y_test, y_pred, metric))
    return np.array(scores).mean()

def select_param_linear(
    X, y, k=5, metric="accuracy", C_range=[], loss="hinge", penalty="l2", dual=True
):
    """Search for hyperparameters of linear SVM with best k-fold CV performance.

    Sweeps different settings for the hyperparameter of a linear-kernel SVM,
    calculating the k-fold CV performance for each setting on X, y.
    Input:
        X: (n,d) array of feature vectors, where n is the number of examples
        and d is the number of features
        y: (n,) array of binary labels {1,-1}
        k: int specifying the number of folds (default=5)
        metric: string specifying the performance metric (default='accuracy',
             other options: 'f1-score', 'auroc', 'precision', 'sensitivity',
             and 'specificity')
        C_range: an array with C values to be searched over
        loss: string specifying the loss function used (default="hinge",
             other option of "squared_hinge")
        penalty: string specifying the penalty type used (default="l2",
             other option of "l1")
        dual: boolean specifying whether to use the dual formulation of the
             linear SVM (set True for penalty "l2" and False for penalty "l1"ß)
    Returns:
        the parameter value for a linear-kernel SVM that maximizes the
        average 5-fold CV performance.
    """
    # TODO: Implement this function
    # HINT: You should be using your cv_performance function here
    # to evaluate the performance of each SVM
    maxPerformance = 0
    index = 0
    for i in C_range:
        clf = LinearSVC(C = i, random_state = 445, loss='hinge', penalty='l2')
        temp = cv_performance(clf, X, y, k, metric)
        if (maxPerformance < temp):
            index = i
            maxPerformance = temp
    return index

def plot_weight(X, y, penalty, C_range, loss, dual):
    """Create a plot of the L0 norm learned by a classifier for each C in C_range.

    Input:
        X: (n,d) array of feature vectors, where n is the number of examples
        and d is the number of features
        y: (n,) array of binary labels {1,-1}
        penalty: penalty to be forwarded to the LinearSVC constructor
        C_range: list of C values to train a classifier on
        loss: loss function to be forwarded to the LinearSVC constructor
        dual: whether to solve the dual or primal optimization problem, to be
            forwarded to the LinearSVC constructor
    Returns: None
        Saves a plot of the L0 norms to the filesystem.
    """
    norm0 = []
    # TODO: Implement this part of the function
    # Here, for each value of c in C_range, you should
    # append to norm0 the L0-norm of the theta vector that is learned
    # when fitting an L2- or L1-penalty, degree=1 SVM to the data (X, y)
    for c in C_range:
        clf = LinearSVC(C = c, random_state = 445, loss=loss, penalty=penalty, dual = dual)
        clf.fit(X, y)
        norm0.append(np.linalg.norm(clf.coef_[0], ord = 0, axis = 0))

    plt.plot(C_range, norm0)
    plt.xscale("log")
    plt.legend(["L0-norm"])
    plt.xlabel("Value of C")
    plt.ylabel("Norm of theta")
    plt.title("Norm-" + penalty + "_penalty.png")
    plt.savefig("Norm-" + penalty + "_penalty.png")
    plt.close()

def select_param_quadratic(X, y, k=5, metric="accuracy", param_range=[]):
    """Search for hyperparameters of quadratic SVM with best k-fold CV performance.
 
    Sweeps different settings for the hyperparameters of an quadratic-kernel SVM,
    calculating the k-fold CV performance for each setting on X, y.
    Input:
        X: (n,d) array of feature vectors, where n is the number of examples
           and d is the number of features
        y: (n,) array of binary labels {1,-1}
        k: an int specifying the number of folds (default=5)
        metric: string specifying the performance metric (default='accuracy'
                 other options: 'f1-score', 'auroc', 'precision', 'sensitivity',
                 and 'specificity')
        param_range: a (num_param, 2)-sized array containing the
            parameter values to search over. The first column should
            represent the values for C, and the second column should
            represent the values for r. Each row of this array thus
            represents a pair of parameters to be tried together.
    Returns:
        The parameter values for a quadratic-kernel SVM that maximize
        the average 5-fold CV performance as a pair (C,r)
    """
    # TODO: Implement this function
    # Hint: This will be very similar to select_param_linear, except
    # the type of SVM model you are using will be different...
    best_C_val, best_r_val = 0.0, 0.0
    maxPerformance = 0
    for c, r in param_range:
        clf = SVC(kernel='poly', random_state = 445, degree=2, C=c, coef0=r, gamma='auto')
        temp = cv_performance(clf, X, y, k, metric)
        if (maxPerformance < temp):
            best_C_val = c
            best_r_val = r
            maxPerformance = temp

    return best_C_val, best_r_val

def main():
    # Read binary data
    # NOTE: READING IN THE DATA WILL NOT WORK UNTIL YOU HAVE FINISHED
    #       IMPLEMENTING generate_feature_matrix AND extract_dictionary
    X_train, Y_train, X_test, Y_test, dictionary_binary = get_split_binary_data(
        fname="data/dataset.csv"
    )
    IMB_features, IMB_labels, IMB_test_features, IMB_test_labels = get_imbalanced_data(
        dictionary_binary, fname="data/dataset.csv"
    )

    # TODO: Questions 3, 4, 5

    # 3.a
    # print(extract_word("'BEST book ever! It\'s great'"))

    # # 3.b
    # print(X_train.shape)

    # # 3.c.i
    # count_list = []
    # for i in range(X_train.shape[0]):
    #     count = 0
    #     for j in range(X_train.shape[1]):
    #         if X_train[i][j] != 0:
    #             count += 1
    #     count_list.append(count)
    # print(sum(count_list) / len(count_list))

    # # # 3.c.ii
    # word_list = np.zeros((1, X_train.shape[1]))
    # for i in range(X_train.shape[0]):
    #     for j in range(X_train.shape[1]):
    #         word_list[0][j] += X_train[i][j]

    # for key, val in dictionary_binary.items():
    #     if val == np.argmax(word_list):
    #         print(key)    

    # # 4.1.b CORRECT
    # C_range = [ 10**(-3), 10**(-2), 10**(-1), 10**(0), 10**(1), 10**(2), 10**(3)]
    # c = select_param_linear(X_train, Y_train, 5, "accuracy", C_range)
    # clf = LinearSVC(C = c, random_state = 445, loss='hinge', penalty='l2')
    # performance = cv_performance(clf, X_train, Y_train, k=5, metric="accuracy")
    # print("\nMetric: accuracy", "\nBest c:", c, "\nCV Score:", performance)

    # c = select_param_linear(X_train, Y_train, 5, "f1-score", C_range)
    # performance = cv_performance(clf, X_train, Y_train, k=5, metric="f1-score")
    # print("\nMetric: f1-score", "\nBest c:", c, "\nCV Score:", performance)

    # c = select_param_linear(X_train, Y_train, 5, "auroc", C_range)
    # performance = cv_performance(clf, X_train, Y_train, k=5, metric="auroc")
    # print("\nMetric: auroc", "\nBest c:", c, "\nCV Score:", performance)

    # c = select_param_linear(X_train, Y_train, 5, "precision", C_range)
    # performance = cv_performance(clf, X_train, Y_train, k=5, metric="precision")
    # print("\nMetric: precision", "\nBest c:", c, "\nCV Score:", performance)

    # c = select_param_linear(X_train, Y_train, 5, "sensitivity", C_range)
    # performance = cv_performance(clf, X_train, Y_train, k=5, metric="sensitivity")
    # print("\nMetric: sensitivity", "\nBest c:", c, "\nCV Score:", performance)

    # c = select_param_linear(X_train, Y_train, 5, "specificity", C_range)
    # performance = cv_performance(clf, X_train, Y_train, k=5, metric="specificity")
    # print("\nMetric: specificity", "\nBest c:", c, "\nCV Score:", performance)

    # # 4.1.c performance
    # clf = LinearSVC(C = 1, random_state = 445, loss='hinge', penalty='l2')
    # clf.fit(X_train, Y_train)
    # y_pred = clf.predict(X_test)
    # print("Accuracy:", performance(Y_test, y_pred, "accuracy"))
    # print("F1-score:", performance(Y_test, y_pred, "f1-score"))
    # print("Auroc:", performance(Y_test, clf.decision_function(X_test), "auroc"))
    # print("Precision:", performance(Y_test, y_pred, "precision"))
    # print("Sensitivity:", performance(Y_test, y_pred, "sensitivity"))
    # print("Specificity:", performance(Y_test, y_pred, "specificity"))

    # # 4.1.d CORRECT
    # plot_weight(X_train, Y_train, "l2", C_range, "hinge", True)

    # # 4.1.e CORRECT
    # clf = LinearSVC(C = 0.1, random_state = 445, loss='hinge', penalty='l2')
    # clf.fit(X_train, Y_train)

    # max_ind = clf.coef_[0].argsort()[-5:]
    # for key, val in dictionary_binary.items():
    #     if val in max_ind:
    #         print(key, clf.coef_[0][val])

    # min_ind = clf.coef_[0].argsort()[:5]
    # for key, val in dictionary_binary.items():
    #     if val in min_ind:
    #         print(key, clf.coef_[0][val])

    # # 4.2.a CORRECT
    # C_range = [ 10**(-3), 10**(-2), 10**(-1), 10**(0)]
    # c = select_param_linear(X_train, Y_train, 5, "auroc", C_range, loss = 'squared_hinge', penalty = 'l1', dual = False)
    # clf = LinearSVC(C = c, random_state = 445, penalty = 'l1', loss = 'squared_hinge', dual = False)
    # mean_score = cv_performance(clf, X_train, Y_train, k=5, metric="accuracy")
    # best_score = cv_performance(clf, X_test, Y_test, k=5, metric="accuracy")
    # print("\nC value:", c, "\nmean CV AUROC score:", mean_score, "\nAUROC score on test set:", best_score)

    # plot_weight(X_train, Y_train, 'l1', C_range, 'squared_hinge', False)

    # # 4.3.a CORRECT
    # c_range = [10**(-2), 10**(-1), 10**(0), 10**(1), 10**(2), 10**(3)]
    # r_range = [10**(-2), 10**(-1), 10**(0), 10**(1), 10**(2), 10**(3)]
    # param = []
    # for i in c_range:
    #     for j in r_range:
    #         param.append([i, j])

    # c, r = select_param_quadratic(X_train, Y_train, 5, "auroc", param)
    # clf = SVC(kernel='poly', random_state = 445, degree=2, C=c, coef0=r, gamma='auto')
    # performance = cv_performance(clf, X_test, Y_test, k=5, metric="auroc")
    # print("\nQuadratic SVM with grid search and auroc metric:", "\nBest c:", c, "\nBest coeff:", r, "\nTest performance:", performance)

    # # 4.3.b CORRECT
    # c_random = np.random.uniform(-2,3,25)
    # c_range = 10**c_random
    # r_random = np.random.uniform(-2,3,25)
    # r_range = 10**r_random
    # param = np.vstack((c_range, r_range)).T

    # c, r = select_param_quadratic(X_train, Y_train, 5, "auroc", param)
    # clf = SVC(kernel='poly', random_state = 445, degree=2, C=c, coef0=r, gamma='auto')
    # performance = cv_performance(clf, X_test, Y_test, k=5, metric="auroc")
    # print("\nQuadratic SVM with random search and auroc metric:", "\nBest c:", c, "\nBest coeff:", r, "\nTest performance:", performance)

    # # 5.1.c CORRECT
    # clf = LinearSVC(C = 0.01, random_state = 445, loss = 'hinge', penalty = 'l2', class_weight = {-1: 1, 1: 10})
    # clf.fit(X_train, Y_train)
    # y_pred = clf.predict(X_test)
    # print("Accuracy:", performance(Y_test, y_pred, "accuracy"))
    # print("F1-score:", performance(Y_test, y_pred, "f1-score"))
    # print("Auroc:", performance(Y_test, clf.decision_function(X_test), "auroc"))
    # print("Precision:", performance(Y_test, y_pred, "precision"))
    # print("Sensitivity:", performance(Y_test, y_pred, "sensitivity"))
    # print("Specificity:", performance(Y_test, y_pred, "specificity"))

    # # 5.2.a CORRECT
    # clf = LinearSVC(C = 0.01, random_state = 445, loss = 'hinge', penalty = 'l2', class_weight = {-1: 1, 1: 1})
    # clf.fit(IMB_features, IMB_labels)
    # y_pred = clf.predict(IMB_test_features)
    # print("Accuracy:", performance(IMB_test_labels, y_pred, "accuracy"))
    # print("F1-score:", performance(IMB_test_labels, y_pred, "f1-score"))
    # print("Auroc:", performance(IMB_test_labels, clf.decision_function(IMB_test_features), "auroc"))
    # print("Precision:", performance(IMB_test_labels, y_pred, "precision"))
    # print("Sensitivity:", performance(IMB_test_labels, y_pred, "sensitivity"))
    # print("Specificity:", performance(IMB_test_labels, y_pred, "specificity"))

    # # 5.3.a 
    # clf = LinearSVC(C = 0.01, random_state = 445, loss = 'hinge', penalty = 'l2', class_weight = {-1: 3, 1: 8})
    # clf.fit(IMB_features, IMB_labels)
    # y_pred = clf.predict(IMB_test_features)
    # print("Class_weight={-1: 3, 1: 8}")
    # print("Test Performance on metric on accuracy:", performance(IMB_test_labels, y_pred, "accuracy"))
    # print("Test Performance on metric on f1-score:", performance(IMB_test_labels, y_pred, "f1-score"))
    # print("Test Performance on metric on auroc:", performance(IMB_test_labels, clf.decision_function(IMB_test_features), "auroc"))
    # print("Test Performance on metric on precision:", performance(IMB_test_labels, y_pred, "precision"))
    # print("Test Performance on metric on sensitivity:", performance(IMB_test_labels, y_pred, "sensitivity"))
    # print("Test Performance on metric on specificity:", performance(IMB_test_labels, y_pred, "specificity"))

    # # 5.4
    # clf_11 = LinearSVC(C = 0.01, random_state = 445, loss = 'hinge', penalty = 'l2', class_weight = {-1: 1, 1: 1})
    # clf_11.fit(IMB_features, IMB_labels)
    
    # clf_38 = LinearSVC(C = 0.01, random_state = 445, loss = 'hinge', penalty = 'l2', class_weight = {-1: 3, 1: 8})
    # clf_38.fit(IMB_features, IMB_labels)

    # fig = metrics.plot_roc_curve( clf_11, IMB_test_features, IMB_test_labels, label = "Wn = 1, Wp = 1")
    # fig = metrics.plot_roc_curve( clf_38, IMB_test_features, IMB_test_labels, ax = fig.ax_, label = "Wn = 3, Wp = 8")

    # plt.title('5.4 The ROC Curve')
    # plt.show()

    # Read multiclass data
    # TODO: Question 6: Apply a classifier to heldout features, and then use
    #       generate_challenge_labels to print the predicted labels
    
    (multiclass_features,
    multiclass_labels,
    multiclass_dictionary) = get_multiclass_training_data()

    heldout_features = get_heldout_reviews(multiclass_dictionary)
        
    C_range = [10**(-3), 10**(-2), 10**(-1), 10**(0), 10**(1), 10**(2), 10**(3)]
    c = select_param_linear(multiclass_features, multiclass_labels, 9, "accuracy", C_range)
    clf = LinearSVC(C = c, random_state = 445, multi_class = 'ovr')
    clf.fit(multiclass_features, multiclass_labels)   
    y_pred = clf.predict(heldout_features) 
    generate_challenge_labels(y_pred, "stae")
    print(cv_performance(clf, multiclass_features, multiclass_labels, k=9, metric="accuracy"))

if __name__ == "__main__":
    main()
