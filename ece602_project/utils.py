import numpy as np
from sklearn.metrics import confusion_matrix

def mean(X, weights = None):
    '''
    Computes the mean of a data set
    
    Parameters:
    X(np.array): rows are samples, columns are pamameters
    weights(np.array): 1D array representing the observation rates (high number represents observe more times)
    '''
    return np.average(X, weights=weights, axis=0)

def cov(X, weights = None):
    '''
    Computes the unbiased covariance matrix of a dataset
    
    Parameters:
    X(np.array): rows are samples, columns are pamameters
    weights(np.array): 1D array representing the observation rates (high number represents observe more times)
    '''
    return np.cov(X.T, aweights=weights)

def p_n_split(X,y, weights=None):
    '''
    Split the data set into positive and negative groups
    
    Parameters:
    X(np.array): rows are samples, columns are pamameters
    y(np.array): labels for the samples
    weights(np.array): observation rates for each sample points
    '''

    X_p = []
    X_n = []
    w_p = []
    w_n = []
    y_p = np.ones(sum(y==1))
    y_n = -np.ones(sum(y==-1))
    
    if weights is None:
        weights = np.ones(len(X))/len(X)
        
    if len(weights) != len(X):
        raise ValueError('input data X and weights dimension does not match')

    for i in range(len(y)):
        if y[i] == 1:
            X_p.append(X[i,:])
            w_p.append(weights[i])
        elif y[i] == -1:
            X_n.append(X[i,:])
            w_n.append(weights[i])
        
    return np.array(X_p), y_p, np.array(w_p), np.array(X_n), y_n, np.array(w_n)

def evaluate(y_test, y_pred, verbose=True):
    '''
    evaluate the performance of a linear classifier in terms of confusion matrix
    
    Parameters:
    y_test(np.array): a 1D array with true labels
    y_pred(np.array): a 1D array with predicted labels
    '''
    results = confusion_matrix(y_test, y_pred).flatten()
    
    results = np.array([results,results/[sum(y_test==-1),sum(y_test==-1),sum(y_test==1),sum(y_test==1)]])
    if verbose == True:
        print('tn: {:.0f} cases which is {:.2%}'.format(results[0,0], results[1,0]))
        print('fp: {:.0f} cases which is {:.2%}'.format(results[0,1], results[1,1]))
        print('fn: {:.0f} cases which is {:.2%}'.format(results[0,2], results[1,2]))
        print('tp: {:.0f} cases which is {:.2%}'.format(results[0,3], results[1,3]))
    
    return results
    
def normalize_weights(weights):
    '''
    normalize weight vectors so that the sum of all elements is equal to 1
    
    parameters:
    weights(np.array): vector of sample weights
    '''
    return weights/sum(weights)
