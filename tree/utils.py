"""
You can add your own functions here according to your decision tree implementation.
There is no restriction on following the below template, these fucntions are here to simply help you.
"""

import pandas as pd
import numpy as np

def one_hot_encoding(X: pd.DataFrame) -> pd.DataFrame:
    """
    Function to perform one hot encoding on the input data
    """

    return pd.get_dummies(X)

def check_ifreal(y: pd.Series) -> bool:
    """
    Function to check if the given series has real or discrete values
    """

    if pd.api.types.is_numeric_dtype(y):
        unique = len(y.unique())
        total = y.count()
        if (total/unique) <= 10:
            return True
    return False


def entropy(Y: pd.Series) -> float:
    """
    Function to calculate the entropy
    """

    probs = Y.value_counts(normalize = True)
    return -1 * np.sum(probs * np.log2(probs + 1e-9))


def gini_index(Y: pd.Series) -> float:
    """
    Function to calculate the gini index
    """
    
    probs = Y.value_counts(normalize = True)
    return 1 - np.sum(probs ** 2)

def mse(Y: pd.Series) -> float:
    """
    Function to claculate mean square error
    """

    sq_err = (Y - Y.mean())**2
    return sq_err.mean()

def information_gain(Y: pd.Series, attr: pd.Series, criterion: str) -> float:
    """
    Function to calculate the information gain using criterion (entropy, gini index or MSE)
    """

    func = entropy # criterion is taken as entropy by default

    if criterion == "mse": func = mse
    elif criterion == "gini_index": func = gini_index

    gain = func(Y)
    for attribute in attr.unique():
        si = []
        for i in range(len(attr)):
            if attr.iloc[i] == attribute:
                si.append(Y.iloc[i])
        gain -= (len(si) / len(Y)) * func(pd.Series(si))

    return gain

# real output
def opt_split_real(X: pd.Series, y: pd.Series, is_real: bool, criterion: str) -> float:
    
    opt_split_val, opt_gain = None, -float('inf')

    if is_real: #real input
        sorted_indices = np.argsort(X)
        sorted_X = X.iloc[sorted_indices]
        sorted_y = y.iloc[sorted_indices]

        for i in range(len(sorted_X)):
            split_val = (sorted_X.iloc[i-1] + sorted_X.iloc[i]) / 2
            left = sorted_X <= split_val
            gain = information_gain(sorted_y, left, 'mse')

            if gain > opt_gain: opt_split_val, opt_gain = split_val, gain
    else: # discrete input
        opt_gain = information_gain(y, X, criterion)

    return opt_split_val, opt_gain

# discrete output
def opt_split_discrete(X: pd.Series, y: pd.Series, is_real: bool, criterion: str = 'entropy') -> float:
    
    opt_split_val, opt_gain = None, -float('inf')

    if is_real: # real input
        sorted_indices = np.argsort(X)
        sorted_X = X.iloc[sorted_indices]
        sorted_y = y.iloc[sorted_indices]

        for i in range(1, len(sorted_X)):
            split_val = (sorted_X.iloc[i-1] + sorted_X.iloc[i]) / 2
            left = sorted_X <= split_val
            gain = information_gain(sorted_y, left, criterion)

            if gain > opt_gain: opt_split_val, opt_gain = split_val, gain
    else: # dsicrete input
        opt_gain = information_gain(y, X, criterion)
    return opt_split_val, opt_gain



def opt_split_attribute(X: pd.DataFrame, y: pd.Series, criterion: str, features: pd.Series, is_real: bool, is_real_feature: list):
    """
    Function to find the optimal attribute to split about.
    If needed you can split this function into 2, one for discrete and one for real valued features.
    You can also change the parameters of this function according to your implementation.

    features: pd.Series is a list of all the attributes we have to split upon

    return: attribute to split upon
    """

    opt_feature = None
    opt_val = None
    opt_gain = -float('inf')

    func = opt_split_discrete
    if is_real: func = opt_split_real

    c = 0
    for i in features:
        split_val, gain = func(X.loc[:,i], y, is_real_feature[c], criterion)
        c += 1

        if gain > opt_gain: opt_feature, opt_val, opt_gain = i, split_val, gain

    return opt_feature, opt_val

def split_data(X: pd.DataFrame, y: pd.Series, attribute: str, value):
    
    if value is None:
        splits = []
        unique_vals = X[attribute].unique()

        for val in unique_vals:
            X_split = X[X[attribute] == val]
            y_split = y[X[attribute] == val]
            splits.append((X_split, y_split))
    else:
        left = X[attribute] <= value

        X_left, y_left = X[left], y[left]
        X_right, y_right = X[~left], y[~left]

        splits = [(X_left, y_left), (X_right, y_right)]
    return splits