"""
The current code given is for the Assignment 1.
You will be expected to use this to make trees for:
> discrete input, discrete output
> real input, real output
> real input, discrete output
> discrete input, real output
"""
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tree.utils import *

np.random.seed(42)

class Node:
    def __init__(self, feature = None, threshold = None, output = None):
        self.feature = feature
        self.threshold = threshold
        self.children = []
        self.output = output

def create_Tree(X: pd.DataFrame, y: pd.Series, depth: int, max_depth: int, is_real: bool, is_real_feature: list, criterion: str):

    if depth >= max_depth or len(np.unique(y)) == 1:
        leaf_val = None
        if len(y): leaf_val = y.mode()[0] if not is_real else y.mean()
        return Node(output = leaf_val)
    
    features = X.columns
    opt_feature, opt_val = opt_split_attribute(X, y, criterion, features, is_real, is_real_feature)

    if opt_feature is None:
        leaf_val = None
        if len(y): leaf_val = y.mode()[0] if not is_real else y.mean()
        return Node(output = leaf_val)

    branch = Node(feature = opt_feature, threshold = (X[opt_feature].unique() if opt_val is None else opt_val))

    splits = split_data(X, y, opt_feature, opt_val)

    for X_split, y_split in splits:
        child = create_Tree(X_split, y_split, depth+1, max_depth, is_real, is_real_feature, criterion)
        branch.children.append(child)
    return branch

def predict_single(root: Node, X_row: pd.Series):
    curr = root

    while curr.threshold is not None:

        feature_val = X_row[curr.feature]

        if isinstance(curr.threshold, (int, float)):
            curr = curr.children[0] if feature_val <= curr.threshold else curr.children[1]
        else:
            if feature_val in curr.threshold:
                index = list(curr.threshold).index(feature_val)
                curr = curr.children[index]
            else:
                print(f"Warning: Feature Value {feature_val} not found in threshold {curr.threshold}")
                return -1
    return curr.output

def predict_result(root: Node, X_test: pd.DataFrame):
    predictions = pd.Series(index = X_test.index)
    for i in X_test.index:
        predictions.loc[i] = predict_single(root, X_test.loc[i, :])
    return predictions

def print_node(root: Node, depth=0, decision="?(X"):
    spacing = "    "*depth

    if root.output is not None:
        print(f" Leaf: Value = {root.output: .3f}")
    elif isinstance(root.threshold, (int, float)):
        print(f" {decision}{root.feature} <= {root.threshold: .3f})")
        print(f"{spacing}   Y:", end = "")
        print_node(root.children[0], depth+1)
        print(f"{spacing}   N:", end = "")
        print_node(root.children[1], depth+1)
    elif root.threshold is not None:
        print(f" {decision}{root.feature} in {list(root.threshold)})")
        for i, val in enumerate(root.threshold):
            print(f"{spacing}   {val}:", end = "")
            print_node(root.children[i], depth+1)

@dataclass
class DecisionTree:
    max_depth: int
    root: Node
    predicted: pd.Series
    criterion: str

    def __init__(self, criterion: str, max_depth: int = 6):
        self.criterion = criterion
        self.max_depth = max_depth
        self.root = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Function to train and construct the decision tree
        """

        X_encoded = one_hot_encoding(X)
        is_real_output = check_ifreal(y)
        is_real_feature = X_encoded.apply(lambda col: check_ifreal(col)).tolist()

        self.root = create_Tree(X_encoded, y, 0, self.max_depth, is_real_output, is_real_feature, self.criterion)



    def predict(self, X: pd.DataFrame) -> pd.Series:
        """
        Funtion to run the decision tree on test inputs
        """

        X_encoded = one_hot_encoding(X)
        self.predicted = predict_result(self.root, X_encoded)
        return self.predicted

    def plot(self) -> None:
        """
        Function to plot the tree

        Output Example:
        ?(X1 > 4)
            Y: ?(X2 > 7)
                Y: Class A
                N: Class B
            N: Class C
        Where Y => Yes and N => No
        """
        
        print_node(self.root)
