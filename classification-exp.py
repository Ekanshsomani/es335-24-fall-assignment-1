import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tree.base import DecisionTree
from metrics import *
from sklearn.datasets import make_classification
from sklearn.model_selection import KFold

np.random.seed(42)

# Code given in the question
X, y = make_classification(
    n_features=2, n_redundant=0, n_informative=2, random_state=1, n_clusters_per_class=2, class_sep=0.5)

# For plotting
plt.scatter(X[:, 0], X[:, 1], c=y)

# Create a pandas data frame and series
X_df = pd.DataFrame(X, columns=['feature_1', 'feature_2'])
y_pds = pd.Series(y)

# split train and test data
train_size = round(0.7 * len(X_df))
X_train, X_test = X_df[:train_size], X_df[train_size:]
y_train, y_test = y_pds[:train_size], y_pds[train_size:]

# Part-A
# create tree by training it on the data
tree = DecisionTree(criterion = "mse", max_depth = 5)
tree.fit(X_train, y_train)

# predict and check the scores
y_pred = tree.predict(X_test)

acc = accuracy(y_pred, y_test)
precision_0 = precision(y_pred, y_test, cls=0)
precision_1 = precision(y_pred, y_test, cls=1)
recall_0 = recall(y_pred, y_test, cls=0)
recall_1 = recall(y_pred, y_test, cls=1)

print(f"Accuracy: {acc}")
print(f"Precision (Class 0): {precision_0}")
print(f"Precision (Class 1): {precision_1}")
print(f"Recall (Class 0): {recall_0}")
print(f"Recall (Class 1): {recall_1}")

# Part-B: 5-Fold Cross-Validation and Nested Cross-Validation for Optimum Depth

# Set up 5-Fold Cross-Validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

depths = range(1, 11)  # Testing tree depths from 1 to 10
best_depth = None
best_score = -np.inf

# Nested Cross-Validation to find the optimum depth
for depth in depths:
    fold_scores = []
    for train_index, test_index in kf.split(X_df):
        X_train_fold, X_test_fold = X_df.iloc[train_index], X_df.iloc[test_index]
        y_train_fold, y_test_fold = y_pds.iloc[train_index], y_pds.iloc[test_index]

        # Train a tree with the current depth
        tree = DecisionTree(criterion="mse", max_depth=depth)
        tree.fit(X_train_fold, y_train_fold)

        # Predict and calculate accuracy on the validation fold
        y_pred_fold = tree.predict(X_test_fold)
        fold_score = accuracy(y_pred_fold, y_test_fold)
        fold_scores.append(fold_score)

    # Average score for this depth
    avg_score = np.mean(fold_scores)
    print(f"Depth {depth}: Avg Accuracy {avg_score}")

    # Check if this depth has the best score
    if avg_score > best_score:
        best_score = avg_score
        best_depth = depth

print(f"Best Depth: {best_depth} with Accuracy: {best_score}")