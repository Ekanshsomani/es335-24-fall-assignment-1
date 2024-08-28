import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from tree.base import DecisionTree
from metrics import *

np.random.seed(42)
num_average_time = 100  # Number of times to run each experiment to calculate the average values

# Function to create fake data (take inspiration from usage.py)
def fake_data(N, M, is_discrete = True, is_real_output = False):
    X = np.random.randint(0, 2, size=(N, M)) if is_discrete else np.random.rand(N, M)
    y = np.random.rand(N) if is_real_output else np.random.randint(0, 2, size=N)
    return pd.DataFrame(X), pd.Series(y)

# Function to calculate average time (and std) taken by fit() and predict() for different N and P for 4 different cases of DTs
def run_experiment(N, M, is_discrete, is_real_output, iterations=3):
    train_time = []
    pred_time = []
    X, y = fake_data(N, M, is_discrete, is_real_output)
    
    for _ in range(iterations):
        
        model = DecisionTree(criterion = ('mse' if is_real_output else 'entropy'))

        start_time = time.time()
        model.fit(X, y)
        train_time.append(time.time() - start_time)
        
        start_time = time.time()
        model.predict(X)
        pred_time.append(time.time() - start_time)
    
    train_avg = np.mean(train_time)
    train_var = np.var(train_time)
    print("Training Variance: ", train_var)
    pred_avg = np.mean(pred_time)
    pred_var = np.var(pred_time)
    print("Prediction Variance: ", pred_var)
    return train_avg, pred_avg

# Run the functions, Learn the DTs and Show the results/plots
results = []

for N in [10, 20, 30, 50]:
    for M in [5, 10]:
        for i in [True, False]:
            for o in [True, False]:
                train_avg, pred_avg = run_experiment(N, M, i, o)
                results.append((N, M, i, o, train_avg, pred_avg))

results_df = pd.DataFrame(results, columns=['N', 'M', 'Is Input Discrete', 'Is Output Real', 'Train Time', 'Prediction Time'])
print(results_df)

# Function to plot the results
def plot(name):
    fig, axs = plt.subplots(2, 2, figsize=(15, 12))

    # Define markers and colors for different M values
    markers = {5: 'o', 10: 's'}  # Marker for M=5 and M=10
    colors = {5: 'blue', 10: 'green'}  # Colors for M=5 and M=10

    # Mapping for subplot titles
    titles = {
        (True, True): 'Discrete Input, Real Output',
        (True, False): 'Discrete Input, Discrete Output',
        (False, True): 'Continuous Input, Real Output',
        (False, False): 'Continuous Input, Discrete Output'
    }

    # Plot Learning Time for each combination of discrete/continuous input and real/discrete output
    for i, (is_discrete_input, is_real_output) in enumerate([(True, True), (True, False), (False, True), (False, False)]):
        for M in [5, 10]:
            subset = results_df[(results_df['Is Input Discrete'] == is_discrete_input) & 
                                (results_df['Is Output Real'] == is_real_output) & 
                                (results_df['M'] == M)]
            axs.plot(subset['N'], subset[name], marker=markers[M], color=colors[M], label=f'M={M}')
        
        axs.set_title(titles[(is_discrete_input, is_real_output)])
        axs.set_xlabel('N (Number of Samples)')
        axs.set_ylabel('Average Learning Time (s)')
        axs.legend()
        axs.grid(True)
    fig.savefig("Time Complexity" + name + ".png", dpi = 300, bbox_inches = 'tight')

plot("Train Time")
plot("Prediction Time")