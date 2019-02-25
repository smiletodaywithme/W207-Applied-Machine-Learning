## Experiment with Gradient Descent for Linear Regression.

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import time
from numpy.linalg import inv
from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing

np.set_printoptions(precision=4, suppress=True)

## Load the Boston housing data. 
boston = load_boston()
X, Y = boston.data, boston.target
plt.hist(Y, 50)
plt.xlabel('Median value (in $1000)')
print(boston.DESCR)
plt.show()

## Shuffle the data, but make sure that the features and accompanying labels stay in sync.
# As usual, let's create separate training and test data.
np.random.seed(0)
shuffle = np.random.permutation(np.arange(X.shape[0]))
X, Y = X[shuffle], Y[shuffle]

# Split into train and test.
train_data, train_labels = X[:350], Y[:350]
test_data, test_labels = X[350:], Y[350:]

## Let's implement gradient descent.
def gradient_descent(train_data, target_data, eta, num_iters):
    # Add a 1 to each feature vector.
    X = np.c_[np.ones(train_data.shape[0]), train_data]
    
    # m = number of samples, k = number of features
    m, k = X.shape
    
    # Initially, set all the parameters to 1.
    theta = np.ones(k)
    
    # Keep track of costs after each step.
    costs = []
    
    for iter in range(0, num_iters):
        # Get the current predictions for the training examples given the current estimate of theta.
        hypothesis = # students code here ...
        
        # The loss is the difference between the predictions and the actual target values.
        loss = # students code here ...
        
        # Мinimize the sum of squared losses.
        cost = #  students code here ...
        costs.append(cost) 
        
        # Compute the gradient.
        gradient = # students code here ...

        # Update theta, scaling the gradient by the learning rate.
        theta = # students code here ...
        
    return theta, costs

# Run gradient descent and plot the cost vs iterations.
theta, costs = gradient_descent(train_data[:,0:1], train_labels, .01, 200)
plt.plot(costs)
plt.xlabel('Iteration'), plt.ylabel('Cost')
plt.show()