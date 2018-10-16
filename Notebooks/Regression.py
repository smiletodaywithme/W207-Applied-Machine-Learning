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
plt.pause(1)

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
        hypothesis = np.dot(X, theta)
        
        # The loss is the difference between the predictions and the actual target values.
        loss = hypothesis - target_data
        
        # In standard linear regression, we want to minimize the sum of squared losses.
        cost = np.sum(loss**2) / (2*m)
        costs.append(cost) 
        
        # Compute the gradient.
        gradient = np.dot(X.T, loss) / m

        # Update theta, scaling the gradient by the learning rate.
        theta = theta - eta * gradient
        
    return theta, costs

# Run gradient descent and plot the cost vs iterations.
theta, costs = gradient_descent(train_data[:,0:1], train_labels, .01, 500)
plt.plot(costs)
plt.xlabel('Iteration'), plt.ylabel('Cost')
plt.show()

## Let's compare our results to sklearn's regression as well as the algebraic solution to "ordinary least squares". Try increasing the number of iterations above to see whether we get closer.
def OLS(X, Y):
    # Add the intercept.
    X = np.c_[np.ones(X.shape[0]), X]
    
    # We use np.linalg.inv() to compute a matrix inverse.
    return np.dot(inv(np.dot(X.T, X)), np.dot(X.T, Y))

ols_solution = OLS(train_data[:,0:1], train_labels)

lr = LinearRegression(fit_intercept=True)
lr.fit(train_data[:,0:1], train_labels)

print('Our estimated theta:     %.4f + %.4f*CRIM' %(theta[0], theta[1]))
print('OLS estimated theta:     %.4f + %.4f*CRIM' %(ols_solution[0], ols_solution[1]))
print('sklearn estimated theta: %.4f + %.4f*CRIM' %(lr.intercept_, lr.coef_[0]))

## Let's try fitting a model that uses more of the variables. Let's run just a few iterations and check the cost function.
num_feats = 5
theta, costs = gradient_descent(train_data[:,0:num_feats], train_labels, .01, 10)
plt.plot(np.log(costs))
plt.xlabel('Iteration'), plt.ylabel('Log Cost')
plt.show()

## Let's reduce the learning rate and try again. 
start_time = time.time()
theta, costs = gradient_descent(train_data[:,0:num_feats], train_labels, .001, 100000)
train_time = time.time() - start_time
plt.plot(np.log(costs))
plt.xlabel('Iteration'), plt.ylabel('Log Cost')
plt.show()

print('Training time: %.2f secs' %train_time)
print('Our estimated theta:', theta)
print('OLS estimated theta:', OLS(train_data[:,0:num_feats], train_labels))

## Let's examine the distributions of the features we're using.
plt.figure(figsize=(15, 3))
for feature in range(num_feats):
    plt.subplot(1, num_feats, feature+1)
    plt.hist(train_data[:,feature])
    plt.title(boston.feature_names[feature])
plt.show()

## Let's apply the standard scaler:
scaler = preprocessing.StandardScaler()
scaler.fit(train_data)
scaled_train_data = scaler.transform(train_data)
scaled_test_data = scaler.transform(test_data)

plt.figure(figsize=(15, 3))

for feature in range(5):
    plt.subplot(1, 5, feature+1)
    plt.hist(scaled_train_data[:,feature])
    plt.title(boston.feature_names[feature])
plt.show()

## Let's try gradient descent again. We can increase the learning rate and decrease the number of iterations.
start_time = time.time()
theta, costs = gradient_descent(scaled_train_data[:,0:5], train_labels, .01, 5000)
train_time = time.time() - start_time
plt.plot(np.log(costs))
plt.xlabel('Iteration'), plt.ylabel('Log Cost')
plt.show()

print('Training time: %.2f secs' %train_time)
print('Our estimated theta:', theta)
print('OLS estimated theta:', OLS(scaled_train_data[:,0:5], train_labels))
plt.show()