import numpy as np
import matplotlib.pyplot as plt

import sklearn
import sklearn.datasets
import sklearn.linear_model
from data_load import plot_decision_boundary, sigmoid, datasets
noisy_circles, noisy_moons, blobs, gaussian_quantiles, no_structure = datasets()
X, Y = noisy_moons
X, Y = X.T, Y.reshape(1, Y.shape[0])


plt.scatter(X[0, :], X[1, :], c=Y[0],s=40, cmap=plt.cm.Spectral);
plt.plot



shape_X = np.shape(X)
shape_Y = np.shape(Y)
m = len(X[0])




def layer_sizes(X, Y):

    n_x = X.shape[0]
    n_h = 4
    n_y = Y.shape[0]

    return (n_x, n_h, n_y)

(n_x, n_h, n_y) = layer_sizes(X, Y)
print(str(n_x))
print(str(n_h))
print( str(n_y))




def initialize_parameters(n_x, n_h, n_y):

    np.random.seed(2)

    W1 = np.random.randn(n_h, n_x) * 0.01
    b1 = np.zeros((n_h, 1))
    W2 = np.random.randn(n_y, n_h) * 0.01
    b2 = np.zeros((n_y, 1))


    assert (W1.shape == (n_h, n_x))
    assert (b1.shape == (n_h, 1))
    assert (W2.shape == (n_y, n_h))
    assert (b2.shape == (n_y, 1))

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}

    return parameters


parameters = initialize_parameters(n_x, n_h, n_y)
print("W1 = " + str(parameters["W1"]))
print("b1 = " + str(parameters["b1"]))
print("W2 = " + str(parameters["W2"]))
print("b2 = " + str(parameters["b2"]))

def forward_propagation(X, parameters):

    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']

    Z1 = np.dot(W1, X) + b1
    A1 = np.tanh(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = sigmoid(Z2)


    assert (A2.shape == (1, X.shape[1]))

    cache = {"Z1": Z1,
             "A1": A1,
             "Z2": Z2,
             "A2": A2}

    return A2, cache



initialize_parameters(n_x, n_h, n_y)
A2, cache = forward_propagation(X, parameters)


print(np.mean(cache['Z1']) ,np.mean(cache['A1']),np.mean(cache['Z2']),np.mean(cache['A2']))

def compute_cost(A2, Y, parameters):

    m = Y.shape[1]


    logprobs = np.multiply(np.log(A2), Y) + np.multiply(np.log(1 - A2), (1 - Y))
    cost = -np.sum(logprobs) / m


    cost = float(np.squeeze(cost))
    assert (isinstance(cost, float))

    return cost




def backward_propagation(parameters, cache, X, Y):

    m = X.shape[1]


    W1 = parameters['W1']
    W2 = parameters['W2']



    A1 = cache['A1']
    A2 = cache['A2']



    dZ2 = A2 - Y
    dAr = np.dot(dZ2, A1.T)
    dW2 = dAr/m

    dAr = np.sum(dZ2, axis=1, keepdims=True)
    db2 = dAr/m

    dZ1 = np.multiply(np.dot(W2.T, dZ2), 1 - np.power(A1, 2))
    dW1 = np.dot(dZ1, X.T)/m
    dAr = np.sum(dZ1, axis=1, keepdims=True)
    db1 = dAr/m


    grads = {"dW1": dW1,
             "db1": db1,
             "dW2": dW2,
             "db2": db2}

    return grads





grads = backward_propagation(parameters, cache, X, Y)
print ("dW1 = "+ str(grads["dW1"]))
print ("db1 = "+ str(grads["db1"]))
print ("dW2 = "+ str(grads["dW2"]))
print ("db2 = "+ str(grads["db2"]))

def update_parameters(parameters, grads, learning_rate=1.2):


    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']

    dW1 = grads['dW1']
    db1 = grads['db1']
    dW2 = grads['dW2']
    db2 = grads['db2']

    W1 = W1 - learning_rate * dW1
    b1 = b1 - learning_rate * db1
    W2 = W2 - learning_rate * dW2
    b2 = b2 - learning_rate * db2


    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}

    return parameters




def nn_model(X, Y, n_h, num_iterations=10000, print_cost=False):


    np.random.seed(3)
    n_x = layer_sizes(X, Y)[0]
    n_y = layer_sizes(X, Y)[2]


    parameters = initialize_parameters(n_x, n_h, n_y)





    for i in range(0, num_iterations):
        A2, cache = forward_propagation(X, parameters)
        cost = compute_cost(A2, Y, parameters)
        grads = backward_propagation(parameters, cache, X, Y)
        parameters = update_parameters(parameters, grads)
        if print_cost and i % 1000 == 0:
            print ("Cost after iteration %i: %f" % (i, cost))

    return parameters



def predict(parameters, X):

    A2, cache = forward_propagation(X, parameters)
    predictions = np.round(A2)


    return predictions


parameters = nn_model(X, Y, n_h = 2, num_iterations = 10000, print_cost=True)

plot_decision_boundary(lambda x: predict(parameters, x.T), X, Y)
plt.title("Classified Data" )
plt.show()


predictions = predict(parameters, X)
print ('Accuracy: %d' % float((np.dot(Y,predictions.T) + np.dot(1-Y,1-predictions.T))/float(Y.size)*100) + '%')
plt.scatter(X[0, :], X[1, :], c=Y[0],s=40, cmap=plt.cm.Spectral);
plt.title("Scattered Data" + str(4))
plt.show()
