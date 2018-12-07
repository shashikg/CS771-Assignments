import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.ticker import PercentFormatter

# Load the data set
X_seen=np.load('data/X_seen.npy') # (40 x N_i x D): 40 feature matrices. X_seen[i] is the N_i x D feature matrix of seen class i
Xtest=np.load('data/Xtest.npy')	# (6180, 4096): feature matrix of the test data.
Ytest=np.load('data/Ytest.npy',)	# (6180, 1): ground truth labels of the test data
class_attributes_seen=np.load('data/class_attributes_seen.npy')	# (40, 85): 40x85 matrix with each row being the 85-dimensional class attribute vector of a seen class.
class_attributes_unseen=np.load('data/class_attributes_unseen.npy')	# (10, 85): 10x85 matrix with each row being the 85-dimensional class attribute vector of an  unseen class.


def meanSeenClass(xs):
    # This function calculates mean of seen classes
    mean = np.zeros((xs.shape[0], xs[0].shape[1]))
    for i in range(0, xs.shape[0]):
        mean[i] = (np.mean(xs[i], axis=0)).reshape(1, xs[0].shape[1])
    return mean

def meanUnseenClass(u_seen, Aus, As, k):
    # This function calculates the mean of unseen classes from the learned W as per Method2.
    W1 = np.dot(As.T, As) + k*(np.eye(As.shape[1]))
    W2 = np.dot(As.T, u_seen)
    W = np.dot(np.linalg.inv(W1), W2)
    mean = np.dot(Aus, W)
    return mean

def predict(u, x_test, y_test, theta):
    acc = 0.
    dist = np.zeros((y_test.shape[0], u.shape[0]))
    for i in range(u.shape[0]):
        diff = u[i] - x_test
        sq = np.square(diff)
        d = np.dot(sq, theta)
        dist[:, i] = d.reshape(d.shape[0],)

    y_pred = np.argmin(dist, axis=1)
    y_pred = y_pred.reshape(y_pred.shape[0],1)
    y_pred+=1
    acc = 1 - np.count_nonzero(y_pred-y_test)/float(y_test.shape[0])
    return y_pred, acc

def classifier():
    uSeen = meanSeenClass(X_seen)
    theta = np.ones((uSeen.shape[1], 1))
    # Test Class
    u_seen = uSeen
    attributes_seen = class_attributes_seen
    attributes_unseen = class_attributes_unseen
    x_test = Xtest
    y_test = Ytest

    for i in [0.01, 0.1, 1, 10, 20, 50, 100]:
        u_unseen = meanUnseenClass(u_seen, attributes_unseen, attributes_seen, i)
        y_pred, acc = predict(u_unseen, x_test, y_test, theta)

        print("Test accuracy for lamba = " + str(i) + " is: " + str(100*acc))

classifier()
