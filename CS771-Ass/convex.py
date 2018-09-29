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

# u = np.zeros()
def meanSeenClass(xs):
    # This function calculates mean of seen classes
    mean = np.zeros((xs.shape[0], xs[0].shape[1]))
    for i in range(0, xs.shape[0]):
        mean[i] = (np.mean(xs[i], axis=0)).reshape(1, xs[0].shape[1])
    return mean

def calcSimilarity(attributes_unseen, attributes_seen):
    # Calculates the similarity
    s = np.dot(attributes_unseen, attributes_seen.T)
    s = s/(np.sum(s, axis=1)).reshape(s.shape[0],1)
    return s

def meanUnseenClass(s, u_seen):
    # This function calculates the mean of unseen classes as per Method1.
    mean = np.dot(s, u_seen)
    print mean
    return mean

# def predict(u, x_test, y_test, theta):
#     y_pred = np.zeros(y_test.shape)
#     for i in range(x_test.shape[0]):
#         diff = u - x_test[i]
#         sq = np.square(diff)
#         dist = np.dot(sq, theta)
#         h = np.argmin(sq, axis = 0).reshape(sq.shape[1], 1)
#         # print(h)
#         # plt.hist(h)
#         # print(hist.shape)
#         hist = np.histogram(h, 10)
#         print(np.argmax(hist[0]))
#         y_pred[i] = np.argmin(dist) + 1
#     acc = 1 - np.count_nonzero(y_pred-y_test)/float(y_test.shape[0])
#     return y_pred, acc

def predict(u, x_test, y_test, theta):
    acc = 0.
    dist = np.zeros((y_test.shape[0], u.shape[0]))
    for i in range(u.shape[0]):
        diff = u[i] - x_test
        sq = np.square(diff)
        d = np.dot(sq, theta)
        dist[:, i] = d.reshape(d.shape[0],)

    y_argsort = np.argsort(dist, axis=1)
    y_argsort+=1
    y_sort = np.sort(dist, axis=1)

    y_pred = np.argmin(dist, axis=1)
    y_pred = y_pred.reshape(y_pred.shape[0],1)
    y_pred+=1
    acc = 1 - np.count_nonzero(y_pred-y_test)/float(y_test.shape[0])
    # for i in range(y_test.shape[0]):
    #     print("For Test No.:" + str(i))
    #     print(y_argsort[i])
    #     print(y_sort[i])
    #     # print()
    #     print(y_test[i])
    #     print(y_pred[i])
    return y_pred, acc

def train(u, x_seen, theta):
    diff = u - x_seen
    sq = np.sum(np.square(diff), axis=0).reshape(diff.shape[1], 1)
    d = theta*sq
    # d/=np.sum(d)
    theta = theta - 0.1*d
    theta/=np.sum(theta)
    # print(np.sum(theta))
    # for i in range(x_seen.shape[0]):
    #     diff = (u - x_seen[i]).reshape(u.shape[0], 1)
    #     sq = np.square(diff)#.reshape(diff.shape[1], 1)
    #     d = theta*sq
    #     theta = theta - 0.01*d
    #     theta/=np.sum(theta)
        # print(i)
    # print(np.sum(d))
    return theta, d

def classifier():
    uSeen = meanSeenClass(X_seen)
    theta = np.ones((uSeen.shape[1], 1))
    theta/=np.sum(theta)
    print(np.sum(theta))

    u_seen = uSeen[:30]
    attributes_seen = class_attributes_seen[:30]
    attributes_unseen = class_attributes_seen[30:uSeen.shape[0]]
    s = calcSimilarity(attributes_unseen, attributes_seen)
    u_unseen = meanUnseenClass(s, u_seen)
    p = 0.

    for j in range(25):
        # u_seen = uSeen[:30]
        # attributes_seen = class_attributes_seen[:30]
        # attributes_unseen = class_attributes_seen[30:uSeen.shape[0]]
        # s = calcSimilarity(attributes_unseen, attributes_seen)
        # u_unseen = meanUnseenClass(s, u_seen)
        p = 0.
        # print("from seen data")
        for i in range(0, 40):
            # p = 0.
            x_test = X_seen[i]
            theta, d = train(uSeen[i], x_test, theta)
            # p = d
            # p+=acc
        #     print(i)
        # print(p)

    # print("from trained one")
        # for i in range(30, uSeen.shape[0]):
        #     x_test = X_seen[i]
        #     y_test = np.ones((X_seen[i].shape[0],1)) + i%30
        #     y_pred, acc = predict(u_unseen, x_test, y_test, theta)
        #     p+=acc
        #     # print(acc)
        #
        # print(10*p)

    # Test Class
    u_seen = uSeen
    attributes_seen = class_attributes_seen
    attributes_unseen = class_attributes_unseen
    x_test = Xtest
    y_test = Ytest

    s = calcSimilarity(attributes_unseen, attributes_seen)
    u_unseen = meanUnseenClass(s, u_seen)
    y_pred, acc = predict(u_unseen, x_test, y_test, theta)

    print(100*acc)

classifier()
# print(np.sum(class_attributes_unseen, axis=0))
