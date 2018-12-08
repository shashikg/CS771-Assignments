import numpy as np
import matplotlib.pyplot as plt

def load_data(loc):
    # Will load the data send the location of the data folder to this function
    # x, y = load_data('data')
    data = np.genfromtxt(loc+'/kmeans_data.txt', delimiter='  ')
    return data

def dist(x, u):
    d = np.zeros((x.shape[0], u.shape[0]))
    for i in range(u.shape[0]):
        diff = x - u[i,:].reshape((1,-1))
        d[:,i] = np.sum(np.square(diff), axis=1)
    return d

def predict(x, u):
    d = dist(x, u)
    c = np.argmin(d, axis=1)
    return c.reshape(-1,1)

def mean(x, c):
    u = np.zeros((2, x.shape[1]))
    u[0,:] = np.mean(x[c==0], axis=0)
    u[1,:] = np.mean(x[c==1], axis=0)
    return u

def cluster():
    x = load_data('data')
    fx = (np.sum(np.square(x), axis=1)).reshape(-1,1)
    u = fx[:2,:]
    c = predict(fx, u)

    for iter in range(10):
        u = mean(fx, c)
        c = predict(fx, u)
        p = (c==1).reshape(c.shape[0])
        n = (c==0).reshape(c.shape[0])

    plt.scatter(x[p,0], x[p,1], c='b')
    plt.scatter(x[n,0], x[n,1], c='g')

cluster()
plt.show()
