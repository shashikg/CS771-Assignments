import numpy as np
import matplotlib.pyplot as plt

data = np.genfromtxt('data/binclassv2.txt',delimiter=',')

def calcMLE(x, pF, nF):
    mup = np.mean(x[pF], axis=0)
    mun = np.mean(x[nF], axis=0)
    k = np.std(x[pF], axis=0)
    sigmap = np.mean(k*k)
    k = np.std(x[nF], axis=0)
    sigman = np.mean(k*k)
    mup.reshape((mup.shape[0],1))
    mun.reshape((mun.shape[0],1))

    return np.array([mup, mun]), np.array([sigmap, sigman])

def plotDataPoints(x, y, pF, nF):
    plt.plot(x[pF,0], x[pF,1], 'r*')
    plt.plot(x[nF,0], x[nF,1], 'b*')

# def DecisionBoundaryA(mu, sigma):


def DecisionBoundaryB(mu, x2):
    a = np.dot(mu[:,1].T, mu[:,1]) - np.dot(mu[:,0].T, mu[:,0])
    h = 2*(mu[:,1] - mu[:,0])
    x1 = (-x2*h[1] + a)/h[0]
    plt.plot(x1,x2, 'k')

def main():
    x = data[:,:data.shape[1]-1]
    y = data[:,data.shape[1]-1]
    pF = y>0
    nF = y<0
    x2 = np.arange(np.min(x[:,1]),np.max(x[:,1]),0.01)

    mu, sigma = calcMLE(x, pF, nF)

    plotDataPoints(x, y, pF, nF)
    DecisionBoundaryB(mu, x2)
    # DecisionBoundaryA(mu, sigma)
    # plt.plot(mu[0,0], mu[1,0], 'ro')
    # plt.plot(mu[0,1], mu[1,1], 'bo')
    plt.show()


main()
