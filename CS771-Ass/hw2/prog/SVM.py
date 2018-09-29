import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm

data = np.genfromtxt('data/binclass.txt',delimiter=',')

def plotDataPoints(x, y, pF, nF):
    plt.plot(x[pF,0], x[pF,1], 'r*')
    plt.plot(x[nF,0], x[nF,1], 'b*')

def plotDecisionBoundary(clf, y, x):
    X, Y = np.meshgrid(x, y)
    Z = clf.predict(np.c_[X.ravel(), Y.ravel()])
    Z = Z.reshape(X.shape)
    plt.contour(X, Y, Z)

def main():
    x = data[:,:data.shape[1]-1]
    y = data[:,data.shape[1]-1]
    pF = y>0
    nF = y<0
    x2 = np.arange(np.min(x[:,1]),np.max(x[:,1]),0.05)
    x1 = np.arange(np.min(x[:,0]),np.max(x[:,0]),0.05)

    clf = svm.SVC(kernel='linear', C=1)
    clf.fit(x, y)

    plotDataPoints(x, y, pF, nF)
    plotDecisionBoundary(clf, x2, x1)
    plt.show()


main()
