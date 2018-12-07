import cPickle as pickle
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv
from numpy.linalg import norm

f = open('data/facedata_py2_3.pkl', 'rb')
data = pickle.load(f)
X = data['X']
D = X.shape[1]
N = X.shape[0]

iter = [10, 20, 30, 40, 50, 100]

u = np.mean(X, axis=0).reshape(1,D)

for K in iter:
    print("Running for K = " + str(K))
    Z = np.random.randn(N, K)
    W = np.zeros((D,K))

    for i in range(100):
        W = np.dot(X.T, np.dot(Z, inv(np.dot(Z.T, Z))))
        Z = np.dot(X, np.dot(W, inv(np.dot(W.T, W))))
        loss = np.mean(np.dot((X-np.dot(Z,W.T)).T, X-np.dot(Z,W.T)))
        print_loss = "Loss for iter = " + str(i) + " is: " + str(loss)
        print(print_loss)

    X_hat = u + np.dot(Z[0:5,:], W.T)
    basis_img = W[:,0:10]

    # plotting results
    plt.figure("Reconstructed")
    plt.suptitle('For K = ' + str(K), fontsize=16)
    for i in range(5):
        img1 = X[i].reshape(64,64)
        img2 = X_hat[i].reshape(64,64)

        plt.subplot(2,5, 1 + i)
        plt.imshow(img1.T)
        plt.title("Original")
        plt.subplot(2,5, 6 + i)
        plt.title("Estimate")
        plt.imshow(img2.T)
    plt.savefig('Reconstructed for K ' + str(K) + '.png', format="png")

    plt.figure('W as Image')
    plt.suptitle('For K = ' + str(K), fontsize=16)
    for i in range(5):
        img1 = basis_img[:,5+i].reshape(64,64)
        img2 = basis_img[:,i].reshape(64,64)

        plt.subplot(2,5, 1 + i)
        plt.imshow(img1.T)
        plt.subplot(2,5, 6 + i)
        plt.imshow(img2.T)
    plt.savefig('Weight for K ' + str(K) + '.png', format="png")
    # plt.suptitle('For K = ' + str(K), fontsize=16)

plt.show()
f.close()
