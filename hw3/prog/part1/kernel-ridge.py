import numpy as np
import matplotlib.pyplot as plt

def load_data(loc):
    # Will load the data send the location of the data folder to this function
    # x_train, y_train, x_test, y_test = load_data('data')
    train = np.genfromtxt(loc+'/ridgetrain.txt',delimiter='  ')
    test = np.genfromtxt(loc+'/ridgetest.txt',delimiter='  ')
    return train[:,0], train[:,1], test[:,0], test[:,1]

def kernel(x, y):
    # K = kernel(x_train, x_train)
    return np.exp(-0.1*np.square(x.reshape((-1,1)) - y.reshape((1,-1))))

x_train, y_train, x_test, y_test = load_data('data')
K = kernel(x_train, x_train)
iter = [0, 0.1, 1, 10, 100]
In = np.eye(x_train.shape[0])

for lam in iter:
    alpha = np.dot(np.linalg.inv(K + lam*In), y_train.reshape((-1,1)))
    K_test = kernel(x_train, x_test)
    y_pred = (np.dot(alpha.T, K_test)).reshape((-1,1))

    rmse = np.sqrt(np.mean(np.square(y_test.reshape((-1,1)) - y_pred)))
    print 'RMSE for lambda = ' + str(lam) + ' is ' + str(rmse)

    plt.figure(lam)
    plt.title('lambda = ' + str(lam) + ', rmse = ' + str(rmse))
    plt.plot(x_test, y_pred, 'r*')
    plt.plot(x_test, y_test, 'b*')

plt.show()
