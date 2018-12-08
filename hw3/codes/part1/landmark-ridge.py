import numpy as np
import matplotlib.pyplot as plt

def load_data(loc):
    # Will load the data send the location of the data folder to this function
    # x_train, y_train, x_test, y_test = load_data('data')
    train = np.genfromtxt(loc+'/ridgetrain.txt',delimiter='  ')
    test = np.genfromtxt(loc+'/ridgetest.txt',delimiter='  ')
    return train[:,0], train[:,1], test[:,0], test[:,1]

def landmark(x, y):
    #  = landmark(x_train, x_train)
    return np.exp(-0.1*np.square(x.reshape((-1,1)) - y.reshape((1,-1))))

x_train, y_train, x_test, y_test = load_data('data')

iter = [2, 5, 20, 50, 100]

for L in iter:
    z = np.random.choice(x_train, L, replace=False)
    Id = np.eye(L)
    xf_train = landmark(x_train, z)

    W = np.dot(np.linalg.inv(np.dot(xf_train.T, xf_train) + 0.1*Id), np.dot(xf_train.T, y_train.reshape((-1,1))))

    xf_test = landmark(x_test, z)

    y_pred = np.dot(xf_test, W)

    rmse = np.sqrt(np.mean(np.square(y_test.reshape((-1,1)) - y_pred)))
    print 'RMSE for lambda = ' + str(L) + ' is ' + str(rmse)

    plt.figure(L)
    plt.title('L = ' + str(L) + ', rmse = ' + str(rmse))
    plt.plot(x_test, y_pred, 'r*')
    plt.plot(x_test, y_test, 'b*')

plt.show()
