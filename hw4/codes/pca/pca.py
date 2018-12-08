from sklearn.decomposition import PCA
import cPickle as pickle
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv
from numpy.linalg import norm

f = open('../data/mnist_small_py2_3.pkl', 'rb')
data = pickle.load(f)
X = data['X']
Y = data['Y']

# print(Y)

# for n in range(1,11):
#     print(n)
#     pca = PCA(n_components=2)
#     X_embedded = pca.fit_transform(X)
#     # print(X_embedded.shape)
#     np.savetxt('pcav' + str(n) + '.txt', X_embedded, delimiter=',')
