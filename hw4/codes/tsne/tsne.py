from sklearn.manifold import TSNE
import cPickle as pickle
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv
from numpy.linalg import norm

f = open('../data/mnist_small_py2_3.pkl', 'rb')
data = pickle.load(f)
X = data['X']
Y = data['Y']

for n in range(2,11):
    print(n)
    X_embedded = TSNE(n_components=2).fit_transform(X)
    np.savetxt('tsnev' + str(n) + '.txt', X_embedded, delimiter=',')
