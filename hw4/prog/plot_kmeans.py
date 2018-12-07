from time import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.cluster import KMeans

X = np.genfromtxt('pca/pcav1.txt',delimiter=',')

for i in range(10):
    print(i)
    y_pred = KMeans(n_clusters=10, n_init=1, init='random').fit_predict(X)
    plt.scatter(X[:, 0], X[:, 1], c=y_pred)
    plt.title("PCA Based Init: "+str(i))
    plt.savefig('PCA Based Init ' + str(i)  + '.png', format="png")

# plt.show()
