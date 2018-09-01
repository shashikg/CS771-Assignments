import numpy as np

X_seen=np.load('data/X_seen.npy') # (40 x N_i x D): 40 feature matrices. X_seen[i] is the N_i x D feature matrix of seen class i
Xtest=np.load('data/Xtest.npy')	# (6180, 4096): feature matrix of the test data.
Ytest=np.load('data/Ytest.npy',)	# (6180, 1): ground truth labels of the test data
class_attributes_seen=np.load('data/class_attributes_seen.npy')	# (40, 85): 40x85 matrix with each row being the 85-dimensional class attribute vector of a seen class.
class_attributes_unseen=np.load('data/class_attributes_unseen.npy')	# (10, 85): 10x85 matrix with each row being the 85-dimensional class attribute vector of an  unseen class.

# u = np.zeros()
def meanSeenClass():
    mean = np.zeros((X_seen.shape[0], X_seen[0].shape[1]))
    for i in range(0, X_seen.shape[0]):
        mean[i] = (np.mean(X_seen[i], axis=0)).reshape(1, X_seen[0].shape[1])
    return mean

def calcSimilarity():
    s = np.dot(class_attributes_unseen, class_attributes_seen.T)
    s = s/(np.sum(s, axis=1)).reshape(s.shape[0],1)
    return s

def meanUnseenClass(s, uSeen):
    mean = np.dot(s, uSeen)
    return mean

def predict(u, x):
    diff = u - x
    sq = np.square(diff)
    theta = np.random.rand(diff.shape[1], 1)
    dist = np.dot(sq, theta)
    return np.argmin(dist) + 1

def classifier(uUnseen):
    # Ytest = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    # Xtest = uUnseen
    Ypred = np.zeros(Ytest.shape)
    error = np.zeros(Ytest.shape)
    for i in range(Ytest.shape[0]):
        Ypred[i] = predict(uUnseen, Xtest[i])
        if (Ypred[i] - Ytest[i]) == 0:
            error[i] = 1

    Acc = np.mean(error)
    return Ypred, Acc


uSeen = meanSeenClass()
s = calcSimilarity()
uUnseen = meanUnseenClass(s, uSeen)
Ypred, Acc = classifier(uUnseen)
print(Acc*100)
