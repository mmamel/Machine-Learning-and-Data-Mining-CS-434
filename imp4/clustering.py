import numpy as np
from random import seed
from random import randint
from math import sqrt
from time import time

class KMeans():
    """
    KMeans. Class for building an unsupervised clustering model
    """

    def __init__(self, k, max_iter=20):

        """
        :param k: the number of clusters
        :param max_iter: maximum number of iterations
        """

        self.k = k
        self.max_iter = max_iter

    def init_center(self, x):
        """
        initializes the center of the clusters using the given input
        :param x: input of shape (n, m)
        :return: updates the self.centers
        """
        #randomly generates one of the training examples to use as the center
        seed()
        usedDict = {}
        self.centers = np.zeros((self.k, x.shape[1]))
        for i in range(self.k):
            repeat=0
            while(repeat == 0):
                index = randint(0,x.shape[0]-1)
                if not str(index) in usedDict:
                    usedDict[str(index)] = "1"
                    repeat = 1
                    self.centers[i]= np.copy(x[index])
        return self.centers

    def revise_centers(self, x, labels):
        """
        it updates the centers based on the labels
        :param x: the input data of (n, m)
        :param labels: the labels of (n, ). Each labels[i] is the cluster index of sample x[i]
        :return: updates the self.centers
        """
        for i in range(self.k):
            wherei = np.squeeze(np.argwhere(labels == i), axis=1)
            self.centers[i, :] = x[wherei, :].mean(0)


    def predict(self, x):
        """
        returns the labels of the input x based on the current self.centers
        :param x: input of (n, m)
        :return: labels of (n,). Each labels[i] is the cluster index for sample x[i]
        """
        #basically calculates the l2 norm (euclidean distance) of every point to every possible center to find the closest
        if self.k ==1:
            labels = np.zeros((x.shape[0]),dtype=int)
        else:
            labels = np.zeros((x.shape[0]), dtype=int)
        dis=np.zeros((self.k,x.shape[0]),dtype=np.float32)
        if self.k != 1:
            for i in range(0, self.k):
                dis[i]=np.add(dis[i],np.linalg.norm(np.subtract(self.centers[i],x),axis=1))
            labels=np.argmin(dis, axis=0)
        return labels

    def get_sse(self, x, labels):
        """
        for a given input x and its cluster labels, it computes the sse with respect to self.centers
        :param x:  input of (n, m)
        :param labels: label of (n,)
        :return: float scalar of sse
        """
        sse = 0.        
        for i in range(x.shape[0]):
            #same l2 norm just summing this time
            sse += np.linalg.norm(np.subtract(self.centers[labels[i]],x[i]))

        ##################################
        #      YOUR CODE GOES HERE       #
        ##################################

        return sse

    def get_purity(self, x, y):
        """
        computes the purity of the labels (predictions) given on x by the model
        :param x: the input of (n, m)
        :param y: the ground truth class labels
        :return:
        """
        #for each cluster(label == k) figure out what the majority class is (1-6), then count how many in each cluster had the correct class using y
        labels = self.predict(x)
        purity = 0
        for i in range(self.k):
            count =[0,0,0,0,0,0,0,0,0,0]
            index = np.where(labels == i)
            for j in range(len(index[0])):
                count[y[index[0][j]]-1]+=1
            maxi=0
            max_index=-1
            for j in range(6):
                if count[j]>maxi:
                    maxi=count[j]
                    max_index=j
            for j in range(len(index[0])):
                if y[index[0][j]] == max_index+1:
                    purity+=1
        purity /= x.shape[0]
        ##################################
        #      YOUR CODE GOES HERE       #
        ##################################
        return purity

    def fit(self, x):
        """
        this function iteratively fits data x into k-means model. The result of the iteration is the cluster centers.
        :param x: input data of (n, m)
        :return: computes self.centers. It also returns sse_veersus_iterations for x.
        """

        # intialize self.centers
        self.init_center(x)
        sse_vs_iter = []
        for iter in range(self.max_iter):
            # finds the cluster index for each x[i] based on the current centers
            labels = self.predict(x)

            # revises the values of self.centers based on the x and current labels
            self.revise_centers(x, labels)
            # computes the sse based on the current labels and centers.
            sse = self.get_sse(x, labels)

            sse_vs_iter.append(sse)

        return sse_vs_iter
