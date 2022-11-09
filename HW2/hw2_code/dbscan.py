import numpy as np
from kmeans import pairwise_dist

class DBSCAN(object):
    
    def __init__(self, eps, minPts, dataset):
        self.eps = eps
        self.minPts = minPts
        self.dataset = dataset
        
    def fit(self):
        """Fits DBSCAN to dataset and hyperparameters defined in init().
        Args:
            None
        Return:
            cluster_idx: (N, ) int numpy array of assignment of clusters for each point in dataset
        Hint: Using sets for visitedIndices may be helpful here.
        Iterate through the dataset sequentially and keep track of your points' cluster assignments.
        If a point is unvisited or is a noise point (has fewer than the minimum number of neighbor points), then its cluster assignment should be -1.
        Set the first cluster as C = 0
        """
        C = 0
        cluster_idx = -1*np.ones(self.dataset.shape[0])
        visitedIndices = set()
        for index in range(self.dataset.shape[0]):
            if index not in visitedIndices:
                visitedIndices.add(index)
                neighborIndices = self.regionQuery(index)
                visitedIndices = self.expandCluster(index,neighborIndices,C,cluster_idx,visitedIndices)
                C += 1
        return cluster_idx

    def expandCluster(self, index, neighborIndices, C, cluster_idx, visitedIndices):
        """Expands cluster C using the point P, its neighbors, and any points density-reachable to P and updates indices visited, cluster assignments accordingly
           HINT: regionQuery could be used in your implementation
        Args:
            index: index of point P in dataset (self.dataset)
            neighborIndices: (N, ) int numpy array, indices of all points witin P's eps-neighborhood
            C: current cluster as an int
            cluster_idx: (N, ) int numpy array of current assignment of clusters for each point in dataset
            visitedIndices: set of indices in dataset visited so far
        Return:
            None
        Hints: 
            np.concatenate(), np.unique(), np.sort(), and np.take() may be helpful here
            A while loop may be better than a for loop
        """
        cluster_idx[index] = C
        nind_ind = 0
        while nind_ind < len(neighborIndices): 
            nind = int(neighborIndices[nind_ind])
            if nind not in visitedIndices:
                visitedIndices.add(nind)
                #print("visitedIndices: " + str(visitedIndices))
                nind_neighbors = self.regionQuery(nind)
                #print("nind_neighbors: " + str(nind_neighbors))
                if len(nind_neighbors) >= self.minPts:
                    new_neighbors = np.array(list(set(nind_neighbors) - set(neighborIndices)))
                    #print(neighborIndices.shape)
                    #print(new_neighbors.shape)
                    neighborIndices = np.concatenate((neighborIndices,new_neighbors))
                    #print("neighborIndices: " + str(neighborIndices))
            if cluster_idx[nind] == -1:
                cluster_idx[nind] = C 
            nind_ind += 1
        return visitedIndices
        
    def regionQuery(self, pointIndex):
        """Returns all points within P's eps-neighborhood (including P)

        Args:
            pointIndex: index of point P in dataset (self.dataset)
        Return:
            indices: (I, ) int numpy array containing the indices of all points within P's eps-neighborhood
        Hint: pairwise_dist (implemented above) and np.argwhere may be helpful here
        """
        pdist = pairwise_dist(np.atleast_2d(self.dataset),np.atleast_2d(self.dataset[pointIndex,:]))
        indices = np.argwhere(pdist < self.eps)[:,0]
        return indices

def pairwise_dist(x, y):  # [5 pts]
    """
        Args:
            x: N x D numpy array
            y: M x D numpy array
        Return:
                dist: N x M array, where dist2[i, j] is the euclidean distance between
                x[i, :] and y[j, :]
    """

    dist = np.linalg.norm(x[:, None, :] - y[None, :, :], axis=-1)
    return dist