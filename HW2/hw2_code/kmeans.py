'''
File: kmeans.py
Project: Downloads
File Created: Feb 2021
Author: Rohit Das
'''

#from itertools import pairwise
import numpy as np


class KMeans(object):

    def __init__(self):  # No need to implement
        pass

    def _init_centers(self, points, K, **kwargs):  # [2 pts]
        """
        Args:
            points: NxD numpy array, where N is # points and D is the dimensionality
            K: number of clusters
            kwargs: any additional arguments you want
        Return:
            centers: K x D numpy array, the centers.
        Hint: Please initialize centers by randomly sampling points from the dataset in case the autograder fails.
        """

        centers = points[np.random.choice(points.shape[0],K,replace=False),:]
        
        return centers

    def _kmpp_init(self, points, K, **kwargs): # [3 pts]
        """
        Args:
            points: NxD numpy array, where N is # points and D is the dimensionality
            K: number of clusters
            kwargs: any additional arguments you want
        Return:
            centers: K x D numpy array, the centers.
        """

        sampled_points = points[np.random.choice(points.shape[0],int(points.shape[0]/100),replace=False),:]
        centers = sampled_points[np.random.choice(sampled_points.shape[0],1),:]
        for cluster_id in range(K-1):
            cluster_pdist = pairwise_dist(sampled_points,centers)
            max_cluster_dist_ind = np.argmax(np.min(cluster_pdist,axis=1),axis=0)
            new_center = sampled_points[max_cluster_dist_ind,:]
            centers = np.append(centers,new_center[None,:],axis=0)
            
        return centers

    def _update_assignment(self, centers, points):  # [10 pts]
        """
        Args:
            centers: KxD numpy array, where K is the number of clusters, and D is the dimension
            points: NxD numpy array, the observations
        Return:
            cluster_idx: numpy array of length N, the cluster assignment for each point

        Hint: You could call pairwise_dist() function.
        """

        cluster_idx = np.argmin(pairwise_dist(centers,points),axis=0)

        return cluster_idx

    def _update_centers(self, old_centers, cluster_idx, points):  # [10 pts]
        """
        Args:
            old_centers: old centers KxD numpy array, where K is the number of clusters, and D is the dimension
            cluster_idx: numpy array of length N, the cluster assignment for each point
            points: NxD numpy array, the observations
        Return:
            centers: new centers, a new K x D numpy array of float dtype, where K is the number of clusters, and D is the dimension.

        HINT: Points may be integer, but the centers should not have to be. Watch out for dtype casting!
        """

        # centers = np.zeros_like(old_centers)
        # for k in range(old_centers.shape[0]):
        #     centers[k,:] = np.mean(points[cluster_idx==k,:],axis=0)

        centers = np.stack([np.mean(points[cluster_idx==k,:],axis=0) for k in range(old_centers.shape[0])],axis=0)

        return centers

    def _get_loss(self, centers, cluster_idx, points):  # [5 pts]
        """
        Args:
            centers: KxD numpy array, where K is the number of clusters, and D is the dimension
            cluster_idx: numpy array of length N, the cluster assignment for each point
            points: NxD numpy array, the observations
        Return:
            loss: a single float number, which is the objective function of KMeans.
        """

        loss = np.sum(np.square(points-centers[cluster_idx]))

        return loss

    def _get_centers_mapping(self, points, cluster_idx, centers):
        # This function has been implemented for you, no change needed.
        # create dict mapping each cluster to index to numpy array of points in the cluster
        centers_mapping = {key : [] for key in [i for i in range(centers.shape[0])]}
        for (p, i) in zip(points, cluster_idx):
            centers_mapping[i].append(p)
        for center_idx in centers_mapping:
            centers_mapping[center_idx] = np.array(centers_mapping[center_idx])
        self.centers_mapping = centers_mapping
        return centers_mapping

    def __call__(self, points, K, max_iters=100, abs_tol=1e-16, rel_tol=1e-16, verbose=False, center_mapping=False, **kwargs):
        """
        This function has been implemented for you, no change needed.

        Args:
            points: NxD numpy array, where N is # points and D is the dimensionality
            K: number of clusters
            max_iters: maximum number of iterations (Hint: You could change it when debugging)
            abs_tol: convergence criteria w.r.t absolute change of loss
            rel_tol: convergence criteria w.r.t relative change of loss
            verbose: boolean to set whether method should print loss (Hint: helpful for debugging)
            kwargs: any additional arguments you want
        Return:
            cluster assignments: Nx1 int numpy array
            cluster centers: K x D numpy array, the centers
            loss: final loss value of the objective function of KMeans
        """
        centers = self._init_centers(points, K, **kwargs)
        for it in range(max_iters):
            cluster_idx = self._update_assignment(centers, points)
            centers = self._update_centers(centers, cluster_idx, points)
            loss = self._get_loss(centers, cluster_idx, points)
            K = centers.shape[0]
            if it:
                diff = np.abs(prev_loss - loss)
                if diff < abs_tol and diff / prev_loss < rel_tol:
                    break
            prev_loss = loss
            if verbose:
                print('iter %d, loss: %.4f' % (it, loss))
        if center_mapping:
            return cluster_idx, centers, loss, self._get_centers_mapping(points, cluster_idx, centers)
        return cluster_idx, centers, loss


def pairwise_dist(x, y):  # [5 pts]
    np.random.seed(1)
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

def silhouette_coefficient(points, cluster_idx, centers, centers_mapping): # [10pts]
    """
    Args:
        points: N x D numpy array
        cluster_idx: N x 1 numpy array
        centers: K x D numpy array, the centers
        centers_mapping: dict with K keys (cluster indicies) each mapping to a C_i x D 
        numpy array with C_i corresponding to the number of points in cluster i
    Return:
        silhouette_coefficient: final coefficient value as a float 
        mu_ins: N x 1 numpy array of mu_ins (one mu_in for each data point)
        mu_outs: N x 1 numpy array of mu_outs (one mu_out for each data point)
    """
    ## 1. Pairwise Distance Matrix
    points_pdist = pairwise_dist(points,points)                                                                                                         # NxN float

    ## 2. In-Cluster Membership Mask
    in_cluster_membership_mask = np.ma.masked_not_equal(np.outer(cluster_idx,np.ones_like(cluster_idx)),cluster_idx)                                  # NxN bool
    ## 3. Calculate Mu_in                               
    points_pdist_in_masked = np.ma.masked_array(points_pdist,mask=in_cluster_membership_mask.mask)                                                  # NxN float
    #print(points_pdist_in_masked)
    mu_ins = np.sum(points_pdist_in_masked,axis=1) / (np.ma.count(in_cluster_membership_mask,axis=1)-1)                                                 # Nx1 float

    ## 4. Out-Cluster Membership Mask
    # take the closest out-cluster, assuming the points' assigned clusters are the closest clusters
    clusters_pdist = pairwise_dist(points,centers)                                                                                                      # NxK float
    clusters_ordinal_dist = np.argsort(clusters_pdist,axis=1)                                                                                           # NxK int
    out_cluster_membership = clusters_ordinal_dist[:,1]                                                                                                 # Nx1 int
    # check for cluster assignments that violate the above assumption
    assumption_violation_mask = np.ma.masked_not_equal(cluster_idx,clusters_ordinal_dist[:,0]).mask                                                       # Nx1 bool           
    if np.any(assumption_violation_mask):
        print("assumption violation mask: " + str(assumption_violation_mask))
        out_cluster_membership[assumption_violation_mask] = clusters_pdist[assumption_violation_mask,0]                                                     # Nx1 int
    out_cluster_membership_mask = np.ma.masked_not_equal(np.outer(out_cluster_membership,np.ones_like(out_cluster_membership)),cluster_idx)  # NxN bool
    ## 5. Calculate Mu_out
    points_pdist_out_masked = np.ma.masked_array(points_pdist,mask=out_cluster_membership_mask.mask)                                                         # NxN float
    mu_outs = np.sum(points_pdist_out_masked,axis=1) / np.ma.count(out_cluster_membership_mask,axis=1)                                              # Nx1 float

    ## 6. Calculate Silhouette Coefficients
    s = (mu_outs - mu_ins) / np.max(np.concatenate((mu_outs.reshape((-1,1)),mu_ins.reshape((-1,1))),axis=1),axis=1)                                     # Nx1 float
    silhouette_coefficient = float(np.mean(s))                                                                                                                 # float

    return silhouette_coefficient, mu_ins, mu_outs