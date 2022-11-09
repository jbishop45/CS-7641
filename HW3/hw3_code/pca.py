import numpy as np
from matplotlib import pyplot as plt


class PCA(object):
    def __init__(self):
        self.U = None
        self.S = None
        self.V = None

    def fit(self, X: np.ndarray) -> None:  # 5 points
        """
        Decompose dataset into principal components by finding the singular value decomposition of the centered dataset X
        You may use the numpy.linalg.svd function
        Don't return anything. You can directly set self.U, self.S and self.V declared in __init__ with
        corresponding values from PCA. See the docstrings below for the expected shapes of U, S, and V

        Args:
            X: (N,D) numpy array corresponding to a dataset

        Return:
            None

        Set:
            self.U: (N, min(N,D)) numpy array
            self.S: (min(N,D), ) numpy array
            self.V: (min(N,D), D) numpy array
        """
        X_centered = np.subtract(X,np.mean(X,axis=0))
        self.U,self.S,self.V = np.linalg.svd(X_centered,full_matrices=False)

    def transform(self, data: np.ndarray, K: int = 2) -> np.ndarray:  # 2 pts
        """
        Transform data to reduce the number of features such that final data (X_new) has K features (columns)
        Utilize self.U, self.S and self.V that were set in fit() method.

        Args:
            data: (N,D) numpy array corresponding to a dataset
            K: int value for number of columns to be kept

        Return:
            X_new: (N,K) numpy array corresponding to data obtained by applying PCA on data
        """
        self.fit(data)
        X_new = self.U[:,:K]*self.S[:K]
        return X_new

    def transform_rv(
        self, data: np.ndarray, retained_variance: float = 0.99
    ) -> np.ndarray:  # 3 pts
        """
        Transform data to reduce the number of features such that the retained variance given by retained_variance is kept
        in X_new with K features
        Utilize self.U, self.S and self.V that were set in fit() method.

        Args:
            data: (N,D) numpy array corresponding to a dataset
            retained_variance: float value for amount of variance to be retained

        Return:
            X_new: (N,K) numpy array corresponding to data obtained by applying PCA on data, where K is the number of columns
                   to be kept to ensure retained variance value is retained_variance
        """
        self.fit(data)
        recovered_var_cumsum = np.power(self.S,2).cumsum(axis=0) / np.power(self.S,2).sum(axis=0)
        K = np.ma.flatnotmasked_edges(np.ma.masked_less(recovered_var_cumsum,retained_variance))[0]+1
        X_new = self.U[:,:K]*self.S[:K]
        # print("X: " + str(data.shape))
        # print("U: " + str(self.U.shape))
        # print("S: " + str(self.S.shape))
        # print("V: " + str(self.V.shape))
        # print("X_new: " + str(X_new.shape))
        return X_new

    def get_V(self) -> np.ndarray:
        """ Getter function for value of V """

        return self.V

    def visualize(self, X: np.ndarray, y: np.ndarray, fig=None) -> None:  # 5 pts
        """
        Use your PCA implementation to reduce the dataset to only 2 features. You'll need to run PCA on the dataset and then transform it so that the new dataset only has 2 features.
        Create a scatter plot of the reduced data set and differentiate points that have different true labels using color.
        Hint: To create the scatter plot, it might be easier to loop through the labels (Plot all points in class '0', and then class '1')
        Hint: To reproduce the scatter plot in the expected outputs, use the colors 'blue', 'magenta', and 'red' for classes '0', '1', '2' respectively.
        
        Args:
            xtrain: (N,D) numpy array, where N is number of instances and D is the dimensionality of each instance
            ytrain: (N,) numpy array, the true labels
            
        Return: None
        """
        X_new = self.transform(X)
        colors = ['blue','magenta','red']
        classes = np.unique(y)
        for i,c in enumerate(classes):
            plt.scatter(X_new[y==c,0],X_new[y==c,1],c=colors[i],marker='x')

        ##################### END YOUR CODE ABOVE, DO NOT CHANGE BELOW #######################
        plt.legend(classes)
        plt.show()
