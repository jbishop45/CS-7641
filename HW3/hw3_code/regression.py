import numpy as np
from typing import Tuple, List


class Regression(object):
    def __init__(self):
        pass

    def rmse(self, pred: np.ndarray, label: np.ndarray) -> float:  # [5pts]
        """
        Calculate the root mean square error.

        Args:
            pred: (N, 1) numpy array, the predicted labels
            label: (N, 1) numpy array, the ground truth labels
        Return:
            A float value
        """
        return np.sqrt(np.mean(np.power(np.subtract(pred,label),2)))

    def construct_polynomial_feats(
        self, x: np.ndarray, degree: int
    ) -> np.ndarray:  # [5pts]
        """
        Given a feature matrix x, create a new feature matrix
        which is all the possible combinations of polynomials of the features
        up to the provided degree

        Args:
            x: N x D numpy array, where N is number of instances and D is the
               dimensionality of each instance.
            degree: the max polynomial degree
        Return:
            feat:
                For 1-D array, numpy array of shape Nx(degree+1), remember to include
                the bias term. feat is in the format of:
                [[1.0, x1, x1^2, x1^3, ....,],
                 [1.0, x2, x2^2, x2^3, ....,],
                 ......
                ]
        Hints:
            - For D-dimensional array: numpy array of shape N x (degree+1) x D, remember to include
            the bias term.
            - Example:
            For inputs x: (N = 3 x D = 2) and degree: 3,
            feat should be:

            [[[ 1.0        1.0]
                [ x_{1,1}    x_{1,2}]
                [ x_{1,1}^2  x_{1,2}^2]
                [ x_{1,1}^3  x_{1,2}^3]]

                [[ 1.0        1.0]
                [ x_{2,1}    x_{2,2}]
                [ x_{2,1}^2  x_{2,2}^2]
                [ x_{2,1}^3  x_{2,2}^3]]

                [[ 1.0        1.0]
                [ x_{3,1}    x_{3,2}]
                [ x_{3,1}^2  x_{3,2}^2]
                [ x_{3,1}^3  x_{3,2}^3]]]

        """
        feat = np.stack([np.power(x,d) for d in range(degree+1)],axis=1)
        return feat

    def predict(self, xtest: np.ndarray, weight: np.ndarray) -> np.ndarray:  # [5pts]
        """
        Using regression weights, predict the values for each data point in the xtest array

        Args:
            xtest: (N,D) numpy array, where N is the number
                   of instances and D is the dimensionality
                   of each instance
            weight: (D,1) numpy array, the weights of linear regression model
        Return:
            prediction: (N,1) numpy array, the predicted labels
        """
        prediction = xtest @ weight
        return prediction

    # =================
    # LINEAR REGRESSION
    # =================

    def linear_fit_closed(
        self, xtrain: np.ndarray, ytrain: np.ndarray
    ) -> np.ndarray:  # [5pts]
        """
        Fit a linear regression model using the closed form solution

        Args:
            xtrain: (N,D) numpy array, where N is number of instances and D is the dimensionality of each instance
            ytrain: (N,1) numpy array, the true labels
        Return:
            weight: (D,1) numpy array, the weights of linear regression model
        Hints:
            - For pseudo inverse, you can use numpy linear algebra function (np.linalg.pinv)
        """
        weight = np.linalg.pinv(xtrain) @ ytrain
        return weight

    def linear_fit_GD(
        self,
        xtrain: np.ndarray,
        ytrain: np.ndarray,
        epochs: int = 5,
        learning_rate: float = 0.001,
    ) -> Tuple[np.ndarray, List[float]]:  # [5pts]
        """
        Fit a linear regression model using gradient descent

        Args:
            xtrain: (N,D) numpy array, where N is number
                    of instances and D is the dimensionality of each
                    instance
            ytrain: (N,1) numpy array, the true labels
        Return:
            weight: (D,1) numpy array, the weights of linear regression model
            loss_per_epoch: (epochs,) list of floats, rmse of each epoch
        """
        N,D = xtrain.shape
        weight = np.zeros((D,1))
        loss_per_epoch = []
        for e in range(epochs):
            weight += learning_rate/N * xtrain.T @ (ytrain - xtrain @ weight)
            loss_per_epoch.append(self.rmse(self.predict(xtrain,weight),ytrain))
        return weight, loss_per_epoch

    def linear_fit_SGD(
        self,
        xtrain: np.ndarray,
        ytrain: np.ndarray,
        epochs: int = 100,
        learning_rate: float = 0.001,
    ) -> Tuple[np.ndarray, List[float]]:  # [5pts]
        """
        Fit a linear regression model using stochastic gradient descent

        Args:
            xtrain: (N,D) numpy array, where N is number
                    of instances and D is the dimensionality of each
                    instance
            ytrain: (N,1) numpy array, the true labels
        Return:
            weight: (D,1) numpy array, the weights of linear regression model
            loss_per_step: (N*epochs,) list of floats, rmse calculated after each update step

        NOTE: For autograder purposes, iterate through the dataset SEQUENTIALLY, NOT stochastically.


        Note: Keep in mind that the number of epochs is the number of
        complete passes through the training dataset. SGD updates the
        weight for one datapoint at a time, but for each epoch, you'll
        need to go through all of the points.
        """
        N,D = xtrain.shape
        weight = np.zeros((D,1))
        loss_per_step = []
        for e in range(epochs):
            for i in range(N):
                weight += learning_rate * np.atleast_2d(xtrain[i,:]).T @ (np.atleast_2d(ytrain[i,:]) - np.atleast_2d(xtrain[i,:]) @ weight)
                loss_per_step.append(self.rmse(self.predict(xtrain,weight),ytrain))
        return weight, loss_per_step

    # =================
    # RIDGE REGRESSION
    # =================

    def ridge_fit_closed(
        self, xtrain: np.ndarray, ytrain: np.ndarray, c_lambda: float
    ) -> np.ndarray:  # [5pts]
        """
        Fit a ridge regression model using the closed form solution

        Args:
            xtrain: (N,D) numpy array, where N is number of instances and D is the dimensionality of each instance
            ytrain: (N,1) numpy array, the true labels
            c_lambda: float value
        Return:
            weight: (D,1) numpy array, the weights of ridge regression model
        Hints:
            - For pseudo inverse, you can use numpy linear algebra function (np.linalg.pinv)
            - You should adjust your I matrix to handle the bias term differently than the rest of the terms
        """
        weight = (np.linalg.pinv(xtrain.T @ xtrain + c_lambda*np.eye(xtrain.shape[1])) @ xtrain.T) @ ytrain
        return weight

    def ridge_fit_GD(
        self,
        xtrain: np.ndarray,
        ytrain: np.ndarray,
        c_lambda: float,
        epochs: int = 500,
        learning_rate: float = 1e-7,
    ) -> Tuple[np.ndarray, List[float]]:  # [5pts]
        """
        Fit a ridge regression model using gradient descent.

        Args:
            xtrain: (N,D) numpy array, where N is number
                    of instances and D is the dimensionality of each
                    instance
            ytrain: (N,1) numpy array, the true labels
            c_lambda: float value
        Return:
            weight: (D,1) numpy array, the weights of linear regression model
            loss_per_epoch: (epochs,) list of floats, rmse of each epoch
        """
        N,D = xtrain.shape
        weight = np.zeros((D,1))
        loss_per_epoch = []
        for e in range(epochs):
            weight += weight * (2* c_lambda * learning_rate) + learning_rate/N * xtrain.T @ (ytrain - xtrain @ weight)
            loss_per_epoch.append(self.rmse(self.predict(xtrain,weight),ytrain))
        return weight, loss_per_epoch

    def ridge_fit_SGD(
        self,
        xtrain: np.ndarray,
        ytrain: np.ndarray,
        c_lambda: float,
        epochs: int = 100,
        learning_rate: float = 0.001,
    ) -> Tuple[np.ndarray, List[float]]:  # [5pts]
        """
        Fit a ridge regression model using stochastic gradient descent.

        Args:
            xtrain: (N,D) numpy array, where N is number
                    of instances and D is the dimensionality of each
                    instance
            ytrain: (N,1) numpy array, the true labels
        Return:
            weight: (D,1) numpy array, the weights of linear regression model
            loss_per_step: (N*epochs,) list of floats, rmse calculated after each update step

        NOTE: For autograder purposes, iterate through the dataset SEQUENTIALLY, NOT stochastically.

        Note: Keep in mind that the number of epochs is the number of
        complete passes through the training dataset. SGD updates the
        weight for one datapoint at a time, but for each epoch, you'll
        need to go through all of the points.
        """
        N,D = xtrain.shape
        weight = np.zeros((D,1))
        loss_per_step = []
        for e in range(epochs):
            for i in range(N):
                # weight = np.linalg.pinv(xtrain) @ ytrain                                                                                        # linear closed
                # weight = (np.linalg.pinv(xtrain.T @ xtrain + c_lambda*np.eye(xtrain.shape[1])) @ xtrain.T) @ ytrain                             # ridge closed
                # weight += learning_rate/N * xtrain.T @ (ytrain - xtrain @ weight)                                                               # linear GD
                # weight += learning_rate * np.atleast_2d(xtrain[i,:]).T @ (np.atleast_2d(ytrain[i,:]) - np.atleast_2d(xtrain[i,:]) @ weight)     # linear SGD
                # weight += weight * (2 * c_lambda * learning_rate) + learning_rate/N * xtrain.T @ (ytrain - xtrain @ weight)                     # ridge GD
                
                weight += weight * (-1*c_lambda * learning_rate)/N + learning_rate * np.atleast_2d(xtrain[i,:]).T @ (np.atleast_2d(ytrain[i,:]) - np.atleast_2d(xtrain[i,:]) @ weight) # ridge SGD
                loss_per_step.append(self.rmse(self.predict(xtrain,weight),ytrain))
        return weight, loss_per_step

    def ridge_cross_validation(
        self, X: np.ndarray, y: np.ndarray, kfold: int = 10, c_lambda: float = 100
    ) -> float:  # [5 pts]
        """
        For each of the kfolds of the provided X, y data, fit a ridge regression model
        and then evaluate the RMSE. Return the mean RMSE across all kfolds

        Args:
            X : (N,D) numpy array, where N is the number of instances and D is the dimensionality of each instance
            y : (N,1) numpy array, true labels
            kfold: Number of folds you should take while implementing cross validation.
            c_lambda: Value of regularization constant
        Returns:
            meanErrors: float, average rmse error
        Hints:
            - np.concatenate might be helpful.
            - Use ridge_fit_closed for this function.
            - Look at 3.5 to see how this function is being used.
            - If kfold=10:
                split X and y into 10 equal-size folds
                use 90 percent for training and 10 percent for test
        """
        fold_size = int(X.shape[0]/kfold) #assumes N evenly divides kfold
        meanErrors = 0
        for k in range(kfold):
            indices=list(range(k*fold_size,k*fold_size+fold_size-1))
            X_train = X
            y_train = y
            X_test = np.take(X_train,indices,axis=0)
            y_test = np.take(y_train,indices,axis=0)
            weight = self.ridge_fit_closed(X_train,y_train,c_lambda)
            meanErrors += self.rmse(self.predict(X_test,weight),y_test)/kfold
        return meanErrors

    def hyperparameter_search(
        self, X: np.ndarray, y: np.ndarray, lambda_list: List[float], kfold: int
    ) -> Tuple[float, float, List[float]]:
        """
        PROVIDED TO STUDENTS
        Search over the given list of possible lambda values lambda_list
        for the one that gives the minimum average error from cross-validation

        Args:
            X : (N,D) numpy array, where N is the number of instances and D is the dimensionality of each instance
            y : (N,1) numpy array, true labels
            lambda_list: list of regularization constants to search from
            kfold: Number of folds you should take while implementing cross validation.
        Returns:
            best_lambda: (float) the best value for the regularization const giving the least RMSE error
            best_error: (float) the RMSE error achieved using the best_lambda
            error_list: list[float] list of errors for each lambda value given in lambda_list
        """

        best_error = None
        best_lambda = None
        error_list = []

        for lm in lambda_list:
            err = self.ridge_cross_validation(X, y, kfold, lm)
            error_list.append(err)
            if best_error is None or err < best_error:
                best_error = err
                best_lambda = lm

        return best_lambda, best_error, error_list
