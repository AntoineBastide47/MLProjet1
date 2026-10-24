"""Basic ML methods required by project 1 (Table 1 of the project description).

All vectors are 1D numpy arrays of shape (N,) or (D,).
Every function returns (w, loss): the last weight vector and its loss.
The loss of regularized methods does not include the penalty term.
"""

import numpy as np


def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    """Linear regression using gradient descent.

    The loss is the MSE with factor 0.5: L(w) = 1/(2N) * ||y - tx w||^2.

    Args:
        y: numpy array of shape (N,), targets.
        tx: numpy array of shape (N, D), features.
        initial_w: numpy array of shape (D,), initial weights.
        max_iters: int, number of gradient steps.
        gamma: float, step size.

    Returns:
        w: numpy array of shape (D,), last weights.
        loss: scalar, MSE loss of the last weights.
    """
    w = initial_w.copy()
    N = y.shape[0]
    invN = 1 / N

    for _ in range(max_iters):
        error = tx @ w - y
        gradient = (tx.T @ error) * invN
        w = w - gamma * gradient

    error = tx @ w - y
    return w, np.sum(error ** 2) / (2 * N)


def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma):
    """Linear regression using stochastic gradient descent with mini-batch size 1.

    The loss is the MSE with factor 0.5: L(w) = 1/(2N) * ||y - tx w||^2.

    Args:
        y: numpy array of shape (N,), targets.
        tx: numpy array of shape (N, D), features.
        initial_w: numpy array of shape (D,), initial weights.
        max_iters: int, number of stochastic gradient steps.
        gamma: float, step size.

    Returns:
        w: numpy array of shape (D,), last weights.
        loss: scalar, MSE loss of the last weights on the full dataset.
    """
    raise NotImplementedError("todo: to be implemented")


def least_squares(y, tx):
    """Least squares regression using the normal equations.

    Do not use numpy.linalg.lstsq.

    Args:
        y: numpy array of shape (N,), targets.
        tx: numpy array of shape (N, D), features.

    Returns:
        w: numpy array of shape (D,), optimal weights.
        loss: scalar, MSE loss of the optimal weights.
    """
    raise NotImplementedError("todo: to be implemented")


def ridge_regression(y, tx, lambda_):
    """Ridge regression using the normal equations.

    Args:
        y: numpy array of shape (N,), targets.
        tx: numpy array of shape (N, D), features.
        lambda_: float, regularization parameter.

    Returns:
        w: numpy array of shape (D,), optimal weights.
        loss: scalar, MSE loss of the optimal weights, without the penalty term.
    """
    raise NotImplementedError("todo: to be implemented")


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """Logistic regression using gradient descent.

    Args:
        y: numpy array of shape (N,), labels in {0, 1}.
        tx: numpy array of shape (N, D), features.
        initial_w: numpy array of shape (D,), initial weights.
        max_iters: int, number of gradient steps.
        gamma: float, step size.

    Returns:
        w: numpy array of shape (D,), last weights.
        loss: scalar, mean negative log-likelihood of the last weights.
    """
    raise NotImplementedError("todo: to be implemented")


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """Regularized logistic regression using gradient descent.

    The objective adds the penalty term lambda_ * ||w||^2.

    Args:
        y: numpy array of shape (N,), labels in {0, 1}.
        tx: numpy array of shape (N, D), features.
        lambda_: float, regularization parameter.
        initial_w: numpy array of shape (D,), initial weights.
        max_iters: int, number of gradient steps.
        gamma: float, step size.

    Returns:
        w: numpy array of shape (D,), last weights.
        loss: scalar, mean negative log-likelihood of the last weights,
            without the penalty term.
    """
    raise NotImplementedError("todo: to be implemented")
