import numpy as np


def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
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
    w = initial_w.copy()
    N = y.shape[0]

    for _ in range(max_iters):
        for j in np.random.permutation(N):
            y_hat = np.dot(tx[j], w)
            gradient = -tx[j] * (y[j] - y_hat)
            w = w - gamma * gradient

    loss = np.sum((tx @ w - y) ** 2) / (2 * N)
    return w, loss


def least_squares(y, tx):
    raise NotImplementedError("todo: to be implemented")


def ridge_regression(y, tx, lambda_):
    N, D = tx.shape
    A = tx.T @ tx + 2 * N * lambda_ * np.eye(D)
    b = tx.T @ y

    w = np.linalg.solve(A, b)
    error = y - tx @ w
    return w, np.mean(error ** 2) / 2


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    raise NotImplementedError("todo: to be implemented")


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    def sigmoid(t):
        return 1 / (1 + np.exp(-t))

    N, _ = tx.shape
    w = initial_w
    for _ in range(max_iters):
        probs = sigmoid(tx @ w)
        gradient = (1 / N) * tx.T @ (probs-y)
        grad_reg_term = 2*lambda_*w
        w = w - gamma * (grad_reg_term + gradient)

    probs = sigmoid(tx @ w)
    loss = -np.mean(y * np.log(probs) + (1 - y) * np.log(1 - probs))

    reg_loss = loss + lambda_ * np.sum(w**2)
    return w, reg_loss