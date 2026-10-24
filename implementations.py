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
    w = np.linalg.lstsq(tx, y, rcond=None)[0]
    loss = np.mean((y - tx @ w) ** 2) / 2
    return w, loss


def ridge_regression(y, tx, lambda_):
    N, D = tx.shape
    A = tx.T @ tx + 2 * N * lambda_ * np.eye(D)
    b = tx.T @ y

    w = np.linalg.solve(A, b)
    error = y - tx @ w
    return w, np.mean(error ** 2) / 2


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    def sigmoid(t):
        return 1 / (1 + np.exp(-t))
    
    N, _ = tx.shape
    w = initial_w
    for _ in range(max_iters):
        probs = sigmoid(tx @ w)
        gradient = (1 / N) * tx.T @ (probs-y)
        w = w - gamma * gradient

    probs = sigmoid(tx @ w)
    loss = -np.mean(y * np.log(probs) + (1 - y) * np.log(1 - probs))

    return w, loss


def reg_logistic_regression(
    y,
    tx,
    lambda_,
    initial_w,
    max_iters,
    gamma,
    early_stop=False,
    class_weight=None,
    return_info=False,
):
    def sigmoid(t):
        t = np.clip(t, -40.0, 40.0)
        return 1.0 / (1.0 + np.exp(-t))

    N, _ = tx.shape
    w = initial_w.copy()

    if class_weight is None:
        sample_weights = np.ones(N)
    else:
        sample_weights = np.where(
            y == 1,
            class_weight[1],
            class_weight[0],
        )

    weight_sum = np.sum(sample_weights)
    regularization_mask = np.ones(w.shape[0])
    # This project represents the intercept as the first design-matrix column.
    regularization_mask[0] = 0.0
    loss_history = []
    converged = False
    gradient_norm = np.inf

    for iteration in range(max_iters):
        probs = sigmoid(tx @ w)

        gradient = (
            tx.T @ (sample_weights * (probs - y))
        ) / weight_sum

        grad_reg_term = 2 * lambda_ * w * regularization_mask
        total_gradient = gradient + grad_reg_term
        gradient_norm = np.linalg.norm(total_gradient)

        if iteration == 0 or (iteration + 1) % 50 == 0:
            per_example_loss = (
                -y * np.log(probs + 1e-12)
                -(1 - y) * np.log(1 - probs + 1e-12)
            )
            loss_history.append(float(np.sum(sample_weights * per_example_loss) / weight_sum))

        if early_stop and gradient_norm < 1e-5:
            converged = True
            break

        w = w - gamma * total_gradient

    probs = sigmoid(tx @ w)
    per_example_loss = (
        -y * np.log(probs + 1e-12)
        -(1 - y) * np.log(1 - probs + 1e-12)
    )
    loss = np.sum(sample_weights * per_example_loss) / weight_sum
    reg_loss = loss + lambda_ * np.sum((w * regularization_mask) ** 2)

    loss_history.append(float(loss))

    if return_info:
        info = {
            "iterations": iteration + 1,
            "converged": converged,
            "gradient_norm": float(gradient_norm),
            "loss_history": loss_history,
        }
        return w, reg_loss, info

    return w, reg_loss
