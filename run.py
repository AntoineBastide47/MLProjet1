"""Produce the best AIcrowd submission from the raw data.

Usage: python run.py
Reads data/dataset/{x_train,y_train,x_test}.csv and writes submission.csv.
"""

from helpers import create_csv_submission, load_csv_data

DATA_PATH = "data/dataset"
SUBMISSION_PATH = "submission.csv"


def preprocess(x_train, x_test):
    """Clean and transform the features of the train and test sets.

    Args:
        x_train: numpy array of shape (N_train, D), raw training features.
        x_test: numpy array of shape (N_test, D), raw test features.

    Returns:
        tx_train: numpy array of shape (N_train, D'), processed training features.
        tx_test: numpy array of shape (N_test, D'), processed test features.
    """
    raise NotImplementedError("todo: to be implemented")


def train(y, tx):
    """Train the final model.

    Args:
        y: numpy array of shape (N,), labels in {-1, 1}.
        tx: numpy array of shape (N, D), processed features.

    Returns:
        w: numpy array of shape (D,), trained weights.
    """
    raise NotImplementedError("todo: to be implemented")


def predict(tx, w):
    """Predict labels for the given features.

    Args:
        tx: numpy array of shape (N, D), processed features.
        w: numpy array of shape (D,), trained weights.

    Returns:
        y_pred: numpy array of shape (N,), predictions in {-1, 1}.
    """
    raise NotImplementedError("todo: to be implemented")


def main():
    x_train, x_test, y_train, _, test_ids = load_csv_data(DATA_PATH)
    tx_train, tx_test = preprocess(x_train, x_test)
    w = train(y_train, tx_train)
    create_csv_submission(test_ids, predict(tx_test, w), SUBMISSION_PATH)


if __name__ == "__main__":
    main()
