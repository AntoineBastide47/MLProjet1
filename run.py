"""Produce the best AIcrowd submission from the raw data.

Usage: python run.py
Reads data/dataset/{x_train,y_train,x_test}.csv and writes submission.csv.
"""

from helpers import create_csv_submission, load_csv_data

DATA_PATH = "data/dataset"
SUBMISSION_PATH = "submission.csv"


def preprocess(x_train, x_test):
    raise NotImplementedError("todo: to be implemented")


def train(y, tx):
    raise NotImplementedError("todo: to be implemented")


def predict(tx, w):
    raise NotImplementedError("todo: to be implemented")


def main():
    x_train, x_test, y_train, _, test_ids = load_csv_data(DATA_PATH)
    tx_train, tx_test = preprocess(x_train, x_test)
    w = train(y_train, tx_train)
    create_csv_submission(test_ids, predict(tx_test, w), SUBMISSION_PATH)


if __name__ == "__main__":
    main()
