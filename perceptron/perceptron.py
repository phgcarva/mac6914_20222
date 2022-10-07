from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import numpy as np


def perceptron(X, y, max_iter=200):
    w = np.random.uniform(low=0, high=1, size=len(X[0]))

    i = 0
    while i < max_iter:
        idx = np.random.randint(low=0, high=X.shape[0])
        if y[idx] == 1 and np.dot(w, X[idx]) < 0:
            w += X[idx]
        if y[idx] == 0 and np.dot(w, X[idx]) >= 0:
            w += X[idx]
        i += 1
    print(f"Number of iterations: {i}")
    return w


def predict_perceptron(X, w):
    preds = np.array([np.dot(w.reshape(1, len(w)), x) for x in X])
    return np.array([1 if pred > 0 else 0 for pred in preds])


if __name__ == "__main__":
    X, y = make_classification(
        n_samples=2000,
        n_features=2,
        n_informative=2,
        n_redundant=0,
        n_repeated=0,
        n_classes=2,
        class_sep=100,
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
    )

    w = perceptron(X_train, y_train, max_iter=1000)

    preds = predict_perceptron(X_test, w)

    print(classification_report(y_test, preds))
