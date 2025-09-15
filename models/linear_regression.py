"""
Gradient Descent model with multiple features using MSE loss, mini-batch SGD.
"""

from typing import NamedTuple

import numpy as np
import pandas as pd


class Hyperparameters(NamedTuple):
    learning_rate: float
    batch_size: int
    n_epochs: int


def create_model(hyperparameters: Hyperparameters):
    learning_rate, batch_size, n_epochs = hyperparameters

    def fit(dataset: pd.DataFrame):
        N = len(dataset)
        X = dataset[["TRIP_MINUTES", "TRIP_MILES"]].to_numpy()
        y = dataset["FARE"].to_numpy()

        # Parameters
        weights = np.zeros(2, dtype=float)
        bias = 0.0

        for epoch in range(n_epochs):
            print(f"Epoch {epoch + 1}/{n_epochs}")

            # Shuffle data each epoch
            perm = np.random.permutation(np.arange(N))
            X, y = X[perm], y[perm]

            # Mini-batch updates
            for i in range(0, N, batch_size):
                X_batch, y_batch = X[i : i + batch_size], y[i : i + batch_size]
                n = len(y_batch)

                # Calc weight and bias gradients (MSE)
                preds = X_batch.dot(weights) + bias
                errors = preds - y_batch
                dW = (2 / n) * X_batch.T.dot(errors)
                db = (2 / n) * np.sum(errors)

                # Gradient descent step
                weights -= learning_rate * dW
                bias -= learning_rate * db

            # Calc MSE loss with the new parameters
            mse = np.sum(((X.dot(weights) + bias) - y) ** 2) / N
            print(f"Loss {mse}\n")

        return weights, bias

    return fit


def train_model(model, dataset: pd.DataFrame):
    weights, bias = model(dataset)
    return lambda trip_minutes, trip_miles: trip_minutes * weights[0] + trip_miles * weights[1] + bias


# Prepare data
dataset = pd.read_csv("data/chicago_taxi.csv", usecols=["TRIP_SECONDS", "TRIP_MILES", "FARE"]).copy()
dataset["TRIP_SECONDS"] = round(dataset["TRIP_SECONDS"] / 60, 2)
dataset = dataset.rename(columns={"TRIP_SECONDS": "TRIP_MINUTES"})


# Create the model with hyperparameters passed in
hyperparameters = Hyperparameters(learning_rate=0.0001, batch_size=30, n_epochs=20)
model = create_model(hyperparameters)

# Train the model with dataset
predict = train_model(model, dataset)

# Try prediction
print("✨ Predictions")
print(f"TRIP_MINUTES={45}, TRIP_MILES={2} -> Predicted FARE={predict(45, 2)}")
print(f"TRIP_MINUTES={30}, TRIP_MILES={1} -> Predicted FARE={predict(30, 1)}")
