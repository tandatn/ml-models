"""
Gradient Descent model with multiple features using MSE loss, mini-batch SGD.
"""

import numpy as np
import pandas as pd


def create_model(features: list, label: str, lr=0.0001, batch_size=30, epochs=20):

    def fit(dataset: pd.DataFrame):
        N = len(dataset)
        X = dataset[features].to_numpy()
        y = dataset[label].to_numpy()

        # Parameters
        weights = np.zeros(2, dtype=float)
        bias = 0.0

        for epoch in range(epochs):
            print(f"Epoch {epoch + 1}/{epochs}")

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
                weights -= lr * dW
                bias -= lr * db

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
training_set = dataset.rename(columns={"TRIP_SECONDS": "TRIP_MINUTES"})
features = ["TRIP_MINUTES", "TRIP_MILES"]
label = "FARE"


# Create and train model
model = create_model(features, label, lr=0.0001, batch_size=30, epochs=20)
predict = train_model(model, training_set)


# Try prediction
print("✨ Predictions")
print(f"TRIP_MINUTES={(tmin := 45)}, TRIP_MILES={(tmil := 2)} -> FARE={predict(tmin, tmil):.2f}")
print(f"TRIP_MINUTES={(tmin := 30)}, TRIP_MILES={(tmil := 1)} -> FARE={predict(tmin, tmil):.2f}")
