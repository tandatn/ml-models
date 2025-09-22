"""
Logistic Regression trained with Gradient Descent, Sigmoid, full-batch iteration.
"""

import numpy as np
import pandas as pd


def sigmoid(z: np.ndarray):
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))


def create_model(features: list, label: str, lr=0.01, epochs=10000):
    def fit(training_set: pd.DataFrame):
        X = training_set[features].to_numpy(dtype=float)
        y = training_set[label].to_numpy(dtype=float)
        N = len(y)

        # Normalization
        mean = X.mean(axis=0)
        std = X.std(axis=0)
        std[std == 0] = 1.0
        X = (X - mean) / std

        weights = np.zeros(X.shape[1], dtype=float)
        bias = 0.0

        for epoch in range(epochs):
            print(f"Epoch {epoch + 1}/{epochs}")

            # Gradient descent on Log Loss function (single example, single feature): -ylog(y') - (1-y)log(1-y')
            # - Derivative of weight: (y'-y)x
            # - Derivative of bias: (y'-y)
            z = X.dot(weights) + bias
            preds = sigmoid(z)
            errors = preds - y
            dW = (1 / N) * (X.T.dot(errors))
            db = np.mean(errors)
            weights -= lr * dW
            bias -= lr * db

            # Measure cost with average log loss
            z = X.dot(weights) + bias
            preds = sigmoid(z)
            eps = 1e-7
            clipped_preds = np.clip(preds, eps, 1 - eps)
            log_loss = (1 / N) * np.sum(-y * np.log(clipped_preds) - (1 - y) * np.log(1 - clipped_preds))
            print(f"Loss: {log_loss}")
            print(f"Weights: {weights}, bias: {bias}\n")

        return weights, bias, mean, std

    return fit


def train_model(model, training_set: pd.DataFrame):
    weights, bias, mean, std = model(training_set)

    def predict(age, sysBP):
        X = np.array([age, sysBP])
        X = (X - mean) / std  # Normalize features
        z = np.dot(weights, X) + bias
        return sigmoid(z)

    return predict


# Prepare data
dataset = pd.read_csv("data/framingham.csv").copy()
features = ["age", "sysBP"]
label = "TenYearCHD"
training_set = dataset[[*features, label]]


# Create and train model
model = create_model(features, label, lr=0.01, epochs=10000)
predict = train_model(model, training_set)


# Prediction
print("✨ Predictions")
print(f"age={(age := 25)}, sysBP={(sysBP := 120)} -> prob={predict(age, sysBP):.2f} (Young, normal BP)")
print(f"age={(age := 65)}, sysBP={(sysBP := 160)} -> prob={predict(age, sysBP):.2f} (Older, high BP)")
print(f"age={(age := 45)}, sysBP={(sysBP := 130)} -> prob={predict(age, sysBP):.2f} (Middle age, slightly elevated BP)")
