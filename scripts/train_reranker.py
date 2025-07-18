import logging
import os
import pickle

import lightgbm as lgb
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


def generate_dummy_data(num_samples=1000):
    """Generates dummy features and target labels for reranker training."""
    X = np.random.rand(num_samples, 2)

    y = 0.6 * X[:, 0] + 0.4 * X[:, 1] + np.random.rand(num_samples) * 0.2
    y = np.clip(y, 0, 1)

    return X, y


def train_reranker_model():
    logging.info("Generating dummy training data for reranker...")
    X, y = generate_dummy_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    logging.info(f"Training data shape: {X_train.shape}, {y_train.shape}")
    logging.info(f"Test data shape: {X_test.shape}, {y_test.shape}")

    lgb_model = lgb.LGBMRegressor(
        objective="regression_l1",
        metric="rmse",
        n_estimators=100,
        learning_rate=0.1,
        num_leaves=31,
        random_state=42,
    )

    logging.info("Training LightGBM reranker model...")
    lgb_model.fit(X_train, y_train)
    logging.info("Model training complete.")

    y_pred = lgb_model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    logging.info(f"Test RMSE: {rmse:.4f}")

    model_dir = os.path.join(os.path.dirname(__file__), "..", "models")
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "lightgbm_reranker.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(lgb_model, f)
    logging.info(f"LightGBM reranker model saved to {model_path}")


if __name__ == "__main__":
    train_reranker_model()
