"""
Advanced Payload Estimator Training & Multi-Metric Benchmarking.

Evaluates deep network architectures, L2 regularization, and tolerance-band
accuracy metrics (tolerance bounds +-0.05, +-0.08, +-0.10) to determine optimal
generalization performance under stochastic turbulence without overfitting.
"""

import numpy as np
import joblib
import sklearn
from pathlib import Path
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error

from adaptive_system import ESTIMATOR_FEATURE_COUNT, ESTIMATOR_FEATURE_NAMES


def train_and_benchmark(data_path="payload_dataset_adaptive.npz", output_path="payload_estimator_advanced.joblib", seed=42):
    p = Path(data_path)
    if not p.is_absolute():
        p = Path(__file__).resolve().with_name(data_path)
    if not p.exists():
        p = Path(__file__).resolve().with_name("payload_dataset.npz")
    
    print(f"Loading dataset from: {p}")
    data = np.load(p)
    X, y = data["X"], data["y"]
    groups = data["groups"] if "groups" in data.files else None

    # Grouped split: flights never leak across train/test splits
    if groups is not None:
        outer = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=seed)
        train_idx, test_idx = next(outer.split(X, y, groups))
        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]
        print(f"Dataset split: {len(np.unique(groups[train_idx]))} training flights, {len(np.unique(groups[test_idx]))} test flights.")
    else:
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=seed)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # Advanced Multi-Layer Perceptron Architectures
    candidate_models = {
        "Baseline MLP (32, 16)": MLPRegressor(
            hidden_layer_sizes=(32, 16),
            activation="relu",
            max_iter=500,
            early_stopping=True,
            random_state=seed,
        ),
        "Deep Regularized MLP (64, 32, 16)": MLPRegressor(
            hidden_layer_sizes=(64, 32, 16),
            activation="relu",
            alpha=0.001,
            max_iter=600,
            early_stopping=True,
            random_state=seed,
        ),
        "Wide Deep MLP (128, 64, 32)": MLPRegressor(
            hidden_layer_sizes=(128, 64, 32),
            activation="relu",
            alpha=0.002,
            max_iter=600,
            early_stopping=True,
            random_state=seed,
        ),
    }

    results = {}
    best_model_name = None
    best_score = -float("inf")

    print("\n" + "=" * 78)
    print("ESTIMATOR BENCHMARK REPORT (Unseen Holdout Flights)")
    print("=" * 78)

    for name, model in candidate_models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train_s, y_train)
        preds = model.predict(X_test_s)

        mae = mean_absolute_error(y_test, preds)
        rmse = root_mean_squared_error(y_test, preds)
        r2 = r2_score(y_test, preds)

        # Operational tolerance band accuracy
        acc_05 = float(np.mean(np.abs(y_test - preds) <= 0.05) * 100)
        acc_08 = float(np.mean(np.abs(y_test - preds) <= 0.08) * 100)
        acc_10 = float(np.mean(np.abs(y_test - preds) <= 0.10) * 100)

        results[name] = {
            "model": model,
            "mae": mae,
            "rmse": rmse,
            "r2": r2,
            "acc_05": acc_05,
            "acc_08": acc_08,
            "acc_10": acc_10,
        }

        print(f"  MAE (Mean Absolute Error)     : {mae:.4f}")
        print(f"  RMSE (Root Mean Squared Error): {rmse:.4f}")
        print(f"  R² Score (Variance Explained) : {r2 * 100:.2f}%")
        print(f"  Accuracy (Within ±0.05 error) : {acc_05:.1f}%")
        print(f"  Accuracy (Within ±0.08 error) : {acc_08:.1f}%")
        print(f"  Accuracy (Within ±0.10 error) : {acc_10:.1f}%")

        if r2 > best_score:
            best_score = r2
            best_model_name = name

    print("\n" + "=" * 78)
    print(f"BEST PERFORMING ARCHITECTURE: {best_model_name} (R² = {best_score * 100:.2f}%)")
    print("=" * 78)

    best_bundle = results[best_model_name]
    clip_low = np.quantile(X_train, 0.005, axis=0)
    clip_high = np.quantile(X_train, 0.995, axis=0)

    out_file = Path(output_path)
    if not out_file.is_absolute():
        out_file = Path(__file__).resolve().with_name(output_path)

    joblib.dump({
        "model": best_bundle["model"],
        "scaler": scaler,
        "feature_count": ESTIMATOR_FEATURE_COUNT,
        "feature_names": ESTIMATOR_FEATURE_NAMES,
        "sklearn_version": sklearn.__version__,
        "training_dataset": str(p),
        "clip_low": clip_low,
        "clip_high": clip_high,
        "target": "payload_level",
        "benchmark": {
            "best_model_name": best_model_name,
            "mae": best_bundle["mae"],
            "rmse": best_bundle["rmse"],
            "r2": best_bundle["r2"],
            "acc_08": best_bundle["acc_08"],
        }
    }, out_file)
    print(f"Saved advanced estimator artifact to: {out_file}\n")
    return results


if __name__ == "__main__":
    train_and_benchmark()

