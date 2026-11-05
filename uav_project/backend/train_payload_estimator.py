"""Train the version-matched payload estimator and save audit metadata."""
import numpy as np
import joblib
import sklearn
from pathlib import Path
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error
from adaptive_system import ESTIMATOR_FEATURE_COUNT, ESTIMATOR_FEATURE_NAMES


def main(data_path="payload_dataset.npz", seed=0, output_path="payload_estimator.joblib"):
    data = np.load(data_path)
    X, y = data["X"], data["y"]
    groups = data["groups"] if "groups" in data.files else None
    if X.ndim != 2 or X.shape[1] != ESTIMATOR_FEATURE_COUNT:
        raise ValueError(
            f"dataset has {X.shape[1] if X.ndim == 2 else 'invalid'} features; "
            f"expected {ESTIMATOR_FEATURE_COUNT}"
        )

    # Split by flight/run, never by overlapping windows from the same run.
    if groups is not None:
        outer = GroupShuffleSplit(n_splits=1, test_size=0.36,
                                  random_state=seed)
        train_idx, temp_idx = next(outer.split(X, y, groups))
        inner = GroupShuffleSplit(n_splits=1, test_size=(0.20 / 0.36),
                                  random_state=seed)
        val_rel, test_rel = next(inner.split(X[temp_idx], y[temp_idx],
                                              groups[temp_idx]))
        val_idx, test_idx = temp_idx[val_rel], temp_idx[test_rel]
        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]
        X_test, y_test = X[test_idx], y[test_idx]
        print(f"Grouped split: {len(np.unique(groups[train_idx]))}/"
              f"{len(np.unique(groups[val_idx]))}/"
              f"{len(np.unique(groups[test_idx]))} flights")
    else:
        # Legacy datasets without run IDs: one row per run, so row split is
        # equivalent; new datasets must always include groups.
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=0.36, random_state=seed)
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=(0.20 / 0.36), random_state=seed)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    model = MLPRegressor(
        hidden_layer_sizes=(32, 16),
        activation="relu",
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=20,
        max_iter=2000,
        random_state=seed,
    )
    model.fit(X_train_s, y_train)

    train_mae = mean_absolute_error(y_train, model.predict(X_train_s))
    val_mae = mean_absolute_error(y_val, model.predict(X_val_s))
    test_mae = mean_absolute_error(y_test, model.predict(X_test_s))

    print(f"Train MAE: {train_mae:.4f}")
    print(f"Val   MAE: {val_mae:.4f}")
    print(f"Test  MAE: {test_mae:.4f}")
    print(f"Feature schema: {list(ESTIMATOR_FEATURE_NAMES)}")

    out_path = Path(output_path)
    if not out_path.is_absolute():
        out_path = Path(__file__).resolve().with_name(output_path)
    # Prevent uncontrolled neural-network extrapolation on out-of-distribution
    # flight windows. These bounds are learned from training data only.
    clip_low = np.quantile(X_train, 0.005, axis=0)
    clip_high = np.quantile(X_train, 0.995, axis=0)
    joblib.dump({"model": model, "scaler": scaler,
                 "feature_count": ESTIMATOR_FEATURE_COUNT,
                 "feature_names": ESTIMATOR_FEATURE_NAMES,
                 "sklearn_version": sklearn.__version__,
                 "training_dataset": str(Path(data_path)),
                 "clip_low": clip_low,
                 "clip_high": clip_high,
                 "target": "payload_level"}, out_path)
    print(f"Saved {out_path}")

    return {"train_mae": train_mae, "val_mae": val_mae, "test_mae": test_mae}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("data_path", nargs="?", default="payload_dataset.npz")
    parser.add_argument("--output", default="payload_estimator.joblib")
    args = parser.parse_args()
    result = main(args.data_path, output_path=args.output)
