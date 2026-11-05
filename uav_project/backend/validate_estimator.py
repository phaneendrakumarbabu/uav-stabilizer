"""Leakage-aware validation of the payload estimator (offline, on windows).
1) 5-fold GroupKFold by flight: constant / ridge / MLP baselines (mean +- SD over folds)
2) Held-out CONDITIONS: unseen airspeed band, gust band, payload band
3) Negative control: shuffled labels must give no skill (rules out leakage)
Writes nn_validation_report.json"""
import json, warnings
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error as mae, r2_score
warnings.filterwarnings("ignore")
d = np.load("payload_dataset_adaptive.npz")
X, y, g = d["X"], d["y"], d["groups"]
rp, rg, rv, trel = d["run_payload"], d["run_gust"], d["run_va"], d["t_rel"]

def fit_predict(kind, Xtr, ytr, Xte, seed=0):
    sc = StandardScaler().fit(Xtr); A, B = sc.transform(Xtr), sc.transform(Xte)
    if kind == "constant": return np.full(len(Xte), ytr.mean())
    if kind == "ridge": return np.clip(Ridge(alpha=1.0).fit(A, ytr).predict(B), 0, 1)
    m = MLPRegressor(hidden_layer_sizes=(32, 16), early_stopping=True, validation_fraction=0.15, n_iter_no_change=20,
                     max_iter=2000, random_state=seed).fit(A, ytr)
    return np.clip(m.predict(B), 0, 1)

def scores(yt, p): return dict(MAE=float(mae(yt, p)), RMSE=float(np.sqrt(np.mean((yt - p) ** 2))), R2=float(r2_score(yt, p)))
rep = {"n_windows": int(len(y)), "n_flights": int(len(np.unique(g)))}
rep["cv"] = {}
for kind in ("constant", "ridge", "mlp"):
    f = [scores(y[te], fit_predict(kind, X[tr], y[tr], X[te])) for tr, te in GroupKFold(5).split(X, y, g)]
    rep["cv"][kind] = {k: [float(np.mean([a[k] for a in f])), float(np.std([a[k] for a in f], ddof=1))] for k in f[0]}
# false alarms / detection from the CV-held-out MLP predictions
pred = np.zeros(len(y))
for tr, te in GroupKFold(5).split(X, y, g): pred[te] = fit_predict("mlp", X[tr], y[tr], X[te])
neg = (y == 0) & (rp == 0)                     # flights that never change payload
rep["false_alarm_rate_pred>0.10_on_zero_payload_flights"] = float(np.mean(pred[neg] > 0.10)) if neg.any() else None
post = (rp > 0) & (trel >= 3.0) & (y > 0)
rep["mae_by_true_payload_band_after_3s"] = {f"{lo}-{hi}": float(mae(y[post & (y >= lo) & (y < hi)], pred[post & (y >= lo) & (y < hi)]))
    for lo, hi in ((0.0, .3), (.3, .55), (.55, .75)) if (post & (y >= lo) & (y < hi)).any()}
rep["heldout_conditions"] = {}
for name, test in (("airspeed>17 m/s (trained on <=17)", rv > 17), ("gust>1.2 (trained on <=1.2)", rg > 1.2),
                   ("payload in [0.35,0.55] (never seen)", (rp >= .35) & (rp <= .55))):
    tr, te = np.where(~test)[0], np.where(test)[0]
    if len(te) and len(tr):
        rep["heldout_conditions"][name] = {k: {"n_test": int(len(te)), **scores(y[te], fit_predict(k, X[tr], y[tr], X[te]))} for k in ("constant", "ridge", "mlp")}
rng = np.random.default_rng(0); ys = y.copy()
for fl in np.unique(g): ys[g == fl] = rng.permutation(ys[g == fl])
f = [scores(y[te], fit_predict("mlp", X[tr], ys[tr], X[te])) for tr, te in GroupKFold(5).split(X, y, g)]
rep["negative_control_shuffled_labels_mlp_MAE"] = float(np.mean([a["MAE"] for a in f]))
json.dump(rep, open("nn_validation_report.json", "w"), indent=2)
print(json.dumps(rep, indent=1))
