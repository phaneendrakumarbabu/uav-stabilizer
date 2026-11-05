"""Closed-loop estimator evaluation on seeds disjoint from training (seeds >= 100000).
Reports final-estimate error, detection rate, false alarms and invalid runs."""
import json, warnings, itertools
import numpy as np
from final_comparison import run_intelligent
warnings.filterwarnings("ignore")
rows = []
for pl, g, va in itertools.product((0.0, 0.2, 0.5, 0.7), (0.0, 1.0, 2.0), (12.0, 16.0, 20.0)):
    for k in range(5):
        _, _, est, st = run_intelligent(pl, va, g, 100000 + k, T=20.0, change_t=5.0, debug=False)
        post = [v for t, v in est if t >= 12.0]
        fin = float(np.mean(post[-3:])) if post else np.nan
        pre = [v for t, v in est if t < 5.0]
        rows.append(dict(payload=pl, gust=g, va=va, seed=100000 + k, valid=bool(st["valid"]), final=fin,
                         false_alarm_pre_change=bool(pre and max(pre) > 0.10)))
json.dump(rows, open("estimator_closed_loop.json", "w"))
v = [r for r in rows if r["valid"] and np.isfinite(r["final"])]
print("flights", len(rows), "invalid", len(rows) - len(v))
for pl in (0.0, 0.2, 0.5, 0.7):
    s = [r for r in v if r["payload"] == pl]
    print(f"payload {pl}: MAE {np.mean([abs(r['final']-pl) for r in s]):.3f}  mean estimate {np.mean([r['final'] for r in s]):.3f}  pre-change false alarm {np.mean([r['false_alarm_pre_change'] for r in s]):.2f}")
