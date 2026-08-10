"""Q1: Is Smart Turn v3.2's P(complete) calibrated on Arabic clean gold?

Computes ECE (equal-mass bins), Brier, AUC, and reliability-curve points,
overall / per dialect / per split; clip-clustered bootstrap CIs; and
post-hoc recalibration (temperature + Platt) fit on train clips and
applied to the two held-out channels (global vs per-dialect fits).

Output: data/results_calibration.json
"""

import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize, minimize_scalar
from scipy.stats import rankdata

DATA = Path(__file__).resolve().parent.parent / "data"
RNG = np.random.default_rng(13)
N_BINS = 10
B_BOOT = 2000
EPS = 1e-4


def load_clean() -> list[dict]:
    rows = json.load(open(DATA / "analysis_table.json"))
    return [r for r in rows if r["refined"] is not None]


def arr(rows, key):
    return np.array([r[key] for r in rows])


def ece_equal_mass(p, y, n_bins=N_BINS):
    """ECE with equal-mass bins; returns (ece, bin table)."""
    order = np.argsort(p)
    p, y = p[order], y[order]
    bins = np.array_split(np.arange(len(p)), n_bins)
    ece, table = 0.0, []
    for idx in bins:
        conf, acc = p[idx].mean(), y[idx].mean()
        w = len(idx) / len(p)
        ece += w * abs(acc - conf)
        table.append({"conf": float(conf), "acc": float(acc), "n": int(len(idx)),
                      "lo": float(p[idx].min()), "hi": float(p[idx].max())})
    return float(ece), table


def brier(p, y):
    return float(np.mean((p - y) ** 2))


def murphy(p, y, n_bins=N_BINS):
    """Murphy decomposition of the Brier score over equal-mass bins:
    Brier = uncertainty - resolution + reliability."""
    order = np.argsort(p)
    p, y = p[order], y[order]
    base = y.mean()
    unc = base * (1 - base)
    rel = res = 0.0
    for idx in np.array_split(np.arange(len(p)), n_bins):
        w = len(idx) / len(p)
        rel += w * (p[idx].mean() - y[idx].mean()) ** 2
        res += w * (y[idx].mean() - base) ** 2
    return {"uncertainty": float(unc), "resolution": float(res),
            "reliability": float(rel)}


def auc(p, y):
    """Tie-corrected Mann-Whitney AUC."""
    r = rankdata(p)
    n1 = int(y.sum())
    n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def nll(p, y):
    p = np.clip(p, EPS, 1 - EPS)
    return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))


def fit_temperature(p, y):
    z = logit(p)

    def loss(t):
        return nll(1 / (1 + np.exp(-z / t)), y)

    res = minimize_scalar(loss, bounds=(0.05, 50.0), method="bounded")
    return float(res.x)


def fit_platt(p, y):
    z = logit(p)

    def loss(ab):
        return nll(1 / (1 + np.exp(-(ab[0] * z + ab[1]))), y)

    res = minimize(loss, x0=np.array([1.0, 0.0]), method="Nelder-Mead")
    return [float(v) for v in res.x]


def apply_temp(p, t):
    return 1 / (1 + np.exp(-logit(p) / t))


def apply_platt(p, ab):
    return 1 / (1 + np.exp(-(ab[0] * logit(p) + ab[1])))


def cluster_ci(rows, stat_fn, b=B_BOOT):
    """Clip-clustered bootstrap CI for a statistic over rows."""
    clips = sorted({r["clip"] for r in rows})
    by_clip = {c: [r for r in rows if r["clip"] == c] for c in clips}
    stats = []
    for _ in range(b):
        sample = []
        for c in RNG.choice(clips, size=len(clips), replace=True):
            sample.extend(by_clip[c])
        p = arr(sample, "prob")
        y = (arr(sample, "refined") == "shift").astype(float)
        if y.sum() in (0, len(y)):
            continue
        stats.append(stat_fn(p, y))
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return [float(lo), float(hi)]


def describe(rows, with_ci=True):
    p = arr(rows, "prob")
    y = (arr(rows, "refined") == "shift").astype(float)
    ece, table = ece_equal_mass(p, y)
    out = {
        "n": len(rows),
        "n_hold": int((y == 0).sum()),
        "ece": ece,
        "brier": brier(p, y),
        "murphy": murphy(p, y),
        "auc": auc(p, y),
        "mean_conf": float(p.mean()),
        "base_rate": float(y.mean()),
        "reliability": table,
    }
    if with_ci:
        out["ece_ci"] = cluster_ci(rows, lambda p_, y_: ece_equal_mass(p_, y_)[0])
        out["brier_ci"] = cluster_ci(rows, brier)
    return out


def main() -> None:
    clean = load_clean()
    train = [r for r in clean if r["split"] == "train"]
    test = [r for r in clean if r["split"] == "test"]

    results = {"n_bins": N_BINS, "boot": B_BOOT, "strata": {}}
    strata = {
        "all": clean,
        "egyptian": [r for r in clean if r["dialect"] == "egyptian"],
        "gulf": [r for r in clean if r["dialect"] == "gulf"],
        "train": train,
        "test": test,
        "test_egyptian": [r for r in test if r["dialect"] == "egyptian"],
        "test_gulf": [r for r in test if r["dialect"] == "gulf"],
    }
    for name, rows in strata.items():
        results["strata"][name] = describe(rows)
        s = results["strata"][name]
        print(f"{name:14s} n={s['n']:5d}  ECE={s['ece']:.3f} {s.get('ece_ci','')}"
              f"  Brier={s['brier']:.3f}  AUC={s['auc']:.3f}"
              f"  conf={s['mean_conf']:.3f} vs base={s['base_rate']:.3f}")

    # sanity: published AUCs (resource paper) 0.6803 / 0.7217 / 0.6677
    assert abs(results["strata"]["all"]["auc"] - 0.6803) < 0.002
    assert abs(results["strata"]["egyptian"]["auc"] - 0.7217) < 0.002
    assert abs(results["strata"]["gulf"]["auc"] - 0.6677) < 0.002

    # --- recalibration: fit on train, evaluate on held-out channels ---
    p_tr = arr(train, "prob")
    y_tr = (arr(train, "refined") == "shift").astype(float)
    t_global = fit_temperature(p_tr, y_tr)
    platt_global = fit_platt(p_tr, y_tr)

    recal = {"temperature_global": t_global, "platt_global": platt_global,
             "per_dialect": {}, "eval": {}}

    fits = {"global": (t_global, platt_global)}
    for d in ("egyptian", "gulf"):
        rows_d = [r for r in train if r["dialect"] == d]
        p_d = arr(rows_d, "prob")
        y_d = (arr(rows_d, "refined") == "shift").astype(float)
        t_d, ab_d = fit_temperature(p_d, y_d), fit_platt(p_d, y_d)
        recal["per_dialect"][d] = {"temperature": t_d, "platt": ab_d}
        fits[d] = (t_d, ab_d)

    for stratum, rows in [("test", test),
                          ("test_egyptian", strata["test_egyptian"]),
                          ("test_gulf", strata["test_gulf"])]:
        p_te = arr(rows, "prob")
        y_te = (arr(rows, "refined") == "shift").astype(float)
        entry = {"raw": {"ece": ece_equal_mass(p_te, y_te)[0],
                         "brier": brier(p_te, y_te), "nll": nll(p_te, y_te)}}
        for fit_name in ("global", "matched"):
            if fit_name == "global":
                t, ab = fits["global"]
            else:
                d = stratum.replace("test_", "")
                if d == "test":
                    continue
                t, ab = fits[d]
            entry[fit_name] = {
                "temp": {"ece": ece_equal_mass(apply_temp(p_te, t), y_te)[0],
                         "brier": brier(apply_temp(p_te, t), y_te),
                         "nll": nll(apply_temp(p_te, t), y_te)},
                "platt": {"ece": ece_equal_mass(apply_platt(p_te, ab), y_te)[0],
                          "brier": brier(apply_platt(p_te, ab), y_te),
                          "nll": nll(apply_platt(p_te, ab), y_te)},
            }
        recal["eval"][stratum] = entry
        print(f"recal {stratum}: raw ECE={entry['raw']['ece']:.3f} -> "
              + ", ".join(f"{k}.temp={v['temp']['ece']:.3f}/platt={v['platt']['ece']:.3f}"
                          for k, v in entry.items() if k != "raw"))

    results["recalibration"] = recal
    json.dump(results, open(DATA / "results_calibration.json", "w"), indent=1)
    print("wrote data/results_calibration.json")


if __name__ == "__main__":
    main()
