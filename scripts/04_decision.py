"""Q3: Respond or wait, with a guarantee on the interruption rate.

Decision rule: commit (respond) iff P(complete) >= tau, else wait.
Risk = false-cutoff rate on holds, R(tau) = P(p >= tau | hold) - monotone
non-increasing in tau, loss bounded by 1 -> conformal risk control (CRC,
Angelopoulos et al. 2024): pick tau_hat = min{tau: (n R_hat(tau) + 1)/(n+1)
<= alpha} on a calibration set; then E[R(tau_hat)] <= alpha under
exchangeability.

Experiments:
  1. In-distribution: repeated clip-level cal/eval splits WITHIN the 26
     train clips (same channels) - does the bound hold? (point-level CRC,
     plus a clip-level CRC variant honest about clustering)
  2. Deployment shift: calibrate on all train clips, evaluate on the two
     held-out channels (Egyptian Ch-E, Gulf Ch-G) - pooled and Mondrian
     (per-dialect) thresholds.
  3. Cost frame: expected cost at interruption:deferral cost ratios
     c in {2, 5, 10} for default tau=0.5, re-thresholded 0.74, train-
     optimal tau*_c, and CRC tau_hat(alpha).

Output: data/results_decision.json
"""

import json
from pathlib import Path

import numpy as np

DATA = Path(__file__).resolve().parent.parent / "data"
RNG = np.random.default_rng(13)
ALPHAS = [0.05, 0.10, 0.15, 0.20]
COSTS = [2, 5, 10]
N_SPLITS = 200  # repeated in-distribution clip splits


def arr(rows, key):
    return np.array([r[key] for r in rows])


def holds_shifts(rows):
    h = np.array([r["prob"] for r in rows if r["refined"] == "hold"])
    s = np.array([r["prob"] for r in rows if r["refined"] == "shift"])
    return h, s


def fc_rate(h, tau):
    return float(np.mean(h >= tau)) if len(h) else float("nan")


def missed_rate(s, tau):
    return float(np.mean(s < tau)) if len(s) else float("nan")


def crc_tau(h_cal, alpha):
    """Smallest tau on a fine grid with (n*R_hat(tau)+1)/(n+1) <= alpha."""
    n = len(h_cal)
    grid = np.unique(np.concatenate([h_cal, [0.0, 1.0]]))
    for tau in grid:
        if (n * fc_rate(h_cal, tau) + 1) / (n + 1) <= alpha:
            return float(tau)
    return 1.0


def crc_tau_cliplevel(rows_cal, alpha):
    """Clip-level CRC: loss per clip = that clip's hold false-cutoff rate.

    Clips are the exchangeable unit; bound is on the expected per-clip risk.
    """
    clips = {}
    for r in rows_cal:
        if r["refined"] == "hold":
            clips.setdefault(r["clip"], []).append(r["prob"])
    probs = {c: np.array(v) for c, v in clips.items()}
    n = len(probs)
    grid = np.unique(np.concatenate([np.concatenate(list(probs.values())), [0.0, 1.0]]))
    for tau in grid:
        risk = np.mean([fc_rate(v, tau) for v in probs.values()])
        if (n * risk + 1) / (n + 1) <= alpha:
            return float(tau)
    return 1.0


def macro_fc(rows, tau):
    """Per-clip mean false-cutoff rate (matches clip-level CRC's risk)."""
    clips = {}
    for r in rows:
        if r["refined"] == "hold":
            clips.setdefault(r["clip"], []).append(r["prob"])
    return float(np.mean([fc_rate(np.array(v), tau) for v in clips.values()]))


def expected_cost(h, s, tau, c):
    """Per-point expected cost: interruptions cost c, deferred shifts cost 1."""
    n = len(h) + len(s)
    return float((c * np.sum(h >= tau) + np.sum(s < tau)) / n)


def main() -> None:
    rows = json.load(open(DATA / "analysis_table.json"))
    clean = [r for r in rows if r["refined"] is not None]
    train = [r for r in clean if r["split"] == "train"]
    test = [r for r in clean if r["split"] == "test"]

    h_tr, s_tr = holds_shifts(train)
    h_te, s_te = holds_shifts(test)
    by_dialect = {d: [r for r in test if r["dialect"] == d]
                  for d in ("egyptian", "gulf")}
    tr_dialect = {d: [r for r in train if r["dialect"] == d]
                  for d in ("egyptian", "gulf")}
    print(f"train holds={len(h_tr)} shifts={len(s_tr)}; "
          f"test holds={len(h_te)} shifts={len(s_te)}")

    results = {"n": {"train_holds": len(h_tr), "train_shifts": len(s_tr),
                     "test_holds": len(h_te), "test_shifts": len(s_te)}}

    # ---- 1. in-distribution: repeated clip splits within train ----
    train_clips = sorted({r["clip"] for r in train})
    indist = {f"{a:.2f}": {"realized": [], "realized_macro": [],
                           "tau": [], "tau_clip": [], "realized_clip_macro": []}
              for a in ALPHAS}
    for _ in range(N_SPLITS):
        perm = RNG.permutation(train_clips)
        cal_clips = set(perm[: int(0.7 * len(perm))])
        cal = [r for r in train if r["clip"] in cal_clips]
        ev = [r for r in train if r["clip"] not in cal_clips]
        h_cal, _ = holds_shifts(cal)
        h_ev, _ = holds_shifts(ev)
        if len(h_ev) < 30 or len(h_cal) < 100:
            continue
        for a in ALPHAS:
            key = f"{a:.2f}"
            tau = crc_tau(h_cal, a)
            indist[key]["tau"].append(tau)
            indist[key]["realized"].append(fc_rate(h_ev, tau))
            indist[key]["realized_macro"].append(macro_fc(ev, tau))
            tau_c = crc_tau_cliplevel(cal, a)
            indist[key]["tau_clip"].append(tau_c)
            indist[key]["realized_clip_macro"].append(macro_fc(ev, tau_c))
    results["in_distribution"] = {}
    for a in ALPHAS:
        key = f"{a:.2f}"
        d = indist[key]
        results["in_distribution"][key] = {
            "n_splits": len(d["realized"]),
            "mean_realized": float(np.mean(d["realized"])),
            "p90_realized": float(np.percentile(d["realized"], 90)),
            "violation_rate": float(np.mean(np.array(d["realized"]) > a)),
            "mean_tau": float(np.mean(d["tau"])),
            "clip_level": {
                "mean_realized_macro": float(np.mean(d["realized_clip_macro"])),
                "violation_rate_macro": float(
                    np.mean(np.array(d["realized_clip_macro"]) > a)),
                "mean_tau": float(np.mean(d["tau_clip"])),
            },
        }
        r = results["in_distribution"][key]
        print(f"in-dist a={key}: realized={r['mean_realized']:.3f} "
              f"(viol {r['violation_rate']:.2f}, tau={r['mean_tau']:.2f}) | "
              f"clip-CRC macro={r['clip_level']['mean_realized_macro']:.3f} "
              f"(viol {r['clip_level']['violation_rate_macro']:.2f})")

    # ---- 2. deployment shift: train -> unseen channels ----
    results["shift"] = {}
    for a in ALPHAS:
        key = f"{a:.2f}"
        tau_pool = crc_tau(h_tr, a)
        entry = {"tau_pooled": tau_pool,
                 "test_fc": fc_rate(h_te, tau_pool),
                 "test_missed": missed_rate(s_te, tau_pool),
                 "per_dialect": {}}
        for d, rows_d in by_dialect.items():
            h_d, s_d = holds_shifts(rows_d)
            h_cal_d, _ = holds_shifts(tr_dialect[d])
            tau_d = crc_tau(h_cal_d, a)
            entry["per_dialect"][d] = {
                "fc_pooled_tau": fc_rate(h_d, tau_pool),
                "missed_pooled_tau": missed_rate(s_d, tau_pool),
                "tau_mondrian": tau_d,
                "fc_mondrian": fc_rate(h_d, tau_d),
                "missed_mondrian": missed_rate(s_d, tau_d),
            }
        results["shift"][key] = entry
        e, g = entry["per_dialect"]["egyptian"], entry["per_dialect"]["gulf"]
        print(f"shift a={key}: tau={tau_pool:.2f} test fc={entry['test_fc']:.3f} "
              f"(EGY {e['fc_pooled_tau']:.3f}, GULF {g['fc_pooled_tau']:.3f}) "
              f"missed={entry['test_missed']:.3f} | mondrian fc "
              f"EGY {e['fc_mondrian']:.3f}@{e['tau_mondrian']:.2f} "
              f"GULF {g['fc_mondrian']:.3f}@{g['tau_mondrian']:.2f}")

    # ---- 3. cost frame on the held-out test ----
    # Bayes rule for cost ratio c (commit costs c on a hold, defer costs 1 on
    # a shift): commit iff P(complete) > c/(c+1). Correct only if P is the
    # true posterior -> calibration is decision-relevant, not cosmetic.
    calres = json.load(open(DATA / "results_calibration.json"))
    t_global = calres["recalibration"]["temperature_global"]

    def temp_scale(p, t=t_global):
        z = np.log(np.clip(p, 1e-4, 1 - 1e-4) / (1 - np.clip(p, 1e-4, 1 - 1e-4)))
        return 1 / (1 + np.exp(-z / t))

    h_te_cal, s_te_cal = temp_scale(h_te), temp_scale(s_te)
    results["cost"] = {"temperature_global": t_global}
    grid = np.unique(np.concatenate([h_te, s_te, [0.0, 1.0]]))
    for c in COSTS:
        tau_b = c / (c + 1)
        entry = {
            "default_0.5": {"tau": 0.5, "cost": expected_cost(h_te, s_te, 0.5, c),
                            "fc": fc_rate(h_te, 0.5), "missed": missed_rate(s_te, 0.5)},
            "rethresh_0.74": {"tau": 0.74, "cost": expected_cost(h_te, s_te, 0.74, c),
                              "fc": fc_rate(h_te, 0.74), "missed": missed_rate(s_te, 0.74)},
            "bayes_raw": {"tau": tau_b, "cost": expected_cost(h_te, s_te, tau_b, c),
                          "fc": fc_rate(h_te, tau_b), "missed": missed_rate(s_te, tau_b)},
            "bayes_calibrated": {"tau": tau_b,
                                 "cost": expected_cost(h_te_cal, s_te_cal, tau_b, c),
                                 "fc": fc_rate(h_te_cal, tau_b),
                                 "missed": missed_rate(s_te_cal, tau_b)},
            "oracle_test": {},
        }
        cost_te = [expected_cost(h_te, s_te, t, c) for t in grid]
        tau_or = float(grid[int(np.argmin(cost_te))])
        entry["oracle_test"] = {"tau": tau_or,
                                "cost": expected_cost(h_te, s_te, tau_or, c),
                                "fc": fc_rate(h_te, tau_or),
                                "missed": missed_rate(s_te, tau_or)}
        results["cost"][f"c{c}"] = entry
        print(f"cost c={c}: " + " | ".join(
            f"{k}: cost={v['cost']:.3f} fc={v['fc']:.3f} miss={v['missed']:.3f}"
            for k, v in entry.items() if isinstance(v, dict)))

    # ---- risk-coverage curve data for figures (test, per dialect) ----
    curves = {}
    for name, rows_ in [("test", test)] + list(by_dialect.items()):
        h, s = holds_shifts(rows_)
        taus = np.unique(np.concatenate([h, s, [0.0, 1.0]]))
        curves[name] = [{"tau": float(t), "fc": fc_rate(h, t),
                         "missed": missed_rate(s, t)} for t in taus]
    results["curves"] = curves

    json.dump(results, open(DATA / "results_decision.json", "w"), indent=1)
    print("wrote data/results_decision.json")


if __name__ == "__main__":
    main()
