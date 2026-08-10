"""Q2: Is the residual uncertainty aleatoric? Judge label variation as signal.

(a) Calibration against the judge-vote distribution instead of hard labels
    (Baan et al. 2022): soft target s = share of the two audio judges voting
    "complete" (0 / 0.5 / 1). Does apparent miscalibration shrink?
(b) ECE stratified by judge agreement: F1 core-agree vs F2/F3 judge-split
    clean gold. Is the detector's miscalibration concentrated on the
    genuinely ambiguous points?
(c) Instance level: does detector uncertainty u = 1 - |2p-1| predict judge
    disagreement / human escalation? AUROC + Spearman with pooled-judge
    uncertainty.

Output: data/results_softlabels.json
"""

import json
from pathlib import Path

import numpy as np
from scipy.stats import rankdata, spearmanr

DATA = Path(__file__).resolve().parent.parent / "data"
N_BINS = 10
RNG = np.random.default_rng(13)
B_BOOT = 2000


def arr(rows, key):
    return np.array([r[key] for r in rows])


def ece_vs_target(p, t, n_bins=N_BINS):
    """Equal-mass-binned ECE of scores p against (possibly soft) target t."""
    order = np.argsort(p)
    p, t = p[order], t[order]
    ece = 0.0
    for idx in np.array_split(np.arange(len(p)), n_bins):
        ece += len(idx) / len(p) * abs(t[idx].mean() - p[idx].mean())
    return float(ece)


def auroc(score, label) -> float:
    r = rankdata(score)
    n1 = int(label.sum())
    n0 = len(label) - n1
    return float((r[label == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def cluster_ci_diff(rows_a, rows_b, stat_fn, b=B_BOOT):
    """Clip-clustered bootstrap CI for stat(rows_a) - stat(rows_b)."""
    clips = sorted({r["clip"] for r in rows_a} | {r["clip"] for r in rows_b})
    a_by = {c: [r for r in rows_a if r["clip"] == c] for c in clips}
    b_by = {c: [r for r in rows_b if r["clip"] == c] for c in clips}
    diffs = []
    for _ in range(b):
        sa, sb = [], []
        for c in RNG.choice(clips, size=len(clips), replace=True):
            sa.extend(a_by[c])
            sb.extend(b_by[c])
        if len(sa) < 50 or len(sb) < 50:
            continue
        diffs.append(stat_fn(sa) - stat_fn(sb))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return [float(lo), float(hi)]


def hard_ece(rows) -> float:
    p = arr(rows, "prob")
    y = (arr(rows, "refined") == "shift").astype(float)
    return ece_vs_target(p, y)


def main() -> None:
    rows = json.load(open(DATA / "analysis_table.json"))
    clean = [r for r in rows if r["refined"] is not None]
    dual = [r for r in clean if r["vote_35"] and r["vote_31"]]

    # (a) soft vs hard target on the same dual-judged clean-gold points
    p = arr(dual, "prob")
    y_hard = (arr(dual, "refined") == "shift").astype(float)
    s_votes = np.array([((r["vote_35"] == "complete") + (r["vote_31"] == "complete")) / 2
                        for r in dual])
    s_teacher = np.array([(r["tp_35"] + r["tp_31"]) / 2 for r in dual])

    res_a = {
        "n_dual_clean": len(dual),
        "ece_hard": ece_vs_target(p, y_hard),
        "ece_soft_votes": ece_vs_target(p, s_votes),
        "ece_soft_teacher_p": ece_vs_target(p, s_teacher),
        "brier_hard": float(np.mean((p - y_hard) ** 2)),
        "brier_soft_votes": float(np.mean((p - s_votes) ** 2)),
        "brier_soft_teacher_p": float(np.mean((p - s_teacher) ** 2)),
    }
    print("(a) targets:", {k: round(v, 4) if isinstance(v, float) else v
                           for k, v in res_a.items()})

    # (b) ECE stratified by judge agreement within clean gold
    f1 = [r for r in clean if r["frame"] == "F1_core_agree"]
    split = [r for r in clean if r["frame"] in ("F2_soft_consensus", "F3_human_clean")]
    res_b = {
        "n_agree": len(f1), "n_split": len(split),
        "ece_agree": hard_ece(f1), "ece_split": hard_ece(split),
        "auc_agree": auroc(arr(f1, "prob"), (arr(f1, "refined") == "shift").astype(float)),
        "auc_split": auroc(arr(split, "prob"), (arr(split, "refined") == "shift").astype(float)),
        "ece_diff_ci": cluster_ci_diff(split, f1, hard_ece),
    }
    print("(b) agree vs split:", {k: (round(v, 4) if isinstance(v, float) else v)
                                  for k, v in res_b.items()})

    # (c) does detector uncertainty know where the ambiguity is?
    u = 1 - np.abs(2 * arr(dual, "prob") - 1)
    disagree = np.array([r["vote_35"] != r["vote_31"] for r in dual]).astype(float)
    pooled_u = 1 - np.abs(2 * arr(dual, "p_complete") - 1)
    rho, rho_p = spearmanr(u, pooled_u)

    # over ALL dual-judged points (any cell): does u predict judge split /
    # human escalation?
    dual_all = [r for r in rows if r["vote_35"] and r["vote_31"]]
    u_all = 1 - np.abs(2 * arr(dual_all, "prob") - 1)
    dis_all = np.array([r["vote_35"] != r["vote_31"] for r in dual_all]).astype(float)
    esc_all = np.array([bool(r["human"]) or r["frame"] == "F6_contested"
                        for r in dual_all]).astype(float)

    res_c = {
        "auroc_u_predicts_split_clean": auroc(u, disagree),
        "auroc_u_predicts_split_all": auroc(u_all, dis_all),
        "auroc_u_predicts_escalation_all": auroc(u_all, esc_all),
        "spearman_u_vs_pooledjudge_u_clean": [float(rho), float(rho_p)],
        "n_dual_all": len(dual_all),
    }
    print("(c) uncertainty-awareness:", {k: (round(v, 4) if isinstance(v, float) else v)
                                         for k, v in res_c.items()})

    # judge-ensemble color: disagreement rate by cell (recomputed)
    cells = {}
    for r in rows:
        if r["vote_35"] and r["vote_31"]:
            c = cells.setdefault(r["cell"], [0, 0])
            c[0] += 1
            c[1] += int(r["vote_35"] != r["vote_31"])
    res_d = {cell: {"n": n, "disagree_rate": d / n} for cell, (n, d) in sorted(cells.items())}
    print("(d) disagreement by cell:", {k: round(v["disagree_rate"], 3) for k, v in res_d.items()})

    json.dump({"a_targets": res_a, "b_strata": res_b, "c_instance": res_c,
               "d_by_cell": res_d},
              open(DATA / "results_softlabels.json", "w"), indent=1)
    print("wrote data/results_softlabels.json")


if __name__ == "__main__":
    main()
