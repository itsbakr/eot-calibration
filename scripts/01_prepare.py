"""Build the single joined analysis table for all downstream scripts.

One row per turn-point (8,581): Smart Turn v3.2 P(complete) joined to the
frozen corpus's judge votes / probabilities / human labels on (clip, at_ms),
plus split membership, clean-gold status, and the F1-F7 provenance frame
from the annotation study.

Output: data/analysis_table.json  (list of dicts)
"""

import json
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data"

# Test clips come from the frozen splits_v4 manifests via probs_test rows'
# clip ids (7 clips, the two held-out channels, one per dialect).


def refined_gold(tp: dict) -> str | None:
    """Clean-gold rule (metrics.refined_gold): diar x judge-consensus cross.

    hold & mid_thought -> HOLD ; shift & clean_shift -> SHIFT ; everything
    else only counts if human-adjudicated (label follows the human).
    """
    cell = tp["eou"].get("cell")
    human = tp["eou"].get("human")
    diar = tp["eou"].get("diar")
    if cell == "mid_thought":
        return "hold"
    if cell == "clean_shift":
        return "shift"
    if human:  # human adjudication rescues non-clean cells
        if diar == "hold" and human == "incomplete":
            return "hold"
        if diar == "shift" and human == "complete":
            return "shift"
    return None


def frame_of(tp: dict) -> str:
    """F1-F7 partition from paper/ANNOTATION_STUDY.md."""
    eou = tp["eou"]
    votes = eou.get("votes", {})
    pair = {j: v for j, v in votes.items() if j in ("3.5-flash", "3.1-pro")}
    cell, human = eou.get("cell"), eou.get("human")
    clean = cell in ("mid_thought", "clean_shift")
    if cell == "contested":
        return "F6_contested"
    if human:
        return "F3_human_clean" if clean else "F4_human_nonclean"
    if cell == "interruption":
        return "F5_interruption"
    if cell == "continuation":
        return "F7_continuation"
    if len(pair) < 2:
        return "single_judged"
    agree = len(set(pair.values())) == 1
    return "F1_core_agree" if agree else "F2_soft_consensus"


def main() -> None:
    corpus = json.load(open(DATA / "corpus_frozen.json"))
    fullset = json.load(open(DATA / "probs_fullset.json"))["rows"]
    test_clips = {r["clip"] for r in json.load(open(DATA / "probs_test.json"))["rows"]}

    probs = {(r["clip"], r["at_ms"]): r for r in fullset}
    assert len(probs) == 8581, "duplicate (clip, at_ms) keys"

    rows = []
    for clip in corpus["clips"]:
        for tp in clip["turn_points"]:
            key = (clip["id"], tp["at_ms"])
            pr = probs[key]
            eou = tp["eou"]
            assert pr["gold"] == tp["gold"], key
            tp_row = {
                "clip": clip["id"],
                "channel": clip.get("attribution"),
                "dialect": clip["dialect"],
                "at_ms": tp["at_ms"],
                "split": "test" if clip["id"] in test_clips else "train",
                "cell": eou.get("cell"),
                "diar": tp["gold"],  # hold / shift (diarization)
                "prob": pr["prob"],  # Smart Turn v3.2 P(complete)
                "p_complete": eou.get("p_complete"),  # pooled judge prob
                "vote_35": eou.get("votes", {}).get("3.5-flash"),
                "vote_31": eou.get("votes", {}).get("3.1-pro"),
                "tp_35": eou.get("teacher_p", {}).get("3.5-flash"),
                "tp_31": eou.get("teacher_p", {}).get("3.1-pro"),
                "human": eou.get("human"),
                "gap_ms": tp.get("gap_ms"),
                "frame": frame_of(tp),
                "refined": refined_gold(tp),
            }
            rows.append(tp_row)

    assert len(rows) == 8581, len(rows)

    # --- cross-checks against every count the two papers rely on ---
    def n(pred) -> int:
        return sum(1 for r in rows if pred(r))

    checks = {
        "clean_gold": (n(lambda r: r["refined"] is not None), 4610),
        "clean_train": (n(lambda r: r["refined"] and r["split"] == "train"), 2909),
        "clean_test": (n(lambda r: r["refined"] and r["split"] == "test"), 1701),
        "test_hold": (n(lambda r: r["refined"] == "hold" and r["split"] == "test"), 702),
        "test_shift": (n(lambda r: r["refined"] == "shift" and r["split"] == "test"), 999),
        "human": (n(lambda r: r["human"]), 511),
        "F1": (n(lambda r: r["frame"] == "F1_core_agree"), 3856),
        "F2": (n(lambda r: r["frame"] == "F2_soft_consensus"), 602),
        "F3": (n(lambda r: r["frame"] == "F3_human_clean"), 150),
        "F4": (n(lambda r: r["frame"] == "F4_human_nonclean"), 361),
        "F5": (n(lambda r: r["frame"] == "F5_interruption"), 111),
        "F6": (n(lambda r: r["frame"] == "F6_contested"), 40),
        "F7": (n(lambda r: r["frame"] == "F7_continuation"), 3459),
        "single_judged": (n(lambda r: r["frame"] == "single_judged"), 2),
    }
    for name, (got, expect) in checks.items():
        status = "OK " if got == expect else "FAIL"
        print(f"{status} {name}: {got} (expect {expect})")
    bad = {k: v for k, v in checks.items() if v[0] != v[1]}
    assert not bad, bad

    # judge-split rate inside clean gold (F2+F3 among refined)
    split_clean = n(lambda r: r["refined"] and r["vote_35"] and r["vote_31"]
                    and r["vote_35"] != r["vote_31"])
    print(f"judge-split within clean gold: {split_clean} (expect 752)")

    json.dump(rows, open(DATA / "analysis_table.json", "w"))
    print(f"wrote {len(rows):,} rows -> data/analysis_table.json")


if __name__ == "__main__":
    main()
