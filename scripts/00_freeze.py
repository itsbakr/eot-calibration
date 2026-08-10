"""Freeze the canonical data snapshot for the UncertaiNLP paper.

Copies byte-identical files from the ArabicNLP paper's frozen snapshot
(itself frozen from voice_eval by paper/scripts/freeze.py), records SHA256,
and asserts the canonical counts so any upstream drift fails loudly.
"""

import hashlib
import json
import os
import shutil
from pathlib import Path

SRC = Path(os.environ["BENCHMARK_SNAPSHOT"])  # path to the benchmark's frozen data/ release
DST = Path(__file__).resolve().parent.parent / "data"

FILES = [
    "probs_fullset.json",
    "probs_train.json",
    "probs_test.json",
    "probs_canonical.json",
    "corpus_frozen.json",
    "alq_v3_human_gold.json",
    "chid_map.json",
]


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    DST.mkdir(exist_ok=True)
    manifest = {}
    for name in FILES:
        src, dst = SRC / name, DST / name
        shutil.copyfile(src, dst)
        digest = sha256(dst)
        assert digest == sha256(src), name
        manifest[name] = {"sha256": digest, "bytes": dst.stat().st_size}
        print(f"froze {name}  {digest[:12]}  {dst.stat().st_size:,} B")

    fullset = json.load(open(DST / "probs_fullset.json"))
    assert len(fullset["rows"]) == 8581, len(fullset["rows"])
    for fname, expect in [("probs_train.json", 2909), ("probs_test.json", 1701),
                          ("probs_canonical.json", 4610)]:
        rows = json.load(open(DST / fname))["rows"]
        assert len(rows) == expect, (fname, len(rows))

    corpus = json.load(open(DST / "corpus_frozen.json"))
    n_tp = sum(len(c["turn_points"]) for c in corpus["clips"])
    assert len(corpus["clips"]) == 33, len(corpus["clips"])
    assert n_tp == 8581, n_tp
    n_human = sum(1 for c in corpus["clips"] for tp in c["turn_points"]
                  if tp["eou"].get("human"))
    assert n_human == 511, n_human

    json.dump(manifest, open(DST / "FREEZE_MANIFEST.json", "w"), indent=1)
    print(f"OK: 33 clips, {n_tp:,} turn-points, {n_human} human-adjudicated")


if __name__ == "__main__":
    main()
