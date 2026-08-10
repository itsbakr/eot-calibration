# Calibration and Selective Prediction for Arabic End-of-Turn Detection

Code and frozen analysis data for the UncertaiNLP 2026 submission
*"Confidently Cutting the Caller Off: Calibration and Selective Prediction
for Arabic End-of-Turn Detection"* (anonymous review copy).

Every number, table, and figure in the paper is computed by the scripts in
this repository from `data/analysis_table.json`, a frozen per-turn-point
table (8,581 rows) joining the detector's probabilities to the benchmark's
judge votes, judge probabilities, human adjudications, cells, and splits.

## Reproduce

```bash
pip install -r requirements.txt
python scripts/02_calibration.py   # Q1: ECE / Brier / Murphy / recalibration -> data/results_calibration.json
python scripts/03_softlabels.py    # Q2: soft labels, ambiguity-blindness    -> data/results_softlabels.json
python scripts/04_decision.py      # Q3: CRC, Mondrian, cost frame           -> data/results_decision.json
python scripts/05_figures.py       # figures/fig_*.pdf and tables/tab_*.tex
```

All randomness (bootstrap CIs, repeated clip splits) is seeded
(`numpy.random.default_rng(13)`), so outputs are deterministic.

## Data and provenance

- `data/analysis_table.json` — one row per turn-point: `clip`, `channel`
  (pseudonymized `Ch1`–`Ch6`, consistent with the benchmark's anonymized
  release), `dialect`, `at_ms`, `split`, `cell`, `diar`, `prob` (Smart Turn
  v3.2 P(complete)), `p_complete` (pooled judge probability), per-judge
  votes and probabilities, `human`, `gap_ms`, `frame` (F1–F7 provenance),
  and `refined` (clean-gold label or null).
- `data/results_*.json` — the computed results the paper reports.
- `data/FREEZE_MANIFEST.json` — SHA-256 hashes of the upstream benchmark
  files the table was built from.
- `scripts/00_freeze.py` and `scripts/01_prepare.py` document how the
  table was built from the benchmark's frozen release (set
  `BENCHMARK_SNAPSHOT` to its `data/` directory); they are included for
  provenance and are not needed to reproduce the paper's numbers.

The underlying audio benchmark (labels, scoring harness, and downloader) is
released separately by its authors; this repository contains only derived,
de-identified analysis data.

## License

Code: MIT. Derived data: CC BY 4.0 upon publication, following the
benchmark's license.
