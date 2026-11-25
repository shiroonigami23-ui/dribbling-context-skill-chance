#!/usr/bin/env bash
# Reproduce every table and figure in the paper, from raw open data, in one go.
# Usage:  cd code && bash run_all.sh        (about 1-2 hours on a laptop; needs ~4 GB free disk)
# Outputs: code/results/ (tables, JSON with every number quoted in the paper) and code/figures/.
set -euo pipefail
cd "$(dirname "$0")"
SKIP_FETCH=${SKIP_FETCH:-0}   # SKIP_FETCH=1 bash run_all.sh  reuses the committed Study 1 events
step() { echo; echo "=== $1 ($(date +%H:%M:%S))"; shift; "$@"; }
if [ "$SKIP_FETCH" != 1 ]; then step "Study 1 data (StatsBomb; re-downloads ~350 MB)" python src/fetch.py; fi
step "Study 1: context model, shrinkage, benchmark" python src/analysis.py
step "Study 1: sensitivity"        python src/sensitivity.py
step "Study 1: robustness"         python src/robustness.py
step "Study 1: practice tables"    python src/practice.py
step "Study 1: decision analysis"  python src/decision.py
step "Studies 2-4 data (StatsBomb women, new men, 360)" python src/fetch_more.py
step "Studies 2-3: replication"    python src/replicate.py
step "Study 4: 360"                python src/study360.py
step "Studies 5-6 data (Wyscout)"  python src/fetch_wyscout.py
step "Study 5: parse Wyscout"      python src/wyscout.py
step "Study 5: replication"        python src/study5_wyscout.py
step "Study 6: crossed GLMM (R/lme4)" python src/hierarchical.py
step "Bayesian decisions"          python src/decision_bayes.py
step "Figures"                     python src/fig_replication.py
step "Figures"                     python src/fig_attack_defence.py
step "Figures"                     python src/figures.py
echo; echo "Done. Compare with the committed results:  python src/check_reproduction.py"
