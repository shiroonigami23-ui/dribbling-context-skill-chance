# Changelog

## [v5.2] 2026-11-26: figures drawn from code
### Changed
- The manuscript now draws all 11 figures itself, in code chunks that read `results/`. The new module `code/src/paperfigs.py` holds one function per figure, using the same drawing code as the analysis scripts.
- `publish.yml` installs Python (jupyter, numpy, pandas, matplotlib, pyarrow) so GitHub Actions can execute those chunks. Notebook links are switched off in the HTML and PDF output.
- The rendered PDF has the same pages, figures and numbers as the submitted version.

## [v5.1] 2026-11-26: reference check
### Fixed
- Checked the cited references. Fixed the Opta Analyst dribbling guide: author Ryan Benson, published November 2023, not 2025.
- Removed the appendix comparison with a social-media Opta graphic (UtdArena, 2020), together with its figure and reference. Its data are not openly available, so the comparison could not be reproduced.

## [v5] 2026-11-25: JSDSS format, own repository
### Changed
- Moved out of the Research-Vault monorepo into this repository, built on the
  [JSDSS manuscript template](https://github.com/jsds-sports/jsdss-manuscript-template).
- Manuscript rewritten in Quarto (`index.qmd`, JASA/JSDSS format, not anonymised; repository linked in
  the Acknowledgements). The main text now covers the method, Study 1 and a summary of the
  replications. Full Study 2–6 details are in Appendix A, and sensitivity, career and data-source material in Appendix B.
- Introduction restated: five research questions; contributions condensed to five.
- Method: added a note on the two-stage plug-in rating against the one-stage mixed model.
### Fixed (found by a full rerun from scratch)
- Several scripts saved figures to a `paper/` folder that does not exist in a fresh clone, so the
  pipeline crashed. All figures now go to `code/figures/`.
- Hard-coded local Wyscout paths replaced by `WYSCOUT_DIR` (default
  `code/data/wyscout_raw/`), filled by the new `src/fetch_wyscout.py`. Temporary files of the R step
  now go to a private temp dir.
- Study 4 (360) results depended on the order in which downloaded files were read. File lists,
  freeze-frame rows and cross-validation folds are now sorted, which makes the run deterministic.
  Updated numbers: the skill-spread estimates in the 360 subsample (4.6/3.8 points at 15 attempts,
  previously 4.4/3.6), the 90% intervals of the 360 break-even rates, and the Study 3 refitted AUC
  (0.586, previously 0.588). No conclusion changes.
- `scikit-learn` was missing from `requirements.txt`.
### Added
- `code/run_all.sh` rebuilds everything from raw open data in one command.
- `code/src/check_reproduction.py` compares a fresh run with the committed results. Verified run on
  25 Nov 2026: 289,371 numbers, all within 1%.
- `code/requirements-lock.txt` lists exact package versions. Added a GitHub Actions reproducibility check, LICENSE and CITATION.cff.

## [v4] 2026-11-17
Second provider (Wyscout), two-provider agreement on World Cup 2018, cross-provider stability,
crossed attacker × defender GLMM, Bayesian dribble-or-pass rule.

## [v3] 2026-11
Studies 2–4: women's football with a next-season test, transfer to 463 new men's matches, 360
freeze frames (with the mirrored-frame fix).

## [v2] 2026-11-14
Reframed from a Messi case study into a general method: context map, reliability curve,
out-of-sample test, break-even rule, released ratings.

## [v1] 2026-11-13
Player-focused Messi case study.
