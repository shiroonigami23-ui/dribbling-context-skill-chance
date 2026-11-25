---
title: "Cover letter"
---

25 November 2026

Dr Michael Schuckers, Editor-in-Chief
*Journal of Statistics and Data Science in Sports*

Dear Dr Schuckers,

I submit the manuscript **"Separating context, skill and chance in football dribbling: empirical Bayes take-on ratings, their reliability and a break-even rule for when to dribble"** for consideration as an original research article in *JSDSS*.

Take-on completion rate is football's standard dribbling statistic, yet it mixes the difficulty of each attempt, the player's skill and chance. The paper separates the three on 188,841 take-ons from 5,849 men's and women's matches and two data providers. It combines a logistic context model, empirical Bayes shrinkage, a reliability curve, a crossed attacker–defender generalized linear mixed model and a break-even decision rule. Its main findings are that context moves success by four times the spread of player skill, that a rating needs 135–180 attempts to reach a reliability of 0.7 while the median player attempts 12–27 a season, and that the raw completion rate predicts a player's future take-ons worse than the league average in every dataset. It also finds that a take-on beats a pass on average only in the centre of the penalty area.

I believe the paper fits *JSDSS* for three reasons:

- **It is an original application of statistical methodology to sports data that is checked out of sample at every step.** The ratings are validated on held-out matches and later half-seasons, and in the WSL on the next season. The results are replicated on women's football, on competitions the model never saw, with 360 freeze frames, and on a second provider (Wyscout). A measurement-agreement study compares the two providers on the same World Cup matches.
- **It is fully reproducible from open data.** All data are public (StatsBomb open data; the Wyscout release of Pappalardo et al., 2019). A single script, `run_all.sh`, downloads the raw data and rebuilds every number, table and figure, and a checking script compares a fresh run with the committed results. The repository is linked in the Acknowledgements. I will move the paper into the *JSDSS* Quarto template, with figures drawn by code in the manuscript, if it is invited to the GitHub review stage. The submitted PDF already uses the template's format.
- **It reports a data issue that other users of the open data need to know about.** In the public 360 release, the freeze frames of almost all failed take-ons are stored mirrored. Used as released they leak the outcome, and a model reaches a spurious AUC of 0.80. The paper documents this and the released code corrects it.

For transparency: an earlier version was submitted to the *Journal of Quantitative Analysis in Sports* (DGJQAS.2026.0321). On 23 November 2026 the Editor-in-Chief declined it without review because JQAS now prioritises new statistical methodology, and he suggested *JSDSS*. Since then the manuscript has been restructured for *JSDSS*. The study details are now in an appendix, the main text is focused on the method and its validation, the reproducibility pipeline has been rerun end to end, and the code paths have been cleaned so that it runs on any machine. The paper is not under consideration elsewhere.

The work uses only publicly released match-event data and involves no human participants. An AI assistant was used to help write analysis code and edit the language, as stated in the Acknowledgements; I checked all code and outputs and take full responsibility for the content.

Thank you for considering the manuscript.

Yours sincerely,

Shiro Oni
Department of Computer Science and Engineering, Rustamji Institute of Technology, Gwalior, India
shiroonigami23@gmail.com · ORCID 0009-0007-2131-9570
