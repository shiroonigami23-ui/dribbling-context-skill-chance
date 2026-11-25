# Submitting to JSDSS (Journal of Statistics and Data Science in Sports)

Diamond open access: **no fee to submit, publish or read**. Review is not anonymised. The repository link stays in the
Acknowledgements.

## 1. Register (one-time email, send it yourself)

**To:** jsds.sports@gmail.com
**From:** shiroonigami23@gmail.com
**Subject:** Registration request: new submission to JSDSS

> Dear JSDSS editors,
>
> I would like to register as an author for the Journal of Statistics and Data Science in Sports.
>
> Full name: Shiro Oni
> Manuscript title: Separating context, skill and chance in football dribbling: empirical Bayes take-on ratings, their reliability and a break-even rule for when to dribble
> Preferred contact email: shiroonigami23@gmail.com
> ORCID: 0009-0007-2131-9570
>
> Thank you,
> Shiro Oni

## 2. Submit the PDF on OJS

Portal: <https://journals.charlotte.edu/index.php/jsdss/index>

| Field | Value |
|---|---|
| Manuscript file | `manuscript.pdf` |
| Title | Separating context, skill and chance in football dribbling: empirical Bayes take-on ratings, their reliability and a break-even rule for when to dribble |
| Abstract | copy from `index.qmd` (198 words) |
| Keywords | empirical Bayes; reliability; generalized linear mixed models; decision analysis; measurement agreement; football |
| Author | Shiro Oni · shiroonigami23@gmail.com · ORCID 0009-0007-2131-9570 · Rustamji Institute of Technology, Gwalior, India |
| Comments to the Editor | paste `cover_letter.md` (it states the earlier JQAS desk rejection) |

**Checklist items to confirm:**
- [x] The work is not published or under consideration elsewhere. JQAS declined it on 23 Nov 2026.
- [x] DOIs or URLs are given for references where available.
- [x] The text follows the Author Guidelines. It uses the JSDSS Quarto template.
- [x] The manuscript file is a PDF.

## 3. If invited to the GitHub review stage

This repository is already built from `jsds-sports/jsdss-manuscript-template`. The steps are in
`jsdss-process/instructions-authors.md`.

The one remaining job is to replace the committed PNGs in `index.qmd` with code chunks that draw each figure from `results/`. JSDSS asks for this at that stage, and the plotting code is already in `code/src/`.

## Files

| File | Use |
|---|---|
| `manuscript.pdf` | Upload to OJS |
| `manuscript.docx` | Editable copy |
| `cover_letter.pdf` / `.docx` / `.md` | Comments to the Editor |
