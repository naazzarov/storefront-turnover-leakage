# arXiv submission checklist

Everything below is ready to paste into the arXiv submission form. The upload
archive is `research/arxiv-submission.tar.gz` (52 KB: `main.tex` plus five
figure PDFs, flat, no subdirectories).

---

## Before you upload

- [x] **Repository stays private.** The paper therefore claims only that code and
      derived data are "available from the authors on request", and prints no
      GitHub link. If you later decide to publish the repository, add the link in
      a revised version rather than leaving a dead one in v1.
- [x] **Affiliation** — School of Software Engineering, Sichuan University, Chengdu.
- [x] **Authors** — Bahodir Nazarov and Oussama Jabrane. Enter *both* in the
      arXiv author field, in this order. arXiv emails every listed author when
      the paper is announced, so Oussama should have seen the final PDF first.
- [x] **Acknowledgements** — Huanyu Li, for supervision.
- [x] **Endorsement** — already held, so the endorsement step below does not apply.
- [ ] **Read the paper.** Your name is on it. Every number is reproducible with
      `python3 run_experiments.py`, so anything you doubt can be checked.
- [x] **Prior-art search** — done 2026-09-20; see the section near the end. It
      materially changed the paper's framing.

---

## Categories

| Field | Value |
|---|---|
| **Primary** | `cs.LG` — Machine Learning |
| Cross-list | `stat.AP` — Applications |
| Cross-list | `physics.soc-ph` — Physics and Society |

`cs.LG` is the honest primary: the contribution is evaluation methodology
(leakage, cross-validation design, model selection), demonstrated on an urban
task rather than proposing an urban method.

## Licence

**CC BY 4.0.** Most permissive; allows reuse of figures with attribution.

---

## Title

```
Spatial Blocking Does Not Prevent Group-Aggregate Leakage: A Case Study in Urban Turnover Prediction
```

## Abstract

Paste the text below (arXiv accepts plain text; keep the line breaks loose).

```
Two evaluation hazards are individually well documented: spatially blocked
cross-validation is required for geographic data, with block size the dominant
choice (Roberts et al. 2017; Ploton et al. 2020; Stock 2025), and target-encoded
categorical features leak unless folds respect the encoding groups
(Micci-Barreca 2001; Prokhorenkova et al. 2018). We show that these interact in a
way that defeats the standard remedy, and that the interaction is easy to miss
because it produces no visible symptom.

On a task built from 1.21 million City of Chicago business licences spanning
1995-2026, we predict which of 135,553 storefronts exhibit excess tenant
turnover. Random k-fold cross-validation reports ROC AUC 0.970 and average
precision 0.756 (15.1x the base rate); partitioning by administrative unit
reports 0.693 and 0.114 (2.3x).

Our contribution is to locate the channel. Ablation shows the inflation is
carried almost entirely by group aggregates - ward and community-area
leave-one-out means, which inflate AUC by 0.321 - rather than by proximity:
neighbourhood features computed at 50-500 m inflate by only 0.034. Because the
channel is group membership rather than distance, spatial blocking does not close
it at any scale a practitioner would plausibly select. Blocks of 100 m to 1 km
leave performance statistically unchanged (AUC 0.971-0.977 against 0.970 for
random splits); only blocks approaching the size of the administrative units
themselves recover the honest estimate. A practitioner following current guidance
- choosing block size from the autocorrelation range of the predictors - would
therefore still obtain an inflated estimate here, because the relevant scale is
set by the aggregation units, not by the autocorrelation of the data.

We confirm in this domain the known result that leakage equalises under honest
evaluation: inflation ranges from 0.013 AUC for logistic regression to 0.277 for
gradient boosting, while blocked scores for all families fall between 0.62 and
0.70, so random splits corrupt model selection as well as assessment.

The substantive finding survives honest evaluation and is, to our knowledge, new:
excess turnover concentrates at individual addresses rather than districts, with
neighbours only weakly elevated and tenancies failing roughly 1.5x faster per
renewal cycle. All inputs are public domain and the pipeline is released in full.
```

## Comments field

```
14 pages, 6 figures, 7 tables. Code and derived data available from the authors
on request.
```

## ACM class (optional)

```
I.2.6; I.5.2; H.2.8
```

---

## Endorsement

Already held for this submission. (Retained for reference: first-time submitters
to `cs.LG` normally need one, and registering with an institutional email often
triggers auto-endorsement instead.)

---

## Submission steps

1. Create or log into an account at `https://arxiv.org`, using your institutional
   email.
2. Start a new submission.
3. Upload `arxiv-submission.tar.gz`. arXiv compiles the LaTeX itself; do not
   upload the PDF instead, as source submissions are preferred and let readers
   fetch the source.
4. Check the PDF arXiv generates. It should match
   `research/paper/main.pdf`: 14 pages, six figures, seven tables.
5. Enter the title, abstract, categories, and comments from above.
6. Choose the CC BY 4.0 licence.
7. Submit. Moderation typically takes one to two business days, after which the
   paper appears with an identifier of the form `arXiv:2609.NNNNN`.

Announcements happen on weekdays; a submission cleared at a weekend appears the
following working day.

---

## Prior-art search: completed 2026-09-20

The search was run and **changed the paper**. Findings:

- **Block size matters most** is already published: Stock (2025), *Choosing blocks
  for spatial cross-validation*, Frontiers in Remote Sensing, across 1,426
  synthetic datasets. blockCV has estimated predictor autocorrelation range for
  block-size selection since 2019.
- **Folds must align with encoding groups** is documented, including in
  scikit-learn's `TargetEncoder` docs (group routing to `GroupKFold`), and
  Cerqua, Letta & Pinto (2024) name spatial leakage as cross-sectional leakage
  with aggregate panel data.
- **Leakage equalises model families** was shown by Rosenblatt et al. (2024) in
  Nature Communications for connectome models.

The paper was accordingly reframed. It no longer claims to discover any of the
above; all three are cited as established. The retained claim is narrower: the
two remedies are not interchangeable, and where features aggregate over
administrative units, spatial blocking at any plausible scale fails because the
governing scale is the aggregation unit rather than the autocorrelation range.
The ablation separating group aggregates (+0.321 AUC) from proximity features
(+0.034) is the evidence for that claim.

If a reviewer finds work stating that specific result, the honest response is to
retitle toward the substantive urban finding, which the search found no prior
work on.

## After posting

- Add the arXiv identifier to the repository README.
- Consider minting a Zenodo DOI for the code, and citing it in a revised version.
- Chicago's licence register updates continuously, so a future version can
  refresh the extract; `fetch_chicago.py` verifies row counts against the server,
  so a truncated refresh will fail loudly rather than silently.
