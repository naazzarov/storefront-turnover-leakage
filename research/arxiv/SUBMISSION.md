# arXiv submission checklist

Everything below is ready to paste into the arXiv submission form. The upload
archive is `research/arxiv-submission.tar.gz` (40 KB: `main.tex` plus five
figure PDFs, flat, no subdirectories).

---

## Before you upload

- [ ] **Make the code repository public.** The paper states that code and derived
      data are available at
      `https://github.com/naazzarov/storefront-turnover-leakage`. That repository
      is currently private, so the claim is not yet true and readers will click
      the link. Either make it public or remove the claim.
- [ ] **Add your affiliation** to the author block in `main.tex` (line 18).
      It currently carries only an email address.
- [ ] **Read the paper.** Your name is on it. Every number is reproducible with
      `python3 run_experiments.py`, so anything you doubt can be checked.
- [ ] **Search for prior art** on target-encoding leakage under grouped
      cross-validation (suggested queries below). This is the single claim most
      exposed to a reviewer finding earlier work.

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
Group-Aggregate Features Defeat Spatial Cross-Validation: A Case Study in Urban Storefront Turnover
```

## Abstract

Paste the text below (arXiv accepts plain text; keep the line breaks loose).

```
Spatially blocked cross-validation is the standard defence against optimistic
performance estimates on geographic data. We show that it fails silently when
block size is not chosen with respect to the spatial support of the features,
and that the dominant leakage channel is not fine-grained spatial autocorrelation
but group-aggregate features, that is, leave-one-out target encodings over
administrative units.

On a task built from 1.21 million City of Chicago business licences spanning
1995-2026, we predict which of 135,553 storefronts exhibit excess tenant
turnover. Random k-fold cross-validation reports ROC AUC 0.970 and average
precision 0.756 (15.1x over the base rate). Spatially blocked cross-validation
over the same data and model reports 0.693 and 0.114 (2.3x), an average-precision
inflation of more than sixfold.

Three findings follow. First, blocking at 100 m to 1 km is statistically
indistinguishable from no blocking at all (AUC 0.971-0.977); performance
collapses only once blocks approach administrative scale. Practitioners who adopt
spatial cross-validation but choose small blocks obtain inflated numbers carrying
a methodologically respectable label. Second, leakage is concentrated in model
families by flexibility: logistic regression inflates by 0.013 AUC while gradient
boosting inflates by 0.277. Under honest evaluation all families are equivalent
(0.62-0.70), so random splits corrupt model selection, not merely model
assessment. Third, ablation localises the leak: ward and community-area
aggregates inflate by 0.321 AUC against 0.034 for 50-500 m neighbourhood
features. Because a leave-one-out group mean is near-constant within its group,
it acts as a group identifier, and the evaluation split must align with the
grouping used to construct it.

The substantive result survives honest evaluation: excess turnover concentrates
at individual addresses rather than districts, with neighbours only weakly
elevated and tenancies failing roughly 1.5x faster per renewal cycle. All inputs
are public domain and the pipeline is released in full.
```

## Comments field

```
13 pages, 5 figures. Code and derived data:
https://github.com/naazzarov/storefront-turnover-leakage
```

## ACM class (optional)

```
I.2.6; I.5.2; H.2.8
```

---

## Endorsement

First-time submitters to `cs.LG` normally need an endorsement.

1. **Register with your university email address**, not a personal one. arXiv
   auto-endorses many institutional domains, which avoids the step entirely.
2. If prompted anyway, arXiv issues an endorsement code. Send it to someone who
   has published in `cs.LG` — a supervisor, a lecturer, or an author you cite.
   They confirm you are a genuine researcher; they are not vouching for the
   paper's correctness.

---

## Submission steps

1. Create or log into an account at `https://arxiv.org`, using your institutional
   email.
2. Start a new submission.
3. Upload `arxiv-submission.tar.gz`. arXiv compiles the LaTeX itself; do not
   upload the PDF instead, as source submissions are preferred and let readers
   fetch the source.
4. Check the PDF arXiv generates. It should match
   `research/paper/main.pdf`: 13 pages, five figures, three tables.
5. Enter the title, abstract, categories, and comments from above.
6. Choose the CC BY 4.0 licence.
7. Submit. Moderation typically takes one to two business days, after which the
   paper appears with an identifier of the form `arXiv:2609.NNNNN`.

Announcements happen on weekdays; a submission cleared at a weekend appears the
following working day.

---

## Suggested prior-art searches

Run these before submitting. If any returns a paper stating the group-aggregate
result directly, the novelty framing in Section 2 needs adjusting — which is
easier to do now than after posting.

- "target encoding" leakage cross-validation grouped
- "leave-one-out encoding" leakage grouped splits
- spatial cross-validation "block size" feature scale
- group k-fold target encoding optimistic bias
- mean target encoding "data leakage" evaluation protocol

Known related work already cited: Roberts et al. (2017), Ploton et al. (2020),
Valavi et al. (2019) on spatial cross-validation; Kaufman et al. (2012) and
Kapoor & Narayanan (2023) on leakage; Micci-Barreca (2001) and Prokhorenkova et
al. (2018) on target encoding.

---

## After posting

- Add the arXiv identifier to the repository README.
- Consider minting a Zenodo DOI for the code, and citing it in a revised version.
- Chicago's licence register updates continuously, so a future version can
  refresh the extract; `fetch_chicago.py` verifies row counts against the server,
  so a truncated refresh will fail loudly rather than silently.
