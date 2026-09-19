"""Predictive models for storefront turnover.

Two questions are asked of the data, and they need different machinery.

*Is a curse predictable in advance?* If a storefront's excess turnover could be
anticipated from its context alone -- its surroundings, its sector, the building
it sits in -- then "cursed" is a shorthand for measurable disadvantage rather
than an unexplained property. This is a classification task, and its honest
answer may well be "barely", which would itself be the finding: that turnover
concentrates in ways the observable context does not explain.

*Does a tenancy's end depend on the location?* Survival analysis on tenancy
durations, which must respect two features of the data established earlier:
durations are quantized to licence terms (so the natural time axis is renewal
cycles, not days) and 38k tenancies are right-censored (still trading at the
snapshot).

Both models are evaluated with spatially blocked cross-validation. Storefronts
near one another share unobserved characteristics -- the same landlord, the same
footfall, the same street works -- so a random split leaks: a model can memorize
a block from its training half and appear to predict the held-out half. Blocking
by community area removes that path and gives an estimate that reflects
generalization to unseen parts of the city.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

RANDOM_STATE = 20260919

# Number of spatial blocks held out in turn during cross-validation.
N_SPATIAL_FOLDS = 5


@dataclass
class ClassificationResult:
    """Cross-validated performance, with the baseline it must beat."""

    auc_mean: float
    auc_std: float
    average_precision: float
    baseline_rate: float
    n_samples: int
    n_positive: int
    feature_importance: pd.DataFrame = field(default_factory=pd.DataFrame)
    fold_aucs: list[float] = field(default_factory=list)

    def summary(self) -> str:
        lift = self.average_precision / self.baseline_rate if self.baseline_rate else np.nan
        return (
            f"n={self.n_samples:,} ({self.n_positive:,} positive, "
            f"{100 * self.baseline_rate:.1f}%)\n"
            f"ROC AUC          {self.auc_mean:.3f} +/- {self.auc_std:.3f}\n"
            f"Average precision {self.average_precision:.3f} "
            f"(baseline {self.baseline_rate:.3f}, lift {lift:.2f}x)"
        )


def spatial_folds(
    df: pd.DataFrame,
    *,
    group_col: str = "community_area",
    n_folds: int = N_SPATIAL_FOLDS,
    random_state: int = RANDOM_STATE,
) -> np.ndarray:
    """Assign rows to spatially blocked folds.

    Whole community areas are held out together. A random row-wise split would
    place near-identical neighbouring storefronts on both sides of the split,
    and the resulting score would measure memorization of local conditions
    rather than transfer to unseen areas.

    Areas are shuffled and distributed greedily so folds are similar in size
    despite areas differing widely in how many storefronts they contain.
    """
    groups = df[group_col].fillna("unknown").astype(str)
    sizes = groups.value_counts()

    rng = np.random.default_rng(random_state)
    order = sizes.index.to_numpy()
    rng.shuffle(order)

    fold_totals = np.zeros(n_folds, dtype=int)
    assignment: dict[str, int] = {}
    # Largest-first keeps one huge area from unbalancing the folds.
    for name in sorted(order, key=lambda g: -sizes[g]):
        target = int(np.argmin(fold_totals))
        assignment[name] = target
        fold_totals[target] += sizes[name]

    log.info("spatial folds sized %s across %d areas", fold_totals.tolist(), len(sizes))
    return groups.map(assignment).to_numpy()


def _prepare_matrix(
    df: pd.DataFrame, feature_cols: list[str]
) -> tuple[np.ndarray, list[str]]:
    """Build the model matrix, dropping all-missing and constant columns.

    Missing values are median-imputed with an accompanying indicator column:
    the fact that a statistic is missing is itself informative (a location with
    no neighbours within 50m is genuinely isolated, not merely unmeasured), and
    silently filling it would discard that.
    """
    x = df.loc[:, feature_cols].astype(float)

    keep = [c for c in x.columns if x[c].notna().any() and x[c].nunique(dropna=True) > 1]
    x = x.loc[:, keep]

    names: list[str] = []
    blocks: list[np.ndarray] = []
    for col in x.columns:
        values = x[col]
        if values.isna().any():
            blocks.append(values.isna().to_numpy(dtype=float)[:, None])
            names.append(f"{col}__missing")
        blocks.append(values.fillna(values.median()).to_numpy()[:, None])
        names.append(col)

    return np.hstack(blocks), names


def predict_curse(
    df: pd.DataFrame,
    feature_cols: list[str],
    *,
    label_col: str = "is_cursed",
    group_col: str = "community_area",
    n_folds: int = N_SPATIAL_FOLDS,
    random_state: int = RANDOM_STATE,
) -> ClassificationResult:
    """Can excess turnover be predicted from context alone?

    Gradient boosting under spatially blocked cross-validation. Average
    precision is reported alongside ROC AUC because the classes are heavily
    imbalanced (~5% positive), and AUC alone flatters a model on imbalanced
    data: the baseline for average precision is the positive rate itself, which
    makes the lift over chance explicit.
    """
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.inspection import permutation_importance
    from sklearn.metrics import average_precision_score, roc_auc_score

    work = df.loc[df[label_col].notna()].copy()
    y = work[label_col].astype(int).to_numpy()
    x, names = _prepare_matrix(work, feature_cols)

    folds = spatial_folds(work, group_col=group_col, n_folds=n_folds,
                          random_state=random_state)

    oof = np.full(len(y), np.nan)
    fold_aucs: list[float] = []
    importances: list[np.ndarray] = []

    for fold in range(n_folds):
        test = folds == fold
        train = ~test
        if y[train].sum() == 0 or y[test].sum() == 0:
            log.warning("fold %d has no positives on one side; skipping", fold)
            continue

        model = HistGradientBoostingClassifier(
            max_iter=300,
            learning_rate=0.06,
            max_leaf_nodes=31,
            l2_regularization=1.0,
            random_state=random_state,
        )
        model.fit(x[train], y[train])
        proba = model.predict_proba(x[test])[:, 1]
        oof[test] = proba
        fold_aucs.append(float(roc_auc_score(y[test], proba)))

        # Permutation importance on held-out data only: importances read from
        # the training fold reflect what the model fitted, not what generalizes.
        perm = permutation_importance(
            model, x[test], y[test], n_repeats=5,
            random_state=random_state, scoring="roc_auc",
        )
        importances.append(perm.importances_mean)

    valid = ~np.isnan(oof)
    result = ClassificationResult(
        auc_mean=float(np.mean(fold_aucs)) if fold_aucs else np.nan,
        auc_std=float(np.std(fold_aucs)) if fold_aucs else np.nan,
        average_precision=float(average_precision_score(y[valid], oof[valid])),
        baseline_rate=float(y.mean()),
        n_samples=int(len(y)),
        n_positive=int(y.sum()),
        fold_aucs=fold_aucs,
    )

    if importances:
        mean_imp = np.mean(importances, axis=0)
        result.feature_importance = (
            pd.DataFrame({"feature": names, "importance": mean_imp})
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )

    log.info("curse prediction:\n%s", result.summary())
    return result


def renewal_survival(
    tenancies: pd.DataFrame,
    locations: pd.DataFrame,
    *,
    label_col: str = "is_cursed",
) -> pd.DataFrame:
    """Discrete-time survival of tenancies, measured in renewal cycles.

    Days are the wrong time axis: licence terms quantize duration to renewal
    multiples, so a duration in days is largely a statement about Chicago's
    renewal calendar. The number of licences a tenancy held is the natural
    discrete clock, and it is immune to that quantization.

    Returns, for each cycle k, the share of tenancies that survived to hold at
    least k licences, split by whether their location is cursed. Right-censored
    tenancies contribute to the risk set only up to their observed cycle.
    """
    df = tenancies.merge(
        locations[["location_id", label_col]], on="location_id", how="inner"
    )
    df = df.loc[df[label_col].notna()].copy()
    df["cycles"] = df["n_licences"].astype(int).clip(lower=1)

    rows: list[dict] = []
    max_cycle = int(min(df["cycles"].max(), 10))

    for cursed_flag, grp in df.groupby(label_col):
        # Censored tenancies are still trading, so their observed cycle count is
        # a lower bound; they remain in the risk set but never count as events.
        censored = grp["is_right_censored"].to_numpy()
        cycles = grp["cycles"].to_numpy()

        for k in range(1, max_cycle + 1):
            at_risk = int((cycles >= k).sum())
            # An event at cycle k means: reached k, did not renew beyond it,
            # and was not still trading when observation stopped.
            events = int(((cycles == k) & (~censored)).sum())
            rows.append({
                label_col: bool(cursed_flag),
                "cycle": k,
                "at_risk": at_risk,
                "ended": events,
                "hazard": events / at_risk if at_risk else np.nan,
            })

    out = pd.DataFrame(rows)
    # Kaplan-Meier survival within each group.
    out["survival"] = (
        out.sort_values("cycle")
        .groupby(label_col)["hazard"]
        .transform(lambda h: (1 - h).cumprod())
    )
    return out


def permutation_null(
    df: pd.DataFrame,
    *,
    label_col: str = "is_cursed",
    value_col: str = "n_tenants_nbr_50m",
    n_permutations: int = 1000,
    group_col: str = "community_area",
    random_state: int = RANDOM_STATE,
) -> dict[str, float]:
    """Test the neighbour contrast against a spatially constrained null.

    The observed gap between cursed and non-cursed locations must be compared
    against a null that preserves spatial structure. Shuffling labels citywide
    would break the clustering of turnover within districts and make almost any
    gap look significant. Labels are therefore permuted *within* community
    areas, so the null retains each district's own turnover level and only the
    assignment of the label inside it is randomized.
    """
    work = df.loc[df[label_col].notna() & df[value_col].notna()].copy()
    labels = work[label_col].astype(bool).to_numpy()
    values = work[value_col].astype(float).to_numpy()
    groups = work[group_col].fillna("unknown").astype(str).to_numpy()

    observed = values[labels].mean() - values[~labels].mean()

    rng = np.random.default_rng(random_state)
    order = np.argsort(groups, kind="stable")
    sorted_groups = groups[order]
    boundaries = np.flatnonzero(np.r_[True, sorted_groups[1:] != sorted_groups[:-1]])
    slices = np.split(np.arange(len(order)), boundaries[1:])

    sorted_labels = labels[order]
    sorted_values = values[order]

    null = np.empty(n_permutations)
    for i in range(n_permutations):
        shuffled = sorted_labels.copy()
        for sl in slices:
            shuffled[sl] = rng.permutation(shuffled[sl])
        null[i] = sorted_values[shuffled].mean() - sorted_values[~shuffled].mean()

    # Two-sided p-value with the +1 correction, so a p of exactly zero is never
    # reported from a finite number of permutations.
    p = (np.sum(np.abs(null) >= abs(observed)) + 1) / (n_permutations + 1)

    return {
        "observed_difference": float(observed),
        "null_mean": float(null.mean()),
        "null_std": float(null.std()),
        "p_value": float(p),
        "n_permutations": int(n_permutations),
    }
