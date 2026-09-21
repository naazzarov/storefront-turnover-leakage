"""Run every experiment in the paper and write its result tables and figures.

One entry point, so the numbers quoted in the text, the numbers in the tables
and the numbers in the figures all come from the same execution. Result tables
are written to ``outputs/tables`` as CSV and the figures read those files rather
than recomputing anything.

Usage:
    python3 run_experiments.py                 # everything
    python3 run_experiments.py --skip-scale    # omit the slow block sweep
"""

from __future__ import annotations

import argparse
import logging
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src import cursed, features, figures, models, paths

warnings.filterwarnings("ignore")
log = logging.getLogger("experiments")

SEED = 20260919
GRID_SIZES_M = (100, 250, 500, 1000, 2000, 5000)
FEATURE_SUPPORT_M = 500.0


def gbm() -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.06, max_leaf_nodes=31,
        l2_regularization=1.0, random_state=SEED,
    )


MODEL_FAMILIES = {
    "logistic regression": lambda: make_pipeline(
        StandardScaler(), LogisticRegression(max_iter=2000, C=1.0)),
    "kNN (k=25)": lambda: make_pipeline(
        StandardScaler(), KNeighborsClassifier(25, n_jobs=-1)),
    "random forest": lambda: RandomForestClassifier(
        300, min_samples_leaf=5, n_jobs=-1, random_state=SEED),
    "gradient boosting": gbm,
}


def evaluate(make_model, folds: np.ndarray, x: np.ndarray, y: np.ndarray) -> dict:
    """Out-of-fold AUC and average precision for one partitioning."""
    oof = np.full(len(y), np.nan)
    aucs: list[float] = []
    for fold in np.unique(folds[folds >= 0]):
        test = folds == fold
        train = (folds >= 0) & ~test
        if y[train].sum() == 0 or y[test].sum() == 0:
            continue
        model = make_model()
        model.fit(x[train], y[train])
        proba = model.predict_proba(x[test])[:, 1]
        oof[test] = proba
        aucs.append(float(roc_auc_score(y[test], proba)))
    valid = ~np.isnan(oof)
    return {
        "auc": float(np.mean(aucs)),
        "auc_std": float(np.std(aucs)),
        "ap": float(average_precision_score(y[valid], oof[valid])),
        "base_rate": float(y.mean()),
    }


def random_folds(x: np.ndarray, y: np.ndarray, k: int = 5) -> np.ndarray:
    folds = np.zeros(len(y), dtype=int)
    for i, (_, test) in enumerate(StratifiedKFold(k, shuffle=True, random_state=1).split(x, y)):
        folds[test] = i
    return folds


def grid_folds(xy: np.ndarray, size_m: float, k: int = 5, seed: int = 7) -> tuple[np.ndarray, int]:
    """Tile space into squares and assign whole squares to folds."""
    folds = np.full(len(xy), -1, dtype=int)
    ok = ~np.isnan(xy).any(axis=1)
    cells = np.floor(xy[ok] / size_m).astype(int)
    keys = [tuple(c) for c in cells]
    unique = sorted(set(keys))
    rng = np.random.default_rng(seed)
    assignment = {u: i % k for u, i in zip(unique, rng.permutation(len(unique)))}
    folds[np.where(ok)[0]] = [assignment[t] for t in keys]
    return folds, len(unique)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-scale", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    paths.ensure_dirs()

    lab = pd.read_parquet(paths.PROCESSED_DIR / "labelled.parquet").reset_index(drop=True)
    ten = pd.read_parquet(paths.PROCESSED_DIR / "tenancies.parquet")

    cols = [c for c in features.feature_columns(lab) if "exposure" not in c]
    y = lab["is_cursed"].astype(int).to_numpy()
    x, names = models._prepare_matrix(lab, cols)
    xy = features._to_xy(lab["latitude"].to_numpy(), lab["longitude"].to_numpy())

    rand = random_folds(x, y)
    admin = models.spatial_folds(lab)

    # --- Experiment A: block-size sweep -----------------------------------
    if not args.skip_scale:
        rows = [{"label": "random (none)", "block_m": np.nan, "n_blocks": np.nan,
                 **evaluate(gbm, rand, x, y)}]
        for size in GRID_SIZES_M:
            folds, n_blocks = grid_folds(xy, size)
            rows.append({"label": f"grid {size}m", "block_m": float(size),
                         "n_blocks": n_blocks, **evaluate(gbm, folds, x, y)})
        rows.append({"label": "community area", "block_m": np.nan,
                     "n_blocks": int(lab["community_area"].nunique()),
                     **evaluate(gbm, admin, x, y)})
        scale = pd.DataFrame(rows)
        scale["lift"] = scale["ap"] / scale["base_rate"]
        scale.to_csv(paths.TABLES_DIR / "block_scale.csv", index=False)
        figures.fig_block_scale(scale, feature_support_m=FEATURE_SUPPORT_M)
        print("\n=== A. block-size sweep ===")
        print(scale[["label", "n_blocks", "auc", "auc_std", "ap", "lift"]].round(3).to_string(index=False))

    # --- Experiment B: model families -------------------------------------
    rows = []
    for name, make in MODEL_FAMILIES.items():
        r = evaluate(make, rand, x, y)
        b = evaluate(make, admin, x, y)
        rows.append({"model": name, "random_auc": r["auc"], "blocked_auc": b["auc"],
                     "inflation": r["auc"] - b["auc"],
                     "random_ap": r["ap"], "blocked_ap": b["ap"]})
    fams = pd.DataFrame(rows)
    fams.to_csv(paths.TABLES_DIR / "model_families.csv", index=False)
    figures.fig_model_families(fams)
    print("\n=== B. model families ===")
    print(fams.round(3).to_string(index=False))

    # --- Experiment C: feature ablation -----------------------------------
    groups = {
        "area aggregates (ward, CA)": [c for c in cols if "_loo" in c and "nbr" not in c and "building" not in c],
        "raw coordinates": [c for c in cols if c in ("latitude", "longitude")],
        "neighbourhood LOO (50-500m)": [c for c in cols if "nbr" in c],
        "density counts": [c for c in cols if "within" in c],
        "building context": [c for c in cols if "building" in c],
    }
    rows = []
    for name, group_cols in groups.items():
        if not group_cols:
            continue
        xg, _ = models._prepare_matrix(lab, group_cols)
        r = evaluate(gbm, rand, xg, y)
        b = evaluate(gbm, admin, xg, y)
        rows.append({"family": name, "k": len(group_cols),
                     "random_auc": r["auc"], "blocked_auc": b["auc"],
                     "inflation": r["auc"] - b["auc"]})
    abl = pd.DataFrame(rows)
    abl.to_csv(paths.TABLES_DIR / "feature_ablation.csv", index=False)
    figures.fig_feature_ablation(abl)
    print("\n=== C. feature ablation ===")
    print(abl.round(3).to_string(index=False))

    # --- Substantive: survival and neighbour decay ------------------------
    surv = models.renewal_survival(ten, lab)
    surv.to_csv(paths.TABLES_DIR / "survival.csv", index=False)
    figures.fig_survival(surv)

    focal = (lab.loc[lab.is_cursed, "n_tenants"].mean()
             - lab.loc[~lab.is_cursed, "n_tenants"].mean())
    rows = []
    for radius in (50, 200, 500):
        res = models.permutation_null(lab, value_col=f"n_tenants_nbr_{radius}m",
                                      n_permutations=1000)
        rows.append({"radius_m": radius, "observed": res["observed_difference"],
                     "null_mean": res["null_mean"], "null_std": res["null_std"],
                     "p_value": res["p_value"], "focal_difference": focal})
    decay = pd.DataFrame(rows)
    decay.to_csv(paths.TABLES_DIR / "neighbour_decay.csv", index=False)
    figures.fig_neighbour_decay(decay)
    print("\n=== neighbour decay ===")
    print(decay.round(4).to_string(index=False))

    # --- Experiment D: mechanism, varying encoding-group size -------------
    from sklearn.cluster import MiniBatchKMeans

    val = lab["n_tenants"].astype(float).to_numpy()
    ok = ~np.isnan(xy).any(axis=1)
    rows = []
    for k in (10, 25, 50, 100, 250, 1000, 4000):
        groups = np.full(len(y), -1)
        groups[np.where(ok)[0]] = MiniBatchKMeans(
            n_clusters=k, random_state=0, n_init=3).fit(xy[ok]).labels_
        gs = pd.Series(groups)

        # The encoding under test: leave-one-out group mean of the outcome-
        # driving variable, supplied to the model as its only feature.
        total = gs.map(pd.Series(val).groupby(gs).sum())
        count = gs.map(gs.value_counts())
        xg = ((total - val) / (count - 1)).to_numpy().reshape(-1, 1)

        # Folds aligned with the encoding groups, which is the correct
        # partition for this feature.
        uniq = np.unique(groups[groups >= 0])
        perm = np.random.default_rng(0).permutation(len(uniq))
        amap = {u: i % 5 for u, i in zip(uniq, perm)}
        gfold = np.array([amap.get(v, -1) for v in groups])

        r = evaluate(gbm, rand, xg, y)
        b = evaluate(gbm, gfold, xg, y)
        rows.append({
            "n_groups": k,
            "mean_group_size": len(y) // k,
            # Between-group variance in label rate: the alternative explanation
            # the experiment is designed to rule out.
            "between_var": float(pd.Series(y).groupby(gs).mean().var()),
            "random_auc": r["auc"], "blocked_auc": b["auc"],
            "inflation": r["auc"] - b["auc"],
        })
    mech = pd.DataFrame(rows)
    mech.to_csv(paths.TABLES_DIR / "mechanism_group_size.csv", index=False)
    figures.fig_mechanism(mech)
    print("\n=== D. mechanism: leakage vs encoding-group size ===")
    print(mech.round(5).to_string(index=False))

    # --- Experiment E: are the leakage results definition-invariant? ------
    rows = []
    for definition in cursed.CurseDefinition:
        relabelled = cursed.label_cursed(
            pd.read_parquet(paths.PROCESSED_DIR / "features.parquet"), ten,
            config=cursed.CurseConfig(definition=definition),
        ).reset_index(drop=True)
        dcols = [c for c in features.feature_columns(relabelled) if "exposure" not in c]
        dy = relabelled["is_cursed"].astype(int).to_numpy()
        dx, _ = models._prepare_matrix(relabelled, dcols)
        r = evaluate(gbm, random_folds(dx, dy), dx, dy)
        b = evaluate(gbm, models.spatial_folds(relabelled), dx, dy)
        rows.append({"definition": definition.value, "n": len(dy),
                     "n_positive": int(dy.sum()), "base_rate": float(dy.mean()),
                     "random_auc": r["auc"], "blocked_auc": b["auc"],
                     "inflation": r["auc"] - b["auc"],
                     "random_ap": r["ap"], "blocked_ap": b["ap"]})
    inv = pd.DataFrame(rows)
    inv.to_csv(paths.TABLES_DIR / "definition_invariance.csv", index=False)
    print("\n=== E. leakage under each target definition ===")
    print(inv.round(3).to_string(index=False))

    # --- Supplement: robustness to the curse definition -------------------
    agree = cursed.definition_agreement(lab, ten)
    agree.to_csv(paths.TABLES_DIR / "definition_agreement.csv", index=False)
    print("\n=== supplement: definition agreement (Jaccard) ===")
    print(agree.pivot(index="definition_a", columns="definition_b",
                      values="jaccard").round(3).to_string())

    print(f"\ntables -> {paths.TABLES_DIR}\nfigures -> {paths.FIGURES_DIR}")


if __name__ == "__main__":
    main()
