"""Figures for the paper.

Every figure is generated from the saved result tables rather than recomputed,
so the numbers in the plots and the numbers in the text cannot drift apart.

Styling is deliberately plain: greyscale-safe, no chartjunk, and readable at the
size a two-column preprint actually prints them.
"""

from __future__ import annotations

import logging

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import paths

log = logging.getLogger(__name__)

# Colour-blind-safe, and distinguishable when printed in greyscale.
C_RANDOM = "#c1272d"
C_BLOCKED = "#0b5394"
C_NEUTRAL = "#666666"

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "legend.frameon": False,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def _save(fig: plt.Figure, name: str) -> None:
    paths.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(paths.FIGURES_DIR / f"{name}.{ext}")
    plt.close(fig)
    log.info("wrote %s", paths.FIGURES_DIR / f"{name}.pdf")


def fig_block_scale(scale: pd.DataFrame, *, feature_support_m: float = 500.0) -> None:
    """The paper's central figure: performance against blocking scale.

    Expects columns ``label``, ``block_m`` (NaN for the random baseline),
    ``auc``, ``auc_std``, ``ap``, ``base_rate``.
    """
    grid = scale.loc[scale["block_m"].notna()].sort_values("block_m")
    rand = scale.loc[scale["block_m"].isna()]
    admin = scale.loc[scale["label"].str.contains("area", case=False, na=False)]

    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.1), constrained_layout=True)

    for ax, (col, err, ylabel) in zip(
        axes,
        [("auc", "auc_std", "ROC AUC"), ("ap", None, "Average precision")],
    ):
        x = grid["block_m"].to_numpy()
        y = grid[col].to_numpy()
        if err and err in grid:
            ax.errorbar(x, y, yerr=grid[err].to_numpy(), fmt="o-",
                        color=C_BLOCKED, capsize=2.5, lw=1.4, ms=4,
                        label="spatial grid blocks")
        else:
            ax.plot(x, y, "o-", color=C_BLOCKED, lw=1.4, ms=4,
                    label="spatial grid blocks")

        if not rand.empty:
            ax.axhline(rand[col].iloc[0], color=C_RANDOM, ls="--", lw=1.3,
                       label="random $k$-fold")
        if not admin.empty:
            ax.axhline(admin[col].iloc[0], color="#1a7a3c", ls=":", lw=1.4,
                       label="community area")

        # The scale at which features stop carrying information across blocks.
        # Placed at the top of the axes so it cannot collide with the legend,
        # which sits low-left where the curve leaves space.
        ax.axvline(feature_support_m, color=C_NEUTRAL, lw=1.0, alpha=0.7)
        ax.annotate("feature support", xy=(feature_support_m, 0.97),
                    xytext=(3, 0), textcoords="offset points",
                    xycoords=("data", "axes fraction"),
                    fontsize=7, color=C_NEUTRAL, ha="left", va="top", rotation=90)

        ax.set_xscale("log")
        ax.set_xlabel("spatial block edge length (m, log scale)")
        ax.set_ylabel(ylabel)
        ax.margins(x=0.08)

    axes[1].set_yscale("log")
    axes[0].legend(loc="lower left", fontsize=7.5)
    fig.suptitle("Blocking below the spatial support of the features does not "
                 "reduce leakage", fontsize=10)
    _save(fig, "fig1_block_scale")


def fig_model_families(models_df: pd.DataFrame) -> None:
    """Inflation by model family, paired random vs blocked."""
    df = models_df.sort_values("inflation")
    y = np.arange(len(df))

    fig, ax = plt.subplots(figsize=(5.6, 2.6))
    ax.hlines(y, df["blocked_auc"], df["random_auc"], color=C_NEUTRAL, lw=1.2, zorder=1)
    ax.scatter(df["random_auc"], y, s=34, color=C_RANDOM, zorder=2, label="random $k$-fold")
    ax.scatter(df["blocked_auc"], y, s=34, color=C_BLOCKED, zorder=2, label="spatially blocked")

    for yi, row in zip(y, df.itertuples()):
        ax.annotate(f"+{row.inflation:.3f}",
                    xy=((row.random_auc + row.blocked_auc) / 2, yi + 0.22),
                    ha="center", fontsize=7, color=C_NEUTRAL)

    ax.set_yticks(y)
    ax.set_yticklabels(df["model"])
    ax.set_xlabel("ROC AUC")
    ax.set_xlim(0.55, 1.02)
    ax.legend(loc="lower right", fontsize=7.5)
    ax.set_title("Inflation grows with model capacity; honest scores are equal")
    _save(fig, "fig2_model_families")


def fig_feature_ablation(ablation: pd.DataFrame) -> None:
    """Which feature families carry the leak."""
    df = ablation.sort_values("inflation")
    y = np.arange(len(df))

    fig, ax = plt.subplots(figsize=(5.8, 2.6))
    ax.barh(y, df["inflation"], color=C_BLOCKED, height=0.6)
    for yi, row in zip(y, df.itertuples()):
        ax.annotate(f"{row.inflation:+.3f}", xy=(row.inflation, yi),
                    xytext=(4, 0), textcoords="offset points",
                    va="center", fontsize=7.5)

    ax.set_yticks(y)
    ax.set_yticklabels(df["family"])
    ax.set_xlabel("AUC inflation (random $-$ blocked)")
    ax.set_xlim(0, max(df["inflation"]) * 1.25)
    ax.set_title("Administrative aggregates dominate the leak")
    _save(fig, "fig3_feature_ablation")


def fig_survival(surv: pd.DataFrame, *, label_col: str = "is_cursed") -> None:
    """Discrete-time survival and hazard over renewal cycles."""
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8))

    styles = {True: (C_RANDOM, "high-turnover locations"),
              False: (C_BLOCKED, "all other locations")}

    for flag, grp in surv.groupby(label_col):
        colour, name = styles[bool(flag)]
        g = grp.sort_values("cycle")
        axes[0].plot(g["cycle"], g["survival"], "o-", color=colour, lw=1.4, ms=3.5, label=name)
        axes[1].plot(g["cycle"], g["hazard"], "o-", color=colour, lw=1.4, ms=3.5, label=name)

    axes[0].set_ylabel("share surviving")
    axes[0].set_title("Survival")
    axes[1].set_ylabel("hazard")
    axes[1].set_title("Per-cycle hazard")
    for ax in axes:
        ax.set_xlabel("licence renewal cycle")
    axes[0].legend(fontsize=7.5)
    fig.suptitle("Tenancies at high-turnover addresses end sooner at every cycle",
                 fontsize=10, y=1.05)
    _save(fig, "fig4_survival")


def fig_neighbour_decay(decay: pd.DataFrame) -> None:
    """Focal effect against neighbour effect by radius."""
    fig, ax = plt.subplots(figsize=(5.2, 2.6))

    ax.plot(decay["radius_m"], decay["observed"], "o-", color=C_BLOCKED,
            lw=1.5, ms=4.5, label="observed difference")
    ax.plot(decay["radius_m"], decay["null_mean"], "s--", color=C_NEUTRAL,
            lw=1.2, ms=3.5, label="within-area permutation null")
    ax.fill_between(
        decay["radius_m"],
        decay["null_mean"] - 2 * decay["null_std"],
        decay["null_mean"] + 2 * decay["null_std"],
        color=C_NEUTRAL, alpha=0.18, lw=0,
    )

    focal = decay["focal_difference"].iloc[0]
    ax.axhline(focal, color=C_RANDOM, ls="-.", lw=1.3)
    ax.annotate(f"focal location: {focal:+.2f} tenants",
                xy=(decay["radius_m"].iloc[-1], focal), xytext=(-4, -11),
                textcoords="offset points", ha="right", fontsize=7.5, color=C_RANDOM)

    ax.set_xlabel("neighbourhood radius (m)")
    ax.set_ylabel("difference in mean tenants")
    ax.set_title("The effect is concentrated at the address, not the street")
    ax.legend(fontsize=7.5, loc="center right")
    _save(fig, "fig5_neighbour_decay")
