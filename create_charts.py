"""Create portfolio figures without retraining.

Usage: python create_charts.py --predictions outputs/test_predictions.csv
Requires pandas, numpy and matplotlib (plotting only).
CV scores are recorded results from the project's five-fold grouped CV run.
"""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, default=Path("outputs/test_predictions.csv"))
    parser.add_argument("--outdir", type=Path, default=Path("docs/images"))
    args = parser.parse_args()
    df = pd.read_csv(args.predictions)
    values = df[["actual_salary", "predicted_salary"]].to_numpy(dtype=float)
    if len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("Expected at least two complete, finite prediction rows.")
    actual, predicted = values.T
    errors = predicted - actual
    mae = np.abs(errors).mean()
    args.outdir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "text.parse_math": False,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.labelcolor": "#333333", "text.color": "#181818",
                         "axes.edgecolor": "#bbbbbb", "savefig.facecolor": "white"})
    money = FuncFormatter(lambda x, _: f"${x / 1000:,.0f}k")

    fig, ax = plt.subplots(figsize=(8.5, 8.5))
    fig.subplots_adjust(left=.14, right=.95, bottom=.19, top=.80)
    fig.text(.14, .94, "Where the salary model misses", fontsize=21, weight="bold")
    fig.text(.14, .895, f"Ridge predictions on {len(df)} held-out listings", fontsize=13, color="#555555")
    fig.text(.14, .85, f"MAE  ${mae:,.0f}    |    Median absolute error  ${np.median(np.abs(errors)):,.0f}", fontsize=11)
    limit = np.ceil(max(actual.max(), predicted.max()) / 50000) * 50000
    ax.plot([0, limit], [0, limit], color="#999999", linestyle="--", linewidth=1.3, label="Perfect prediction")
    ax.scatter(actual, predicted, s=43, color="#202020", alpha=.68, edgecolors="white", linewidth=.5)
    ax.set(xlim=(0, limit), ylim=(0, limit), xlabel="Listed annual salary midpoint (USD)", ylabel="Predicted annual salary midpoint (USD)")
    ax.set_aspect("equal", adjustable="box")
    ax.xaxis.set_major_formatter(money)
    ax.yaxis.set_major_formatter(money)
    ax.set_axisbelow(True)
    ax.grid(alpha=.20)
    ax.legend(loc="upper left", frameon=False, fontsize=10)
    fig.text(.14, .105, "Below the dashed line = underprediction; above it = overprediction.", fontsize=10)
    fig.text(.14, .065, "Source: saved test_predictions.csv. Historical US listing estimates, not verified earnings.\nNo retraining. MAE describes average error, not an individual prediction interval.", fontsize=9, color="#666666", linespacing=1.6)
    fig.savefig(args.outdir / "actual_vs_predicted.png", dpi=180)
    plt.close(fig)

    names = ["Ridge", "Random Forest", "Median baseline"]
    scores = [18845, 19261, 27111]
    fig, ax = plt.subplots(figsize=(10, 5.6))
    fig.subplots_adjust(left=.23, right=.94, top=.72, bottom=.25)
    fig.text(.06, .92, "Ridge narrowly leads cross-validation", fontsize=22, weight="bold")
    fig.text(.06, .855, "Mean absolute error (USD) · lower is better", fontsize=13, color="#555555")
    bars = ax.barh(names, scores, color=["#202020", "#999999", "#d4d4d4"], height=.58)
    ax.invert_yaxis()
    ax.set_xlim(0, 32000)
    ax.xaxis.set_major_formatter(money)
    ax.set_xticks([0, 10000, 20000, 30000])
    ax.set_axisbelow(True)
    ax.grid(axis="x", alpha=.20)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0, pad=12)
    for bar, score in zip(bars, scores):
        ax.text(score + 450, bar.get_y() + bar.get_height()/2, f"${score:,}", va="center", weight="bold")
    fig.text(.06, .15, "Five-fold GroupKFold on 365 training listings; repeated descriptions kept together.", fontsize=10)
    fig.text(.06, .075, "Source: recorded project CV results, rounded to USD. These are not held-out test scores.\nThe $416 gap between Ridge and Random Forest does not establish a statistically reliable advantage.", fontsize=9, color="#666666", linespacing=1.6)
    fig.savefig(args.outdir / "model_comparison.png", dpi=180)
    plt.close(fig)
    print(f"Created two charts in {args.outdir.resolve()}")


if __name__ == "__main__":
    main()
