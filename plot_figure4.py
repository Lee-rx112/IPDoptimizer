# plot_figure4_diagnostics.py
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# =========================
# Paths and basic settings
# =========================
ROOT = Path(r"experiment_results\ResNet")
IPD_DIR = ROOT / "AdamW_IPD_diff"
JSON_NAME = "metrics2.json"

OUT_DIR = Path("figures")
OUT_NAME = "figure4"

ACTIVE_EPOCHS = 20  # highlighted active window

# Keys to plot
METRIC_KEYS = {
    "rho_diff": {
        "title": "(a) Relative field strength",
        "ylabel": r"$\rho_{\mathrm{diff}}$",
    },
    "cos_F_vs_-g": {
        "title": r"(b) Alignment with $-g$",
        "ylabel": r"$\cos(F,-g)$",
    },
    "cos_geff_prev": {
        "title": "(c) Update-direction consistency",
        "ylabel": r"$\cos(g_{\mathrm{eff},t}, g_{\mathrm{eff},t-1})$",
    },
    "E_diff": {
        "title": "(d) Coulomb energy",
        "ylabel": r"$E_{\mathrm{diff}}$",
    },
}


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        return data

    if isinstance(data, dict):
        for key in ["history", "metrics", "records", "logs"]:
            if key in data and isinstance(data[key], list):
                return data[key]

        if all(k in data for k in METRIC_KEYS.keys()):
            n = min(len(data[k]) for k in METRIC_KEYS.keys())
            records = []
            for i in range(n):
                row = {k: data[k][i] for k in METRIC_KEYS.keys()}
                row["epoch"] = i + 1
                records.append(row)
            return records

    raise ValueError(f"Unsupported json format: {path}")


def to_float(x):
    if x is None:
        return np.nan
    if isinstance(x, str):
        x = x.strip().replace("%", "")
        if x.lower() in ["nan", "none", "null", ""]:
            return np.nan
    try:
        return float(x)
    except Exception:
        return np.nan


def extract_series(records):
    out = {}
    n = len(records)

    if n == 0:
        raise ValueError("Empty records")

    if "epoch" in records[0]:
        epochs = np.asarray([to_float(r["epoch"]) for r in records], dtype=float)
    else:
        epochs = np.arange(1, n + 1, dtype=float)

    out["epoch"] = epochs

    for key in METRIC_KEYS.keys():
        values = [to_float(r.get(key, np.nan)) for r in records]
        out[key] = np.asarray(values, dtype=float)

    return out


def load_experiment(exp_dir: Path, json_name: str):
    seed_dirs = sorted(
        [p for p in exp_dir.iterdir() if p.is_dir() and p.name.startswith("seed_")]
    )

    if not seed_dirs:
        raise FileNotFoundError(f"No seed_* folders found in {exp_dir}")

    result = {}
    for seed_dir in seed_dirs:
        json_path = seed_dir / json_name
        if not json_path.exists():
            raise FileNotFoundError(f"Missing file: {json_path}")

        records = load_json(json_path)
        result[seed_dir.name] = extract_series(records)

    return result


def stack_runs(data_dict, key):
    seeds = sorted(data_dict.keys())
    min_len = min(len(data_dict[s][key]) for s in seeds)

    arr = np.stack([data_dict[s][key][:min_len] for s in seeds], axis=0)
    epochs = data_dict[seeds[0]]["epoch"][:min_len]

    return epochs, arr, seeds


def nanmean_sem(arr):
    mean = np.nanmean(arr, axis=0)
    n_valid = np.sum(~np.isnan(arr), axis=0)
    std = np.nanstd(arr, axis=0, ddof=1)

    sem = np.full_like(mean, np.nan, dtype=float)
    valid = n_valid > 1
    sem[valid] = std[valid] / np.sqrt(n_valid[valid])
    sem[n_valid == 1] = 0.0

    return mean, sem


def plot_metric(ax, epochs, arr, title, ylabel, active_epochs=10):
    mean, sem = nanmean_sem(arr)

    # 1. 活跃窗口背景：固定使用极浅灰色，绝不喧宾夺主
    ax.axvspan(1, active_epochs, color='gray', alpha=0.08)

    # 2. 画主曲线，并获取分配到的颜色
    line, = ax.plot(epochs, mean, linewidth=2)

    # 3. 画方差阴影（std/sem）：颜色与折线自动保持一致
    ax.fill_between(epochs, mean - sem, mean + sem, color=line.get_color(), alpha=0.18)

    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    return mean, sem


def add_inset_for_ediff(ax, epochs, arr, active_epochs=10):
    """
    Add a lower-right inset to show the first 10 epochs of E_diff.
    """
    mean, sem = nanmean_sem(arr)

    valid = ~np.isnan(mean)
    x = epochs[valid]
    y = mean[valid]
    y_sem = sem[valid]

    mask20 = x <= active_epochs
    if np.sum(mask20) == 0:
        return

    x20 = x[mask20]
    y20 = y[mask20]
    y20_sem = y_sem[mask20]

    # [x0, y0, width, height]
    axins = ax.inset_axes([0.48, 0.18, 0.45, 0.4])

    # 放大图里的活跃背景框（灰色）
    axins.axvspan(1, active_epochs, color='gray', alpha=0.08)

    # 放大图的曲线和方差阴影（自动取色）
    line, = axins.plot(x20, y20, linewidth=1.8)
    axins.fill_between(x20, y20 - y20_sem, y20 + y20_sem, color=line.get_color(), alpha=0.18)

    axins.set_xlim(x20.min(), x20.max())

    y_min = np.nanmin(y20 - y20_sem)
    y_max = np.nanmax(y20 + y20_sem)
    pad = 0.08 * (y_max - y_min + 1e-12)
    axins.set_ylim(y_min - pad, y_max + pad)

    axins.set_xlabel("Epoch", fontsize=8)
    axins.set_ylabel(r"$E_{\mathrm{coul}}$", fontsize=8)

    axins.tick_params(axis="both", labelsize=7)
    axins.grid(True, alpha=0.25, linestyle='--')


def main():
    ipd = load_experiment(IPD_DIR, JSON_NAME)

    fig, axes = plt.subplots(2, 2, figsize=(10.8, 7.4))
    axes = axes.flatten()

    for ax, key in zip(axes[:3], list(METRIC_KEYS.keys())[:3]):
        epochs, arr, seeds = stack_runs(ipd, key)
        plot_metric(
            ax=ax,
            epochs=epochs,
            arr=arr,
            title=METRIC_KEYS[key]["title"],
            ylabel=METRIC_KEYS[key]["ylabel"],
            active_epochs=ACTIVE_EPOCHS
        )

    key = "E_diff"
    ax = axes[3]
    epochs, arr, seeds = stack_runs(ipd, key)
    plot_metric(
        ax=ax,
        epochs=epochs,
        arr=arr,
        title=METRIC_KEYS[key]["title"],
        ylabel=METRIC_KEYS[key]["ylabel"],
        active_epochs=ACTIVE_EPOCHS
    )
    add_inset_for_ediff(
        ax=ax,
        epochs=epochs,
        arr=arr,
        active_epochs=ACTIVE_EPOCHS,
    )

    fig.tight_layout()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pdf_path = OUT_DIR / f"{OUT_NAME}.pdf"
    png_path = OUT_DIR / f"{OUT_NAME}.png"

    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")

    print(f"Saved: {pdf_path}")
    print(f"Saved: {png_path}")
    print(f"Seeds ({len(seeds)}): {', '.join(seeds)}")


if __name__ == "__main__":
    main()