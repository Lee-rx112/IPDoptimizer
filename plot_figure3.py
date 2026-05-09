# plot_figure3_resnet_v5.py
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# 路径与配置
ROOT = Path(r"experiment_results\Densenet")
BASELINE_DIR = ROOT / "AdamW_Pure"
IPD_DIR = ROOT / "AdamW_IPD_coul"
BASELINE_JSON = "metrics1.json"
IPD_JSON = "metrics1.json"
OUT_DIR = Path("figures")
OUT_NAME = "figureD7"

# 数据键名
TRAIN_LOSS_KEY = "train_loss"
TEST_ACC_KEY = "test_acc"
TIME_KEY = "epoch_time_sec"


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list): return data
    if isinstance(data, dict):
        for key in ["history", "metrics", "records", "logs"]:
            if key in data and isinstance(data[key], list): return data[key]
        if TRAIN_LOSS_KEY in data and TEST_ACC_KEY in data:
            n = min(len(data[TRAIN_LOSS_KEY]), len(data[TEST_ACC_KEY]))
            records = []
            for i in range(n):
                record = {TRAIN_LOSS_KEY: data[TRAIN_LOSS_KEY][i],
                          TEST_ACC_KEY: data[TEST_ACC_KEY][i], "epoch": i + 1}
                if TIME_KEY in data: record[TIME_KEY] = data[TIME_KEY][i]
                records.append(record)
            return records
    raise ValueError(f"Unsupported json format: {path}")


def extract_series(records):
    train_loss = np.asarray([float(r[TRAIN_LOSS_KEY]) for r in records], dtype=float)
    test_acc = np.asarray([float(r[TEST_ACC_KEY]) for r in records], dtype=float)
    time_sec = np.asarray([float(r.get(TIME_KEY, 0)) for r in records], dtype=float)
    if np.nanmax(test_acc) <= 1.5: test_acc = test_acc * 100.0
    epochs = np.asarray([float(r.get("epoch", i + 1)) for i, r in enumerate(records)], dtype=float)
    return epochs, train_loss, test_acc, time_sec


def load_experiment(exp_dir: Path, json_name: str):
    seed_dirs = sorted([p for p in exp_dir.iterdir() if p.is_dir() and p.name.startswith("seed_")])
    result = {}
    for seed_dir in seed_dirs:
        json_path = seed_dir / json_name
        if not json_path.exists(): continue
        records = load_json(json_path)
        epochs, train_loss, test_acc, time_sec = extract_series(records)
        result[seed_dir.name] = {"epoch": epochs, "train_loss": train_loss, "test_acc": test_acc, "time": time_sec}
    return result


def stack_matched_runs(baseline, ipd, key: str, exclude_seeds=None):
    if exclude_seeds is None: exclude_seeds = []
    common_seeds = sorted([s for s in (set(baseline.keys()) & set(ipd.keys())) if s not in exclude_seeds])
    min_len = min(min(len(baseline[s][key]), len(ipd[s][key])) for s in common_seeds)
    base_arr = np.stack([baseline[s][key][:min_len] for s in common_seeds], axis=0)
    ipd_arr = np.stack([ipd[s][key][:min_len] for s in common_seeds], axis=0)
    epochs = baseline[common_seeds[0]]["epoch"][:min_len]
    return epochs, base_arr, ipd_arr, common_seeds


def mean_sem(arr):
    mean = arr.mean(axis=0)
    sem = (arr.std(axis=0, ddof=1) / np.sqrt(arr.shape[0])) if arr.shape[0] > 1 else np.zeros_like(mean)
    return mean, sem


def plot_curve_with_sem(ax, x, arr, label, color=None):
    mean, sem = mean_sem(arr)
    line, = ax.plot(x, mean, linewidth=2, label=label, color=color)
    ax.fill_between(x, mean - sem, mean + sem, alpha=0.18, color=line.get_color())


def main():
    baseline = load_experiment(BASELINE_DIR, BASELINE_JSON)
    ipd = load_experiment(IPD_DIR, IPD_JSON)

    epochs, base_loss, ipd_loss, _ = stack_matched_runs(baseline, ipd, "train_loss")
    _, base_acc, ipd_acc, _ = stack_matched_runs(baseline, ipd, "test_acc")
    _, _, ipd_time, seeds_time = stack_matched_runs(baseline, ipd, "time", exclude_seeds=["seed_42","seed_46"])

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes_flat = axes.flatten()

    for i, ax in enumerate(axes_flat):
        ax.axvspan(1, 10, color='gray', alpha=0.15, label='Active Window' if i == 0 else "")
        ax.grid(True, alpha=0.2, linestyle='--')
        ax.set_xlabel("Epoch")

    # (a) Training loss
    plot_curve_with_sem(axes_flat[0], epochs, base_loss, "AdamW")
    plot_curve_with_sem(axes_flat[0], epochs, ipd_loss, "AdamW + IPD-Coul")
    axes_flat[0].set_ylabel("Training Loss")
    axes_flat[0].set_title("Training Convergence")
    axes_flat[0].legend(frameon=False)

    # (b) Test accuracy + Adjusted Inset
    ax_acc = axes_flat[1]
    plot_curve_with_sem(ax_acc, epochs, base_acc, "AdamW")
    plot_curve_with_sem(ax_acc, epochs, ipd_acc, "AdamW + IPD-Coul")
    ax_acc.set_ylabel("Test Accuracy (%)")
    ax_acc.set_title("Generalization Performance")
    ax_acc.legend(frameon=False, loc='upper left')

    # 调整位置：向左移 0.04，向上移 0.06 -> [x0, y0, width, height]
    ax_ins = ax_acc.inset_axes([0.48, 0.18, 0.45, 0.4])
    plot_curve_with_sem(ax_ins, epochs, base_acc, "AdamW")
    plot_curve_with_sem(ax_ins, epochs, ipd_acc, "AdamW + IPD-Coul")

    # 限制范围：最后 30 个 epoch
    ax_ins.set_xlim(epochs[-30], epochs[-1])
    tail_acc = np.concatenate([base_acc[:, -30:], ipd_acc[:, -30:]])
    y_min, y_max = np.min(tail_acc), np.max(tail_acc)
    ax_ins.set_ylim(y_min - 0.2, y_max + 0.2)

    # 添加轴名称和样式
    ax_ins.set_xlabel("Epoch", fontsize=8)
    ax_ins.set_ylabel("Acc (%)", fontsize=8)
    ax_ins.tick_params(labelsize=7)
    ax_ins.grid(True, alpha=0.15, linestyle='--')

    # (c) Accuracy difference
    acc_delta = ipd_acc - base_acc
    delta_mean, delta_sem = mean_sem(acc_delta)
    axes_flat[2].axhline(0.0, color='black', linestyle="--", linewidth=0.8, alpha=0.5)
    axes_flat[2].plot(epochs, delta_mean, color='C2', linewidth=2, label="$\Delta$ Accuracy")
    axes_flat[2].fill_between(epochs, delta_mean - delta_sem, delta_mean + delta_sem, color='C2', alpha=0.18)
    axes_flat[2].set_ylabel("Paired Difference (%)")
    axes_flat[2].set_title("Incremental Gain")
    axes_flat[2].legend(frameon=False)

    # (d) Computational Time (Filtered)
    plot_curve_with_sem(axes_flat[3], epochs, ipd_time, "AdamW + IPD-Coul", color='C1')
    axes_flat[3].set_ylabel("Time per Epoch (sec)")
    axes_flat[3].set_title(f"Computational Profile")
    axes_flat[3].legend(frameon=False)

    plt.tight_layout()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / f"{OUT_NAME}.png", dpi=300, bbox_inches="tight")
    print(f"Figure saved. Inset plot adjusted and labeled.")


if __name__ == "__main__":
    main()