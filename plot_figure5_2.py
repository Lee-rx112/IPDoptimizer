#
import re
import json
import copy
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.models import resnet50
import matplotlib.pyplot as plt


# ============================================================
# Configuration
# ============================================================
CONFIG = {
    # checkpoint paths
    "checkpoint_root": Path("./experiment_results/Checkpoints/Resnet"),
    "baseline_run1": "AdamW_Pure",
    "baseline_run2":"SGD_Pure",
    "ipd_run": "AdamW_IPD_diff",
    "seed": 45,

    # output
    "out_dir": Path("figures"),
    "out_name": "figureD1",

    # model / data
    "num_classes": 100,
    "data_root": "./data",
    "device": "cuda" if torch.cuda.is_available() else "cpu",

    # landscape evaluation
    "test_subset_size": 2048,
    "landscape_batch_size": 256,

    # endpoint plane only
    "basis_method": "endpoint",
    # "basis_method":"pca",

    # grid
    "grid_size": 71,
    "margin_ratio": 0.06,

    # only show/evaluate loss near trajectories
    "use_trajectory_tube": True,
    "tube_radius_ratio": 0.055,
    "tube_radius_min": 20.0,

    # color clipping for visualization
    "clip_loss_percentile": 95,

    # BN handling:
    # "fixed": model.eval(), use BN buffers from reference checkpoint
    # "batch": model.train(), use batch statistics during loss evaluation
    #          recommended for this visualization with ResNet/BN
    "bn_mode": "batch",

    # cache
    "use_cached_landscape": False,

    # plot
    "dpi": 300,
}


# ============================================================
# Model and data
# ============================================================
def build_model(num_classes=100):
    model = resnet50(weights=None, num_classes=num_classes)
    model.conv1 = nn.Conv2d(
        3, 64, kernel_size=3, stride=1, padding=1, bias=False
    )
    model.maxpool = nn.Identity()
    return model


def build_test_subset_loader():
    mean = (0.5071, 0.4867, 0.4408)
    std = (0.2675, 0.2565, 0.2761)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    test_set = datasets.CIFAR100(
        root=CONFIG["data_root"],
        train=False,
        download=True,
        transform=transform,
    )

    rng = np.random.RandomState(CONFIG["seed"])
    n = min(CONFIG["test_subset_size"], len(test_set))
    idx = rng.choice(np.arange(len(test_set)), size=n, replace=False)
    idx = sorted(idx.tolist())

    subset = Subset(test_set, idx)
    loader = DataLoader(
        subset,
        batch_size=CONFIG["landscape_batch_size"],
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    return loader, idx


# ============================================================
# Checkpoint handling
# ============================================================
def get_run_dir(run_name):
    return CONFIG["checkpoint_root"] / run_name / f"seed_{CONFIG['seed']}"


def extract_epoch_from_name(path: Path):
    if path.stem == "final":
        return None
    m = re.search(r"epoch_(\d+)", path.stem)
    if m is None:
        return None
    return int(m.group(1))


def collect_epoch_checkpoints(run_dir):
    files = []
    for p in sorted(run_dir.glob("epoch_*.pt")):
        ep = extract_epoch_from_name(p)
        if ep is not None:
            files.append((ep, p))

    files = sorted(files, key=lambda x: x[0])

    if len(files) == 0:
        raise FileNotFoundError(f"No epoch_*.pt checkpoints found in {run_dir}")

    return files


def find_final_checkpoint(run_dir):
    final_path = run_dir / "final.pt"
    if final_path.exists():
        return final_path
    ckpts = collect_epoch_checkpoints(run_dir)
    return ckpts[-1][1]


def load_ckpt(path, map_location="cpu"):
    ckpt = torch.load(path, map_location=map_location)

    if isinstance(ckpt, dict):
        if "model_state_dict" in ckpt:
            return ckpt["model_state_dict"], ckpt.get("epoch", None)
        if "model_state" in ckpt:
            return ckpt["model_state"], ckpt.get("epoch", None)

        # raw state_dict
        if all(torch.is_tensor(v) for v in ckpt.values()):
            return ckpt, None

    raise ValueError(f"Cannot parse checkpoint: {path}")


def strip_module_prefix_if_needed(state_dict):
    keys = list(state_dict.keys())
    if len(keys) > 0 and all(k.startswith("module.") for k in keys):
        return {k.replace("module.", "", 1): v for k, v in state_dict.items()}
    return state_dict


# ============================================================
# Vectorization
# ============================================================
def get_param_names(model):
    return [name for name, _ in model.named_parameters()]


def state_dict_to_param_vector(state_dict, param_names):
    state_dict = strip_module_prefix_if_needed(state_dict)
    vecs = []
    for name in param_names:
        if name not in state_dict:
            raise KeyError(f"Parameter {name} not found in checkpoint.")
        vecs.append(state_dict[name].detach().float().reshape(-1).cpu())
    return torch.cat(vecs, dim=0)


def load_vector_to_model(model, vec, reference_state_dict, param_names):
    """
    Load parameter vector into model.
    Non-parameter buffers such as BN running statistics are taken from reference_state_dict.
    """
    state = copy.deepcopy(strip_module_prefix_if_needed(reference_state_dict))

    pointer = 0
    for name in param_names:
        shape = state[name].shape
        numel = state[name].numel()
        state[name] = vec[pointer:pointer + numel].view(shape).to(state[name].dtype)
        pointer += numel

    if pointer != vec.numel():
        raise RuntimeError("Vector length does not match model parameters.")

    model.load_state_dict(state, strict=True)


def load_state_to_model(model, state_dict):
    state_dict = strip_module_prefix_if_needed(state_dict)
    model.load_state_dict(state_dict, strict=True)


# ============================================================
# Basis construction
# ============================================================
def normalize(v, eps=1e-12):
    n = torch.norm(v)
    if n < eps:
        raise ValueError("Cannot normalize a near-zero vector.")
    return v / n


def build_endpoint_basis(theta0, theta_base_final, theta_ipd_final):
    """
    Endpoint-defined common 2D plane.

    e1: direction from common initialization to AdamW final.
    e2: orthogonalized direction from common initialization to IPD final.
    """
    v1 = theta_base_final - theta0
    e1 = normalize(v1)


    v2_raw = theta_ipd_final - theta0
    v2 = v2_raw - torch.dot(v2_raw, e1) * e1
    e2 = normalize(v2)

    return e1, e2


def project(theta, theta0, e1, e2):
    d = theta - theta0
    x = torch.dot(d, e1).item()
    y = torch.dot(d, e2).item()
    return x, y


def load_trajectory_vectors(ckpts, param_names):
    epochs = []
    vectors = []
    for ep, p in ckpts:
        state, _ = load_ckpt(p, map_location="cpu")
        vec = state_dict_to_param_vector(state, param_names)
        epochs.append(ep)
        vectors.append(vec)
    return np.asarray(epochs, dtype=np.int32), vectors


def project_trajectory(vectors, epochs, theta0, e1, e2):
    coords = []
    for v in vectors:
        coords.append(project(v, theta0, e1, e2))
    return epochs, np.asarray(coords, dtype=np.float32)


# ============================================================
# Loss evaluation
# ============================================================
@torch.no_grad()
def evaluate_loss(model, loader, device, bn_mode="batch"):
    """
    bn_mode:
      - fixed: use model.eval() and checkpoint BN running stats.
      - batch: use model.train() so BN uses current batch statistics.
               This is often more stable for loss-landscape visualization
               when weights are moved away from the reference checkpoint.
    """
    criterion = nn.CrossEntropyLoss(reduction="sum")

    if bn_mode == "fixed":
        model.eval()
    elif bn_mode == "batch":
        model.train()
    else:
        raise ValueError("bn_mode must be 'fixed' or 'batch'.")

    total_loss = 0.0
    total_num = 0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        logits = model(x)
        loss = criterion(logits, y)

        total_loss += loss.item()
        total_num += y.size(0)

    return total_loss / total_num


def eval_full_state_loss(state_dict, loader, device, label):
    model = build_model(CONFIG["num_classes"]).to(device)
    load_state_to_model(model, state_dict)
    loss_fixed = evaluate_loss(model, loader, device, bn_mode="fixed")

    model = build_model(CONFIG["num_classes"]).to(device)
    load_state_to_model(model, state_dict)
    loss_batch = evaluate_loss(model, loader, device, bn_mode="batch")

    print(f"[Sanity] {label}: loss(eval BN)={loss_fixed:.4f}, loss(batch BN)={loss_batch:.4f}")


def eval_vector_loss(vec, reference_state, param_names, loader, device, label):
    model = build_model(CONFIG["num_classes"]).to(device)
    load_vector_to_model(model, vec, reference_state, param_names)
    loss = evaluate_loss(model, loader, device, bn_mode=CONFIG["bn_mode"])
    print(f"[Sanity] {label}: vector loss ({CONFIG['bn_mode']} BN)={loss:.4f}")


# ============================================================
# Grid and trajectory-neighborhood mask
# ============================================================
def compute_grid_range(base_coords, ipd_coords, margin_ratio):
    all_xy = np.concatenate([base_coords, ipd_coords], axis=0)

    x_min, y_min = all_xy.min(axis=0)
    x_max, y_max = all_xy.max(axis=0)

    x_margin = max((x_max - x_min) * margin_ratio, 1e-6)
    y_margin = max((y_max - y_min) * margin_ratio, 1e-6)

    return (
        x_min - x_margin,
        x_max + x_margin,
        y_min - y_margin,
        y_max + y_margin,
    )


def compute_tube_radius(base_coords, ipd_coords):
    all_xy = np.concatenate([base_coords, ipd_coords], axis=0)
    span = max(
        float(all_xy[:, 0].max() - all_xy[:, 0].min()),
        float(all_xy[:, 1].max() - all_xy[:, 1].min()),
    )
    return max(CONFIG["tube_radius_min"], CONFIG["tube_radius_ratio"] * span)


def min_distance_to_polyline(points, polyline):
    """
    points: [N, 2]
    polyline: [M, 2]
    Return min distance from each point to any segment in the polyline.
    """
    points = np.asarray(points, dtype=np.float64)
    polyline = np.asarray(polyline, dtype=np.float64)

    if len(polyline) < 2:
        return np.linalg.norm(points - polyline[0][None, :], axis=1)

    min_dist = np.full(points.shape[0], np.inf, dtype=np.float64)

    for i in range(len(polyline) - 1):
        a = polyline[i]
        b = polyline[i + 1]
        ab = b - a
        ab2 = np.dot(ab, ab)

        if ab2 < 1e-12:
            dist = np.linalg.norm(points - a[None, :], axis=1)
        else:
            ap = points - a[None, :]
            t = np.clip((ap @ ab) / ab2, 0.0, 1.0)
            proj = a[None, :] + t[:, None] * ab[None, :]
            dist = np.linalg.norm(points - proj, axis=1)

        min_dist = np.minimum(min_dist, dist)

    return min_dist


def build_trajectory_tube_mask(xs, ys, base_coords, ipd_coords, tube_radius):
    X, Y = np.meshgrid(xs, ys)
    pts = np.stack([X.reshape(-1), Y.reshape(-1)], axis=1)

    d_base = min_distance_to_polyline(pts, base_coords)
    d_ipd = min_distance_to_polyline(pts, ipd_coords)

    d = np.minimum(d_base, d_ipd)
    mask = (d <= tube_radius).reshape(len(ys), len(xs))
    return mask


# ============================================================
# Landscape computation
# ============================================================
def compute_loss_landscape(
    model,
    param_names,
    reference_state_dict,
    theta0,
    e1,
    e2,
    test_loader,
    x_range,
    y_range,
    mask=None,
):
    device = CONFIG["device"]
    grid_size = CONFIG["grid_size"]

    xs = np.linspace(x_range[0], x_range[1], grid_size)
    ys = np.linspace(y_range[0], y_range[1], grid_size)

    Z = np.full((grid_size, grid_size), np.nan, dtype=np.float32)

    model = model.to(device)

    if mask is None:
        mask = np.ones((grid_size, grid_size), dtype=bool)

    total = int(mask.sum())
    count = 0

    print("\nComputing loss landscape on fixed test subset...")
    print(f"Grid size: {grid_size} x {grid_size}; evaluated points: {total}/{grid_size * grid_size}")
    print(f"BN mode for landscape: {CONFIG['bn_mode']}")

    for iy, y in enumerate(ys):
        for ix, x in enumerate(xs):
            if not mask[iy, ix]:
                continue

            theta = theta0 + float(x) * e1 + float(y) * e2

            load_vector_to_model(
                model=model,
                vec=theta,
                reference_state_dict=reference_state_dict,
                param_names=param_names,
            )

            loss = evaluate_loss(model, test_loader, device, bn_mode=CONFIG["bn_mode"])
            Z[iy, ix] = loss

            count += 1
            if count % 20 == 0 or count == total:
                print(f"  progress: {count}/{total}")

    return xs, ys, Z


# ============================================================
# Plotting
# ============================================================
def annotate_epochs(ax, epochs, coords, color, label_set=None):
    if label_set is None:
        label_set = {0, 1, 5, 10, 20, 50, 100, 150, 200}

    for ep, (x, y) in zip(epochs, coords):
        if int(ep) in label_set:
            ax.text(x, y, str(int(ep)), fontsize=8, color=color)


def plot_landscape(xs, ys, Z, base_epochs, base_coords, ipd_epochs, ipd_coords, out_path):
    X, Y = np.meshgrid(xs, ys)

    finite = np.isfinite(Z)
    if finite.sum() == 0:
        raise RuntimeError("No finite loss values to plot.")

    z_valid = Z[finite]
    vmax = np.nanpercentile(z_valid, CONFIG["clip_loss_percentile"])
    vmin = np.nanpercentile(z_valid, 5)

    if vmax <= vmin:
        vmax = np.nanmax(z_valid)
        vmin = np.nanmin(z_valid)

    Z_clip = np.clip(Z, vmin, vmax)
    Z_masked = np.ma.masked_invalid(Z_clip)

    fig, ax = plt.subplots(figsize=(7.4, 6.2))

    levels = np.linspace(vmin, vmax, 28)
    cf = ax.contourf(X, Y, Z_masked, levels=levels, cmap="viridis", extend="max")
    cbar = fig.colorbar(cf, ax=ax)
    cbar.set_label("Cross-entropy loss on test subset")

    try:
        ax.contour(
            X, Y, Z_masked,
            levels=np.linspace(vmin, vmax, 12),
            colors="white",
            linewidths=0.45,
            alpha=0.45,
        )
    except Exception:
        pass

    ax.plot(
        base_coords[:, 0], base_coords[:, 1],
        marker="o", markersize=3, linewidth=2.0,
        label="AdamW",
    )

    ax.plot(
        ipd_coords[:, 0], ipd_coords[:, 1],
        marker="o", markersize=3, linewidth=2.0,
        label="AdamW + IPD-Diff",
    )

    # Common initialization
    ax.scatter(
        [base_coords[0, 0]], [base_coords[0, 1]],
        marker="*", s=120, color="red", label="Initialization", zorder=5,
    )

    # Final points
    ax.scatter(
        [base_coords[-1, 0]], [base_coords[-1, 1]],
        marker="s", s=65, color="tab:blue", zorder=6,
    )
    ax.scatter(
        [ipd_coords[-1, 0]], [ipd_coords[-1, 1]],
        marker="s", s=65, color="tab:orange", zorder=6,
    )

    annotate_epochs(ax, base_epochs, base_coords, color="tab:blue")
    annotate_epochs(ax, ipd_epochs, ipd_coords, color="tab:orange")

    ax.set_xlabel("Direction 1")
    ax.set_ylabel("Direction 2")
    ax.set_title("Optimization trajectories on an endpoint-defined 2D loss landscape")
    ax.legend(frameon=False)

    # Optional: keep geometric aspect faithful.
    ax.set_aspect("equal", adjustable="box")

    fig.tight_layout()

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".png"), dpi=CONFIG["dpi"], bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {out_path.with_suffix('.pdf')}")
    print(f"Saved: {out_path.with_suffix('.png')}")


# ============================================================
# Main
# ============================================================
def main():
    baseline_dir = get_run_dir(CONFIG["baseline_run"])
    ipd_dir = get_run_dir(CONFIG["ipd_run"])

    print(f"Baseline checkpoint dir: {baseline_dir}")
    print(f"IPD checkpoint dir:      {ipd_dir}")

    baseline_ckpts = collect_epoch_checkpoints(baseline_dir)
    ipd_ckpts = collect_epoch_checkpoints(ipd_dir)

    model_template = build_model(CONFIG["num_classes"])
    param_names = get_param_names(model_template)

    test_loader, subset_indices = build_test_subset_loader()

    out_dir = CONFIG["out_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "figure5_test_subset_indices.json", "w", encoding="utf-8") as f:
        json.dump(subset_indices, f)

    # Initial state: assume both runs share epoch_000.
    init_state, _ = load_ckpt(baseline_dir / "epoch_000.pt", map_location="cpu")
    theta0 = state_dict_to_param_vector(init_state, param_names)

    # Final states.
    base_final_state, _ = load_ckpt(find_final_checkpoint(baseline_dir), map_location="cpu")
    ipd_final_state, _ = load_ckpt(find_final_checkpoint(ipd_dir), map_location="cpu")

    theta_base_final = state_dict_to_param_vector(base_final_state, param_names)
    theta_ipd_final = state_dict_to_param_vector(ipd_final_state, param_names)

    # Check both initial checkpoints match.
    ipd_init_state, _ = load_ckpt(ipd_dir / "epoch_000.pt", map_location="cpu")
    theta0_ipd = state_dict_to_param_vector(ipd_init_state, param_names)
    init_diff = torch.norm(theta0 - theta0_ipd).item()
    print(f"[Sanity] ||theta0_baseline - theta0_ipd|| = {init_diff:.6e}")

    # Sanity losses using each checkpoint's own full state.
    device = CONFIG["device"]
    print("\nSanity check: direct checkpoint losses")
    eval_full_state_loss(init_state, test_loader, device, "Initialization")
    eval_full_state_loss(base_final_state, test_loader, device, "AdamW final")
    eval_full_state_loss(ipd_final_state, test_loader, device, "AdamW + IPD-Diff final")

    # Build endpoint basis.
    e1, e2 = build_endpoint_basis(theta0, theta_base_final, theta_ipd_final)

    # Load trajectory vectors.
    base_epochs, base_vectors = load_trajectory_vectors(baseline_ckpts, param_names)
    ipd_epochs, ipd_vectors = load_trajectory_vectors(ipd_ckpts, param_names)

    # Project trajectories.
    base_epochs, base_coords = project_trajectory(base_vectors, base_epochs, theta0, e1, e2)
    ipd_epochs, ipd_coords = project_trajectory(ipd_vectors, ipd_epochs, theta0, e1, e2)

    # Print key coordinates.
    print("\nProjected coordinates:")
    print(f"  AdamW final:        {base_coords[-1]}")
    print(f"  AdamW + IPD final:  {ipd_coords[-1]}")

    # Grid range.
    x_min, x_max, y_min, y_max = compute_grid_range(
        base_coords, ipd_coords, CONFIG["margin_ratio"]
    )

    xs_tmp = np.linspace(x_min, x_max, CONFIG["grid_size"])
    ys_tmp = np.linspace(y_min, y_max, CONFIG["grid_size"])

    # Tube mask around trajectories.
    mask = None
    if CONFIG["use_trajectory_tube"]:
        tube_radius = compute_tube_radius(base_coords, ipd_coords)
        mask = build_trajectory_tube_mask(xs_tmp, ys_tmp, base_coords, ipd_coords, tube_radius)
        print(f"\nTrajectory tube radius: {tube_radius:.4f}")
        print(f"Tube grid coverage: {mask.sum()}/{mask.size}")

    # Use baseline final as reference for non-parameter buffers.
    reference_state = strip_module_prefix_if_needed(base_final_state)

    # Optional sanity: vector losses with reference buffers and chosen BN mode.
    print("\nSanity check: vector-loaded losses")
    eval_vector_loss(theta0, reference_state, param_names, test_loader, device, "theta0 with reference buffers")
    eval_vector_loss(theta_base_final, reference_state, param_names, test_loader, device, "AdamW final with reference buffers")
    eval_vector_loss(theta_ipd_final, reference_state, param_names, test_loader, device, "IPD final with reference buffers")

    # Cache paths.
    cache_path = out_dir / f"{CONFIG['out_name']}_data.npz"

    if CONFIG["use_cached_landscape"] and cache_path.exists():
        print(f"\nLoading cached landscape data from {cache_path}")
        data = np.load(cache_path)
        xs = data["xs"]
        ys = data["ys"]
        Z = data["Z"]
        base_epochs = data["base_epochs"]
        base_coords = data["base_coords"]
        ipd_epochs = data["ipd_epochs"]
        ipd_coords = data["ipd_coords"]
    else:
        landscape_model = build_model(CONFIG["num_classes"])
        xs, ys, Z = compute_loss_landscape(
            model=landscape_model,
            param_names=param_names,
            reference_state_dict=reference_state,
            theta0=theta0,
            e1=e1,
            e2=e2,
            test_loader=test_loader,
            x_range=(x_min, x_max),
            y_range=(y_min, y_max),
            mask=mask,
        )

        np.savez(
            cache_path,
            xs=xs,
            ys=ys,
            Z=Z,
            base_epochs=base_epochs,
            base_coords=base_coords,
            ipd_epochs=ipd_epochs,
            ipd_coords=ipd_coords,
        )
        print(f"Saved raw landscape data to: {cache_path}")

    out_path = out_dir / CONFIG["out_name"]
    plot_landscape(
        xs=xs,
        ys=ys,
        Z=Z,
        base_epochs=base_epochs,
        base_coords=base_coords,
        ipd_epochs=ipd_epochs,
        ipd_coords=ipd_coords,
        out_path=out_path,
    )

    print("\nDone.")
    print("Notes:")
    print("  - The 2D plane is defined by endpoint directions.")
    print("  - The displayed landscape is restricted to a tube around the trajectories.")
    print("  - Color values are clipped by percentile for visualization.")


if __name__ == "__main__":
    main()

# plot_figureD1_three_trajectories.py

# import re
# import json
# import copy
# from pathlib import Path
#
# import numpy as np
# import torch
# import torch.nn as nn
# from torch.utils.data import DataLoader, Subset
# from torchvision import datasets, transforms
# from torchvision.models import resnet50
# import matplotlib.pyplot as plt
#
#
# # ============================================================
# # Configuration
# # ============================================================
# CONFIG = {
#     # checkpoint paths
#     "checkpoint_root": Path("./experiment_results/Checkpoints/Resnet"),
#     "baseline_run1": "AdamW_Pure",
#     "baseline_run2": "SGD_Pure",
#     "ipd_run": "AdamW_IPD_diff",
#     "seed": 45,
#
#     # output
#     "out_dir": Path("figures"),
#     "out_name": "figureD1_three_trajectories",
#
#     # model / data
#     "num_classes": 100,
#     "data_root": "./data",
#     "device": "cuda" if torch.cuda.is_available() else "cpu",
#
#     # landscape evaluation
#     "test_subset_size": 2048,
#     "landscape_batch_size": 256,
#
#     # endpoint plane only
#     # Direction 1: shared init -> AdamW final
#     # Direction 2: orthogonalized shared init -> AdamW+IPD-Diff final
#     "basis_method": "endpoint",
#
#     # grid
#     "grid_size": 71,
#     "margin_ratio": 0.06,
#
#     # only show/evaluate loss near trajectories
#     "use_trajectory_tube": True,
#     "tube_radius_ratio": 0.055,
#     "tube_radius_min": 20.0,
#
#     # color clipping for visualization
#     "clip_loss_percentile": 95,
#
#     # BN handling:
#     # "fixed": model.eval(), use BN buffers from reference checkpoint
#     # "batch": model.train(), use batch statistics during loss evaluation
#     "bn_mode": "batch",
#
#     # cache
#     "use_cached_landscape": False,
#
#     # plot
#     "dpi": 300,
# }
#
#
# # ============================================================
# # Model and data
# # ============================================================
# def build_model(num_classes=100):
#     model = resnet50(weights=None, num_classes=num_classes)
#     model.conv1 = nn.Conv2d(
#         3, 64, kernel_size=3, stride=1, padding=1, bias=False
#     )
#     model.maxpool = nn.Identity()
#     return model
#
#
# def build_test_subset_loader():
#     mean = (0.5071, 0.4867, 0.4408)
#     std = (0.2675, 0.2565, 0.2761)
#
#     transform = transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Normalize(mean, std),
#     ])
#
#     test_set = datasets.CIFAR100(
#         root=CONFIG["data_root"],
#         train=False,
#         download=True,
#         transform=transform,
#     )
#
#     rng = np.random.RandomState(CONFIG["seed"])
#     n = min(CONFIG["test_subset_size"], len(test_set))
#     idx = rng.choice(np.arange(len(test_set)), size=n, replace=False)
#     idx = sorted(idx.tolist())
#
#     subset = Subset(test_set, idx)
#     loader = DataLoader(
#         subset,
#         batch_size=CONFIG["landscape_batch_size"],
#         shuffle=False,
#         num_workers=0,
#         pin_memory=True,
#     )
#
#     return loader, idx
#
#
# # ============================================================
# # Checkpoint handling
# # ============================================================
# def get_run_dir(run_name):
#     return CONFIG["checkpoint_root"] / run_name / f"seed_{CONFIG['seed']}"
#
#
# def extract_epoch_from_name(path: Path):
#     if path.stem == "final":
#         return None
#     m = re.search(r"epoch_(\d+)", path.stem)
#     if m is None:
#         return None
#     return int(m.group(1))
#
#
# def collect_epoch_checkpoints(run_dir):
#     files = []
#     for p in sorted(run_dir.glob("epoch_*.pt")):
#         ep = extract_epoch_from_name(p)
#         if ep is not None:
#             files.append((ep, p))
#
#     files = sorted(files, key=lambda x: x[0])
#
#     if len(files) == 0:
#         raise FileNotFoundError(f"No epoch_*.pt checkpoints found in {run_dir}")
#
#     return files
#
#
# def find_final_checkpoint(run_dir):
#     final_path = run_dir / "final.pt"
#     if final_path.exists():
#         return final_path
#     ckpts = collect_epoch_checkpoints(run_dir)
#     return ckpts[-1][1]
#
#
# def load_ckpt(path, map_location="cpu"):
#     ckpt = torch.load(path, map_location=map_location)
#
#     if isinstance(ckpt, dict):
#         if "model_state_dict" in ckpt:
#             return ckpt["model_state_dict"], ckpt.get("epoch", None)
#         if "model_state" in ckpt:
#             return ckpt["model_state"], ckpt.get("epoch", None)
#
#         # raw state_dict
#         if all(torch.is_tensor(v) for v in ckpt.values()):
#             return ckpt, None
#
#     raise ValueError(f"Cannot parse checkpoint: {path}")
#
#
# def strip_module_prefix_if_needed(state_dict):
#     keys = list(state_dict.keys())
#     if len(keys) > 0 and all(k.startswith("module.") for k in keys):
#         return {k.replace("module.", "", 1): v for k, v in state_dict.items()}
#     return state_dict
#
#
# # ============================================================
# # Vectorization
# # ============================================================
# def get_param_names(model):
#     return [name for name, _ in model.named_parameters()]
#
#
# def state_dict_to_param_vector(state_dict, param_names):
#     state_dict = strip_module_prefix_if_needed(state_dict)
#     vecs = []
#     for name in param_names:
#         if name not in state_dict:
#             raise KeyError(f"Parameter {name} not found in checkpoint.")
#         vecs.append(state_dict[name].detach().float().reshape(-1).cpu())
#     return torch.cat(vecs, dim=0)
#
#
# def load_vector_to_model(model, vec, reference_state_dict, param_names):
#     """
#     Load parameter vector into model.
#     Non-parameter buffers such as BN running statistics are taken from reference_state_dict.
#     """
#     state = copy.deepcopy(strip_module_prefix_if_needed(reference_state_dict))
#
#     pointer = 0
#     for name in param_names:
#         shape = state[name].shape
#         numel = state[name].numel()
#         state[name] = vec[pointer:pointer + numel].view(shape).to(state[name].dtype)
#         pointer += numel
#
#     if pointer != vec.numel():
#         raise RuntimeError("Vector length does not match model parameters.")
#
#     model.load_state_dict(state, strict=True)
#
#
# def load_state_to_model(model, state_dict):
#     state_dict = strip_module_prefix_if_needed(state_dict)
#     model.load_state_dict(state_dict, strict=True)
#
#
# # ============================================================
# # Basis construction
# # ============================================================
# def normalize(v, eps=1e-12):
#     n = torch.norm(v)
#     if n < eps:
#         raise ValueError("Cannot normalize a near-zero vector.")
#     return v / n
#
#
# def build_endpoint_basis(theta0, theta_base_final, theta_ipd_final):
#     """
#     Endpoint-defined common 2D plane.
#
#     e1: direction from common initialization to AdamW final.
#     e2: orthogonalized direction from common initialization to AdamW+IPD-Diff final.
#     """
#     v1 = theta_base_final - theta0
#     e1 = normalize(v1)
#
#     v2_raw = theta_ipd_final - theta0
#     v2 = v2_raw - torch.dot(v2_raw, e1) * e1
#     e2 = normalize(v2)
#
#     return e1, e2
#
#
# def project(theta, theta0, e1, e2):
#     d = theta - theta0
#     x = torch.dot(d, e1).item()
#     y = torch.dot(d, e2).item()
#     return x, y
#
#
# def load_trajectory_vectors(ckpts, param_names):
#     epochs = []
#     vectors = []
#     for ep, p in ckpts:
#         state, _ = load_ckpt(p, map_location="cpu")
#         vec = state_dict_to_param_vector(state, param_names)
#         epochs.append(ep)
#         vectors.append(vec)
#     return np.asarray(epochs, dtype=np.int32), vectors
#
#
# def project_trajectory(vectors, epochs, theta0, e1, e2):
#     coords = []
#     for v in vectors:
#         coords.append(project(v, theta0, e1, e2))
#     return epochs, np.asarray(coords, dtype=np.float32)
#
#
# # ============================================================
# # Loss evaluation
# # ============================================================
# @torch.no_grad()
# def evaluate_loss(model, loader, device, bn_mode="batch"):
#     """
#     bn_mode:
#       - fixed: model.eval(), use checkpoint BN running stats.
#       - batch: model.train(), use batch statistics during loss evaluation.
#     """
#     criterion = nn.CrossEntropyLoss(reduction="sum")
#
#     if bn_mode == "fixed":
#         model.eval()
#     elif bn_mode == "batch":
#         model.train()
#     else:
#         raise ValueError("bn_mode must be 'fixed' or 'batch'.")
#
#     total_loss = 0.0
#     total_num = 0
#
#     for x, y in loader:
#         x = x.to(device, non_blocking=True)
#         y = y.to(device, non_blocking=True)
#
#         logits = model(x)
#         loss = criterion(logits, y)
#
#         total_loss += loss.item()
#         total_num += y.size(0)
#
#     return total_loss / total_num
#
#
# def eval_full_state_loss(state_dict, loader, device, label):
#     model = build_model(CONFIG["num_classes"]).to(device)
#     load_state_to_model(model, state_dict)
#     loss_fixed = evaluate_loss(model, loader, device, bn_mode="fixed")
#
#     model = build_model(CONFIG["num_classes"]).to(device)
#     load_state_to_model(model, state_dict)
#     loss_batch = evaluate_loss(model, loader, device, bn_mode="batch")
#
#     print(f"[Sanity] {label}: loss(eval BN)={loss_fixed:.4f}, loss(batch BN)={loss_batch:.4f}")
#
#
# def eval_vector_loss(vec, reference_state, param_names, loader, device, label):
#     model = build_model(CONFIG["num_classes"]).to(device)
#     load_vector_to_model(model, vec, reference_state, param_names)
#     loss = evaluate_loss(model, loader, device, bn_mode=CONFIG["bn_mode"])
#     print(f"[Sanity] {label}: vector loss ({CONFIG['bn_mode']} BN)={loss:.4f}")
#
#
# # ============================================================
# # Grid and trajectory-neighborhood mask
# # ============================================================
# def compute_grid_range_multi(coords_list, margin_ratio):
#     all_xy = np.concatenate(coords_list, axis=0)
#
#     x_min, y_min = all_xy.min(axis=0)
#     x_max, y_max = all_xy.max(axis=0)
#
#     x_margin = max((x_max - x_min) * margin_ratio, 1e-6)
#     y_margin = max((y_max - y_min) * margin_ratio, 1e-6)
#
#     return (
#         x_min - x_margin,
#         x_max + x_margin,
#         y_min - y_margin,
#         y_max + y_margin,
#     )
#
#
# def compute_tube_radius_multi(coords_list):
#     all_xy = np.concatenate(coords_list, axis=0)
#     span = max(
#         float(all_xy[:, 0].max() - all_xy[:, 0].min()),
#         float(all_xy[:, 1].max() - all_xy[:, 1].min()),
#     )
#     return max(CONFIG["tube_radius_min"], CONFIG["tube_radius_ratio"] * span)
#
#
# def min_distance_to_polyline(points, polyline):
#     """
#     points: [N, 2]
#     polyline: [M, 2]
#     Return min distance from each point to any segment in the polyline.
#     """
#     points = np.asarray(points, dtype=np.float64)
#     polyline = np.asarray(polyline, dtype=np.float64)
#
#     if len(polyline) < 2:
#         return np.linalg.norm(points - polyline[0][None, :], axis=1)
#
#     min_dist = np.full(points.shape[0], np.inf, dtype=np.float64)
#
#     for i in range(len(polyline) - 1):
#         a = polyline[i]
#         b = polyline[i + 1]
#         ab = b - a
#         ab2 = np.dot(ab, ab)
#
#         if ab2 < 1e-12:
#             dist = np.linalg.norm(points - a[None, :], axis=1)
#         else:
#             ap = points - a[None, :]
#             t = np.clip((ap @ ab) / ab2, 0.0, 1.0)
#             proj = a[None, :] + t[:, None] * ab[None, :]
#             dist = np.linalg.norm(points - proj, axis=1)
#
#         min_dist = np.minimum(min_dist, dist)
#
#     return min_dist
#
#
# def build_trajectory_tube_mask_multi(xs, ys, coords_list, tube_radius):
#     X, Y = np.meshgrid(xs, ys)
#     pts = np.stack([X.reshape(-1), Y.reshape(-1)], axis=1)
#
#     d_all = []
#     for coords in coords_list:
#         d_all.append(min_distance_to_polyline(pts, coords))
#
#     d = np.min(np.stack(d_all, axis=0), axis=0)
#     mask = (d <= tube_radius).reshape(len(ys), len(xs))
#     return mask
#
#
# # ============================================================
# # Landscape computation
# # ============================================================
# def compute_loss_landscape(
#     model,
#     param_names,
#     reference_state_dict,
#     theta0,
#     e1,
#     e2,
#     test_loader,
#     x_range,
#     y_range,
#     mask=None,
# ):
#     device = CONFIG["device"]
#     grid_size = CONFIG["grid_size"]
#
#     xs = np.linspace(x_range[0], x_range[1], grid_size)
#     ys = np.linspace(y_range[0], y_range[1], grid_size)
#
#     Z = np.full((grid_size, grid_size), np.nan, dtype=np.float32)
#
#     model = model.to(device)
#
#     if mask is None:
#         mask = np.ones((grid_size, grid_size), dtype=bool)
#
#     total = int(mask.sum())
#     count = 0
#
#     print("\nComputing loss landscape on fixed test subset...")
#     print(f"Grid size: {grid_size} x {grid_size}; evaluated points: {total}/{grid_size * grid_size}")
#     print(f"BN mode for landscape: {CONFIG['bn_mode']}")
#
#     for iy, y in enumerate(ys):
#         for ix, x in enumerate(xs):
#             if not mask[iy, ix]:
#                 continue
#
#             theta = theta0 + float(x) * e1 + float(y) * e2
#
#             load_vector_to_model(
#                 model=model,
#                 vec=theta,
#                 reference_state_dict=reference_state_dict,
#                 param_names=param_names,
#             )
#
#             loss = evaluate_loss(model, test_loader, device, bn_mode=CONFIG["bn_mode"])
#             Z[iy, ix] = loss
#
#             count += 1
#             if count % 20 == 0 or count == total:
#                 print(f"  progress: {count}/{total}")
#
#     return xs, ys, Z
#
#
# # ============================================================
# # Plotting
# # ============================================================
# def annotate_epochs(ax, epochs, coords, color, label_set=None):
#     if label_set is None:
#         label_set = {0, 10, 20, 50, 100, 150, 200}
#
#     for ep, (x, y) in zip(epochs, coords):
#         if int(ep) in label_set:
#             ax.text(x, y, str(int(ep)), fontsize=8, color=color)
#
#
# def plot_landscape(xs, ys, Z, trajectories, out_path):
#     """
#     trajectories: list of dicts:
#       {
#         "label": str,
#         "epochs": np.ndarray,
#         "coords": np.ndarray,
#         "color": str,
#         "marker": str,
#         "linewidth": float,
#       }
#     """
#     X, Y = np.meshgrid(xs, ys)
#
#     finite = np.isfinite(Z)
#     if finite.sum() == 0:
#         raise RuntimeError("No finite loss values to plot.")
#
#     z_valid = Z[finite]
#     vmax = np.nanpercentile(z_valid, CONFIG["clip_loss_percentile"])
#     vmin = np.nanpercentile(z_valid, 5)
#
#     if vmax <= vmin:
#         vmax = np.nanmax(z_valid)
#         vmin = np.nanmin(z_valid)
#
#     Z_clip = np.clip(Z, vmin, vmax)
#     Z_masked = np.ma.masked_invalid(Z_clip)
#
#     fig, ax = plt.subplots(figsize=(7.4, 6.2))
#
#     cmap = plt.cm.viridis.copy()
#     cmap.set_bad(color="0.92")
#
#     levels = np.linspace(vmin, vmax, 28)
#     cf = ax.contourf(X, Y, Z_masked, levels=levels, cmap=cmap, extend="max")
#     cbar = fig.colorbar(cf, ax=ax)
#     cbar.set_label("Cross-entropy loss on test subset")
#
#     try:
#         ax.contour(
#             X, Y, Z_masked,
#             levels=np.linspace(vmin, vmax, 12),
#             colors="white",
#             linewidths=0.45,
#             alpha=0.45,
#         )
#     except Exception:
#         pass
#
#     for tr in trajectories:
#         coords = tr["coords"]
#         ax.plot(
#             coords[:, 0],
#             coords[:, 1],
#             marker=tr.get("marker", "o"),
#             markersize=3,
#             linewidth=tr.get("linewidth", 2.0),
#             label=tr["label"],
#             color=tr["color"],
#         )
#
#     # Common initialization from the first trajectory.
#     init_xy = trajectories[0]["coords"][0]
#     ax.scatter(
#         [init_xy[0]],
#         [init_xy[1]],
#         marker="*",
#         s=120,
#         color="red",
#         label="Initialization",
#         zorder=6,
#     )
#
#     # Final points.
#     for tr in trajectories:
#         coords = tr["coords"]
#         ax.scatter(
#             [coords[-1, 0]],
#             [coords[-1, 1]],
#             marker="s",
#             s=65,
#             color=tr["color"],
#             zorder=7,
#         )
#
#     label_set = {0, 10, 20, 50, 100, 150, 200}
#     for tr in trajectories:
#         annotate_epochs(
#             ax,
#             tr["epochs"],
#             tr["coords"],
#             color=tr["color"],
#             label_set=label_set,
#         )
#
#     ax.set_xlabel("Direction 1")
#     ax.set_ylabel("Direction 2")
#     ax.set_title("Projected trajectories on an endpoint-defined 2D loss landscape")
#     ax.legend(frameon=False)
#
#     ax.set_aspect("equal", adjustable="box")
#     fig.tight_layout()
#
#     out_path = Path(out_path)
#     out_path.parent.mkdir(parents=True, exist_ok=True)
#
#     fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
#     fig.savefig(out_path.with_suffix(".png"), dpi=CONFIG["dpi"], bbox_inches="tight")
#     plt.close(fig)
#
#     print(f"Saved: {out_path.with_suffix('.pdf')}")
#     print(f"Saved: {out_path.with_suffix('.png')}")
#
#
# # ============================================================
# # Main
# # ============================================================
# def main():
#     baseline1_dir = get_run_dir(CONFIG["baseline_run1"])  # AdamW_Pure
#     baseline2_dir = get_run_dir(CONFIG["baseline_run2"])  # SGD_Pure
#     ipd_dir = get_run_dir(CONFIG["ipd_run"])              # AdamW_IPD_diff
#
#     print(f"Baseline-1 checkpoint dir: {baseline1_dir}")
#     print(f"Baseline-2 checkpoint dir: {baseline2_dir}")
#     print(f"IPD checkpoint dir:        {ipd_dir}")
#
#     baseline1_ckpts = collect_epoch_checkpoints(baseline1_dir)
#     baseline2_ckpts = collect_epoch_checkpoints(baseline2_dir)
#     ipd_ckpts = collect_epoch_checkpoints(ipd_dir)
#
#     model_template = build_model(CONFIG["num_classes"])
#     param_names = get_param_names(model_template)
#
#     test_loader, subset_indices = build_test_subset_loader()
#
#     out_dir = CONFIG["out_dir"]
#     out_dir.mkdir(parents=True, exist_ok=True)
#
#     with open(out_dir / "figureD1_test_subset_indices.json", "w", encoding="utf-8") as f:
#         json.dump(subset_indices, f)
#
#     # AdamW epoch_000 is used as the reference initialization.
#     init_state, _ = load_ckpt(baseline1_dir / "epoch_000.pt", map_location="cpu")
#     theta0 = state_dict_to_param_vector(init_state, param_names)
#
#     # Final states.
#     base1_final_state, _ = load_ckpt(find_final_checkpoint(baseline1_dir), map_location="cpu")
#     ipd_final_state, _ = load_ckpt(find_final_checkpoint(ipd_dir), map_location="cpu")
#     base2_final_state, _ = load_ckpt(find_final_checkpoint(baseline2_dir), map_location="cpu")
#
#     theta_base1_final = state_dict_to_param_vector(base1_final_state, param_names)
#     theta_ipd_final = state_dict_to_param_vector(ipd_final_state, param_names)
#     theta_base2_final = state_dict_to_param_vector(base2_final_state, param_names)
#
#     # Check initial checkpoints.
#     ipd_init_state, _ = load_ckpt(ipd_dir / "epoch_000.pt", map_location="cpu")
#     base2_init_state, _ = load_ckpt(baseline2_dir / "epoch_000.pt", map_location="cpu")
#
#     theta0_ipd = state_dict_to_param_vector(ipd_init_state, param_names)
#     theta0_base2 = state_dict_to_param_vector(base2_init_state, param_names)
#
#     init_diff_ipd = torch.norm(theta0 - theta0_ipd).item()
#     init_diff_base2 = torch.norm(theta0 - theta0_base2).item()
#
#     print(f"[Sanity] ||theta0_AdamW - theta0_IPD|| = {init_diff_ipd:.6e}")
#     print(f"[Sanity] ||theta0_AdamW - theta0_SGD|| = {init_diff_base2:.6e}")
#
#     if init_diff_ipd > 1e-6:
#         print("[Warning] AdamW and IPD do not share exactly the same initialization.")
#     if init_diff_base2 > 1e-6:
#         print("[Warning] SGD does not share exactly the same initialization as AdamW. "
#               "It will still be projected into the AdamW/IPD endpoint plane, but interpretation should be cautious.")
#
#     # Sanity losses using each checkpoint's own full state.
#     device = CONFIG["device"]
#     print("\nSanity check: direct checkpoint losses")
#     eval_full_state_loss(init_state, test_loader, device, "Initialization")
#     eval_full_state_loss(base1_final_state, test_loader, device, "AdamW final")
#     eval_full_state_loss(ipd_final_state, test_loader, device, "AdamW + IPD-Diff final")
#     eval_full_state_loss(base2_final_state, test_loader, device, "SGD final")
#
#     # Build endpoint basis from AdamW and AdamW+IPD-Diff.
#     e1, e2 = build_endpoint_basis(theta0, theta_base1_final, theta_ipd_final)
#
#     # Load trajectory vectors.
#     base1_epochs, base1_vectors = load_trajectory_vectors(baseline1_ckpts, param_names)
#     base2_epochs, base2_vectors = load_trajectory_vectors(baseline2_ckpts, param_names)
#     ipd_epochs, ipd_vectors = load_trajectory_vectors(ipd_ckpts, param_names)
#
#     # Project all trajectories to the same endpoint-defined plane.
#     base1_epochs, base1_coords = project_trajectory(base1_vectors, base1_epochs, theta0, e1, e2)
#     base2_epochs, base2_coords = project_trajectory(base2_vectors, base2_epochs, theta0, e1, e2)
#     ipd_epochs, ipd_coords = project_trajectory(ipd_vectors, ipd_epochs, theta0, e1, e2)
#
#     print("\nProjected coordinates:")
#     print(f"  AdamW final:        {base1_coords[-1]}")
#     print(f"  AdamW + IPD final:  {ipd_coords[-1]}")
#     print(f"  SGD final:          {base2_coords[-1]}")
#
#     coords_list = [base1_coords, ipd_coords, base2_coords]
#
#     # Grid range over all three projected trajectories.
#     x_min, x_max, y_min, y_max = compute_grid_range_multi(
#         coords_list,
#         CONFIG["margin_ratio"],
#     )
#
#     xs_tmp = np.linspace(x_min, x_max, CONFIG["grid_size"])
#     ys_tmp = np.linspace(y_min, y_max, CONFIG["grid_size"])
#
#     # Tube mask around all three trajectories.
#     mask = None
#     if CONFIG["use_trajectory_tube"]:
#         tube_radius = compute_tube_radius_multi(coords_list)
#         mask = build_trajectory_tube_mask_multi(
#             xs_tmp,
#             ys_tmp,
#             coords_list,
#             tube_radius,
#         )
#         print(f"\nTrajectory tube radius: {tube_radius:.4f}")
#         print(f"Tube grid coverage: {mask.sum()}/{mask.size}")
#
#     # Use AdamW final as reference for non-parameter buffers.
#     reference_state = strip_module_prefix_if_needed(base1_final_state)
#
#     print("\nSanity check: vector-loaded losses")
#     eval_vector_loss(theta0, reference_state, param_names, test_loader, device, "theta0 with reference buffers")
#     eval_vector_loss(theta_base1_final, reference_state, param_names, test_loader, device, "AdamW final with reference buffers")
#     eval_vector_loss(theta_ipd_final, reference_state, param_names, test_loader, device, "IPD final with reference buffers")
#     eval_vector_loss(theta_base2_final, reference_state, param_names, test_loader, device, "SGD final with reference buffers")
#
#     # Cache path.
#     cache_path = out_dir / f"{CONFIG['out_name']}_data.npz"
#
#     if CONFIG["use_cached_landscape"] and cache_path.exists():
#         print(f"\nLoading cached landscape data from {cache_path}")
#         data = np.load(cache_path)
#         xs = data["xs"]
#         ys = data["ys"]
#         Z = data["Z"]
#         base1_epochs = data["base1_epochs"]
#         base1_coords = data["base1_coords"]
#         base2_epochs = data["base2_epochs"]
#         base2_coords = data["base2_coords"]
#         ipd_epochs = data["ipd_epochs"]
#         ipd_coords = data["ipd_coords"]
#     else:
#         landscape_model = build_model(CONFIG["num_classes"])
#         xs, ys, Z = compute_loss_landscape(
#             model=landscape_model,
#             param_names=param_names,
#             reference_state_dict=reference_state,
#             theta0=theta0,
#             e1=e1,
#             e2=e2,
#             test_loader=test_loader,
#             x_range=(x_min, x_max),
#             y_range=(y_min, y_max),
#             mask=mask,
#         )
#
#         np.savez(
#             cache_path,
#             xs=xs,
#             ys=ys,
#             Z=Z,
#             base1_epochs=base1_epochs,
#             base1_coords=base1_coords,
#             base2_epochs=base2_epochs,
#             base2_coords=base2_coords,
#             ipd_epochs=ipd_epochs,
#             ipd_coords=ipd_coords,
#         )
#         print(f"Saved raw landscape data to: {cache_path}")
#
#     trajectories = [
#         {
#             "label": "AdamW",
#             "epochs": base1_epochs,
#             "coords": base1_coords,
#             "color": "tab:blue",
#             "marker": "o",
#             "linewidth": 2.0,
#         },
#         {
#             "label": "AdamW + IPD-Diff",
#             "epochs": ipd_epochs,
#             "coords": ipd_coords,
#             "color": "tab:orange",
#             "marker": "o",
#             "linewidth": 2.0,
#         },
#         {
#             "label": "SGD",
#             "epochs": base2_epochs,
#             "coords": base2_coords,
#             "color": "tab:green",
#             "marker": "o",
#             "linewidth": 2.0,
#         },
#     ]
#
#     out_path = out_dir / CONFIG["out_name"]
#     plot_landscape(
#         xs=xs,
#         ys=ys,
#         Z=Z,
#         trajectories=trajectories,
#         out_path=out_path,
#     )
#
#     print("\nDone.")
#     print("Notes:")
#     print("  - The 2D plane is defined by AdamW and AdamW+IPD-Diff endpoint directions.")
#     print("  - SGD is projected into this same plane as an additional reference trajectory.")
#     print("  - The displayed landscape is restricted to a tube around all shown trajectories.")
#     print("  - Color values are clipped by percentile for visualization.")
#
#
# if __name__ == "__main__":
#     main()