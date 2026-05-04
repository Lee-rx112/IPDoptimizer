import torch.nn as nn
from typing import Dict, Any
from collections import Counter


def build_neighborhood_map(model: nn.Module, verbose: bool = True) -> Dict[int, Dict[str, Any]]:
    """
    通用型邻域与角色构建函数。

    当前策略（与原实现保持一致）：
    1. 所有 requires_grad 且 dim > 1 的参数视为“计算型权重”。
    2. dim == 4 -> intra_kernel_conv
       dim == 2 -> intra_vector_fc
       其他参数（bias, BN 等） -> none
    3. role 策略：
       - 前 25% 的计算层，或前 5 个计算层：active
       - 其余：passive

    返回:
        neighborhood_map[id(param)] = {
            "type": "...",
            "role": "...",
            "name": "...",
            "layer_idx": ...
        }
    """
    neighborhood_map: Dict[int, Dict[str, Any]] = {}

    # 先收集所有“计算型权重”
    weight_entries = []
    for name, p in model.named_parameters():
        if p.requires_grad and p.dim() > 1:
            weight_entries.append((name, p))

    total_weights = len(weight_entries)
    if total_weights == 0:
        if verbose:
            print("Warning: No weight parameters found!")
        return {}

    weight_id_to_idx = {id(p): i for i, (_, p) in enumerate(weight_entries)}

    # 先为所有参数填默认信息
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        neighborhood_map[id(p)] = {
            "type": "none",
            "role": "passive",
            "name": name,
            "layer_idx": -1,
        }

    # 再给计算型权重打标
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue

        pid = id(p)
        info = neighborhood_map[pid]

        if p.dim() == 4:
            info["type"] = "intra_kernel_conv"
        elif p.dim() == 2:
            info["type"] = "intra_vector_fc"
        else:
            info["type"] = "none"
            continue

        layer_idx = weight_id_to_idx[pid]
        info["layer_idx"] = layer_idx
        ratio = layer_idx / max(total_weights, 1)

        # 与你原逻辑保持一致
        if ratio < 0.25 or layer_idx < 5:
            info["role"] = "active"
        else:
            info["role"] = "passive"

    if verbose:
        type_counter = Counter(v["type"] for v in neighborhood_map.values())
        role_counter = Counter(v["role"] for v in neighborhood_map.values())
        active_named = [
            (v["layer_idx"], v["name"], v["type"])
            for v in neighborhood_map.values()
            if v["role"] == "active" and v["layer_idx"] >= 0
        ]
        active_named = sorted(active_named, key=lambda x: x[0])

        print("Neighborhood map built successfully")
        print(f"  Total params in map      : {len(neighborhood_map)}")
        print(f"  Total weight tensors     : {total_weights}")
        print(f"  Type counter             : {dict(type_counter)}")
        print(f"  Role counter             : {dict(role_counter)}")
        print("  First active layers      :")
        for item in active_named[:10]:
            print(f"    idx={item[0]:>3d} | {item[1]} | {item[2]}")

    return neighborhood_map