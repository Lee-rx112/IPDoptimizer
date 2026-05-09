import torch
import torch.optim as optim
import numpy as np
import os

from utils import  get_model, set_seed, get_official_loaders, train_one_run
from map_construction import build_neighborhood_map
from IPD_opt import IPD

def _to_serializable(obj):
    if isinstance(obj, dict):
        return {k: _to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_to_serializable(v) for v in obj]
    elif isinstance(obj, tuple):
        return [_to_serializable(v) for v in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.float16, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.int8, np.int16, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, torch.Tensor):
        if obj.numel() == 1:
            return obj.item()
        return obj.detach().cpu().tolist()
    else:
        return obj

# random seeds
SEEDS = [46]
EPOCHS = 100
BATCH_SIZE = 128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

RESULTS_DIR = "./experiment_results/Checkpoints/Densenet"
# for CIFAR100 + ResNet50
EXPERIMENTS = [
    # Group A: Baseline IPD_SGD_pure
    {
        'id': 'SGD_Pure',
        'base_opt': 'SGD',
        'base_params': {'lr': 0.1, 'momentum': 0.9, 'weight_decay': 5e-4},
        'ipd_params': {'enable_diff': False, 'enable_coul': False}
    },
    # # Group A: IPD Iso + Diff for Resnet
    # {
    #     'id': 'SGD_IPD_diff',
    #     'base_opt': 'SGD',
    #     'base_params': {'lr': 0.1, 'momentum': 0.9, 'weight_decay': 5e-4},
    #     'ipd_params': {
    #         'enable_diff': True,
    #         'r_diff': 1e-3,            # post-update     3e-4 for pre-update
    #         'diff_decay_ratio': 0.4    # post-update     0.2 for pre-update
    #     }
    # },
    # Group A: IPD Iso + coul  for Densenet
    {
        'id': 'SGD_IPD_coul',
        'base_opt': 'SGD',
        'base_params': {'lr': 0.1, 'momentum': 0.9, 'weight_decay': 5e-4},
        'ipd_params': {
            'enable_coul': True,
            'r_coul': 1e-2,
            'coul_decay_ratio': 0.2
        }
    },
    # Group B: IPD_AdamW_pure
    # {
    #     'id': 'AdamW_Pure',
    #     'base_opt': 'AdamW',
    #     'base_params': {'lr': 1e-3, 'weight_decay': 1e-2},
    #     'ipd_params': {'enable_diff': False, 'enable_coul': False}
    # },
    # Group B: IPD_AdamW_diff
    # {
    #     'id': 'AdamW_IPD_diff',
    #     'base_opt': 'AdamW',
    #     'base_params': {'lr': 1e-3, 'weight_decay': 1e-2},
    #     'ipd_params': {
    #         'enable_diff': True,
    #         'r_diff': 2e-2,
    #         'diff_decay_ratio': 0.2
    #     }
    # },
    # Group B: IPD_AdamW_coul
    # {
    #     'id': 'AdamW_IPD_coul',
    #     'base_opt': 'AdamW',
    #     'base_params': {'lr': 1e-3, 'weight_decay': 1e-2},
    #     'ipd_params': {
    #         'enable_coul': True,
    #         'r_coul': 2e-2,
    #         'coul_decay_ratio': 0.1
    #     }
    # }
]


if __name__ == '__main__':
    # 确保保存目录存在
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)
    print(f"Using Device: {DEVICE}")

    dataset_name = 'CIFAR100'
    # 1. 获取数据
    print("Loading Official CIFAR-100 Data (50k Train / 10k Test)...")
    train_loader, test_loader = get_official_loaders(BATCH_SIZE)
    final_benchmark_report = {}
    num_classes = 100

    total_steps = len(train_loader) * EPOCHS

    for config in EXPERIMENTS:
        exp_id = config["id"]
        base_name = config["base_opt"]
        print(f"\n{'=' * 50}\nRunning Benchmark: {exp_id}\n{'=' * 50}")


        for seed in SEEDS:
            print(f"\n>>> Seed {seed}...")
            set_seed(seed)

            # 2. 初始化模型
            model = get_model('densenet121', num_classes).to(DEVICE)
            # model = get_model('resnet50', num_classes).to(DEVICE)

            # 3. 初始化优化器
            base_opt = None
            if base_name == 'SGD':
                base_opt = optim.SGD(
                    model.parameters(),
                    **config['base_params']
                )
            elif base_name == 'AdamW':
                base_opt = optim.AdamW(
                    model.parameters(),
                    **config['base_params']
                )

            ipd_cfg = config['ipd_params']
            neighbor_map = None
            if ipd_cfg.get('enable_diff', False) or ipd_cfg.get('enable_coul', False):
                print("  Building Neighborhood Map for IPD forces...")
                neighbor_map = build_neighborhood_map(model)

            optimizer = IPD(
                base_optimizer=base_opt,
                map=neighbor_map,
                total_steps=total_steps,
                **ipd_cfg
            )


            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

            ckpt_dir = train_one_run(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                optimizer_name=base_name,
                train_loader=train_loader,
                epochs=EPOCHS,
                checkpoint_root=RESULTS_DIR,
                run_name=exp_id,
                seed=seed,
            )

    print("\nAll experiments finished. Results saved.")

