import torch
import torch.optim as optim
import numpy as np
import os
import json

from utils import  get_model, set_seed, get_official_loaders, final_train,get_tiny_imagenet_official_loaders
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
SEEDS = [42]
EPOCHS = 100
BATCH_SIZE = 128
# VAL_SPLIT = 0.1  # 划出 10% 训练数据作为验证集用于选模型
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

RESULTS_DIR = "./experiment_results/Densenet"
# for CIFAR100 + ResNet50
EXPERIMENTS = [
    # Group A: Baseline IPD_SGD_pure
    #     {
    #         'id': 'SGD_Pure',
    #         'base_opt': 'SGD',
    #         'base_params': {'lr': 0.1, 'momentum': 0.9, 'weight_decay': 5e-4},
    #         'ipd_params': {'enable_diff': False, 'enable_coul': False}
#     },
    # Group A: IPD Iso + Diff for Resnet
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
            'r_coul': 1e-3,
            'coul_decay_ratio': 0.4
        }
    },
    # Group B: IPD_AdamW_pure
    # {
    #     'id': 'AdamW_Pure',
    #     'base_opt': 'AdamW',
    #     'base_params': {'lr': 1e-3, 'weight_decay': 1e-2},
    #     'ipd_params': {'enable_diff': False, 'enable_coul': False}
    # }
    # # Group B: IPD_AdamW_diff
    # {
    #     'id': 'AdamW_IPD_diff',
    #     'base_opt': 'AdamW',
    #     'base_params': {'lr': 1e-3, 'weight_decay': 1e-2},
    #     'ipd_params': {
    #         'enable_diff': True,
    #         'r_diff': 1e-3,
    #         'diff_decay_ratio': 0.4
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
    #         'coul_decay_ratio': 0.15
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
    # train_loader, val_loader, test_loader, _, num_classes = get_dataloaders(
    #     dataset_name, BATCH_SIZE, VAL_SPLIT, seed=2025, augment=True
    # )

    print("Loading Official CIFAR-100 Data (50k Train / 10k Test)...")
    train_loader, test_loader = get_official_loaders(BATCH_SIZE)
    final_benchmark_report = {}
    num_classes = 100

    # print("Loading Official TinyImageNet-200 Data (50k Train / 10k Test)...")
    # train_loader, test_loader = get_tiny_imagenet_official_loaders(
    #     data_root='./data/tiny-imagenet-200',
    #     batch_size=BATCH_SIZE,
    #     num_workers=4
    # )
    # final_benchmark_report = {}
    # num_classes = 200

    total_steps = len(train_loader) * EPOCHS

    for config in EXPERIMENTS:
        exp_id = config["id"]
        base_name = config["base_opt"]
        print(f"\n{'=' * 50}\nRunning Benchmark: {exp_id}\n{'=' * 50}")

        seed_results = {
            'best_accs':[],
            'last_accs':[]
        }

        for seed in SEEDS:
            print(f"\n>>> Seed {seed}...")
            set_seed(seed)

            # 2. 初始化模型
            model = get_model('densenet121', num_classes).to(DEVICE)
            # model = get_model('resnet50', num_classes).to(DEVICE)

            # --- 计算参数量和 FLOPs ---
            # num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            # # ResNet50 on CIFAR100 (32x32) 估算 FLOPs
            # forward_flops = 1.3e9
            # print(f"Model Params: {num_params / 1e6:.2f}M | Est. FLOPs: {forward_flops / 1e9:.2f}G")

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
            save_dir = os.path.join(RESULTS_DIR, exp_id, f"seed_{seed}")
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            model_save_path = os.path.join(save_dir, "best_model1.pth")


            history = final_train(
                model, optimizer, scheduler, exp_id,
                train_loader, test_loader,
                EPOCHS, model_save_path
            )

            # 7. 保存该 Seed 的详细日志
            os.makedirs(save_dir, exist_ok=True)
            with open(os.path.join(save_dir, "metrics1.json"), "w", encoding="utf-8") as f:
                json.dump(_to_serializable(history), f, indent=4, ensure_ascii=False)

            seed_results['best_accs'].append(float(history.get('best_test_acc', 0.0)))
            seed_results['last_accs'].append(float(history.get('last_test_acc', 0.0)))

            # 释放显存
            del model, optimizer, scheduler
            torch.cuda.empty_cache()

        mean_best = float(np.mean(seed_results['best_accs'])) if len(seed_results['best_accs']) > 0 else 0.0
        std_best = float(np.std(seed_results['best_accs'])) if len(seed_results['best_accs']) > 0 else 0.0
        mean_last = float(np.mean(seed_results['last_accs'])) if len(seed_results['last_accs']) > 0 else 0.0
        std_last = float(np.std(seed_results['last_accs'])) if len(seed_results['last_accs']) > 0 else 0.0

        print(f"\n[SUMMARY] {exp_id} Finished.")
        print(f"Best Accs: {seed_results['best_accs']} -> Mean: {mean_best:.2f}% ± {std_best:.2f}")
        print(f"Last Accs: {seed_results['last_accs']} -> Mean: {mean_last:.2f}% ± {std_last:.2f}")

        final_benchmark_report[exp_id] = {
            'mean_best': mean_best,
            'std_best': std_best,
            'mean_last': mean_last,
            'std_last': std_last,
            'raw_data': _to_serializable(seed_results)
        }

        # 9. 保存最终对比报告
        exp_dir = os.path.join(RESULTS_DIR, exp_id)
        os.makedirs(exp_dir, exist_ok=True)
        with open(os.path.join(exp_dir, "final_benchmark_report1.json"), "w", encoding="utf-8") as f:
            json.dump(_to_serializable(final_benchmark_report), f, indent=4, ensure_ascii=False)

    print("\nAll experiments finished. Results saved.")

