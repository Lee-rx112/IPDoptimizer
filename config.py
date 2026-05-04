# =======实验配置=========
import os
CONFIG= {
    'dataset': 'CIFAR100',    # 可选择CIFAR100等其他数据集
    'model': 'resnet50',     # 可选择LeNet5等其他模型
    'val_split': 0.2,
    'tuning_epochs': 20,
    'final_epochs': 200,
    'batch_size': 128,
    'lr_scheduler': 'CosineAnnealingLR', #在最终训练中接入学习率调节器sin函数
    'num_workers': 0 if os.name != 'nt' else 0,
    'profile_model': True # 添加一个开关来控制是否进行模型画像
}

HYPERPARAM_SPACE = {
    'SGD': {
        'lr': [1e-2, 1e-3, 5e-4, 1e-4],
        'momentum': [0.9],
        'weight_decay': [5e-4]
    },
    'Adam': {
        'lr': [1e-2, 1e-3, 5e-4, 1e-4],
        'betas': [(0.9, 0.999)]
    },
    'AdamW': {
        'lr': [1e-2,1e-3, 5e-4, 1e-4],
        'betas': [(0.9, 0.999)],
        'weight_decay': [0.01]
    },
    'SAM': {
        'lr': [1e-2, 1e-3, 5e-4, 1e-4],
        'momentum': [0.9],
        'rho': [0.05]
    },
    'HHDyn': {
        'lr': [1e-2, 5e-3, 1e-3, 5e-4, 1e-4]
    }
}