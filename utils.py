import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.models import resnet18, resnet50, densenet121
from torch.utils.data import Dataset,DataLoader, Subset
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import os
from PIL import Image
import time
from config import CONFIG
import math, random, sys, copy






# from myoptimizer import HHDyn
from map_construction import *
from samoptimizer import SAM

# ===========1. 实验配置与准备 ===============
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===========2. dataset loading ================
def get_official_loaders(batch_size=128, num_workers=2):
    """
    加载 CIFAR-100 官方划分数据 (无验证集模式)
    Train: 50,000 images (全部用于训练)
    Test:  10,000 images (用于测试)
    """

    # 1. 定义标准的数据增强和归一化
    stats = ((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(*stats),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(*stats),
    ])

    # 2. 加载训练集
    trainset = datasets.CIFAR100(
        root='./data',
        train=True,
        download=True,
        transform=transform_train
    )

    trainloader = DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    # 3. 加载测试集
    testset = datasets.CIFAR100(
        root='./data',
        train=False,
        download=True,
        transform=transform_test
    )

    testloader = DataLoader(
        testset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return trainloader, testloader
# def get_dataloaders(dataset_name, batch_size, val_split, seed, augment = True):
#     '''
#     加载并划分指定的数据集
#     :param dataset_name: CiFAR10 或者 CIFAR100
#     :param batch_size:
#     :param val_split: 从训练集中划分出的验证集比例
#     :param augment: 是否对训练集进行数据增强
#     :return:
#     Tuple of DataLoaders and int: (train_loader, val_loader, test_loader, full_train_loader, num_classes)
#     '''
#     # 为CIFAR10和CIFAR100定义均值和标准差
#     cifar10_mean = (0.4914, 0.4822, 0.4465)
#     cifar10_std = (0.2023, 0.1994, 0.2010)
#     cifar100_mean = (0.5071, 0.4867, 0.4408)
#     cifar100_std = (0.2675, 0.2565, 0.2761)
#
#     # 定义基础变换（不含增强）
#     transform_base = transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Normalize(cifar10_mean, cifar10_std) if dataset_name == 'CIFAR10' else transforms.Normalize(
#             cifar100_mean, cifar100_std)
#     ])
#     if augment:
#         transform_train = transforms.Compose([
#             transforms.RandomCrop(32, padding=4),
#             transforms.RandomHorizontalFlip(),
#             transform_base
#         ])
#     else:
#         transform_train= transform_base
#
#     DatasetClass = datasets.CIFAR10 if dataset_name == 'CIFAR10' else datasets.CIFAR100
#     num_classes = 10 if dataset_name == 'CIFAR10' else 100
#
#     full_train_dataset = DatasetClass(root='./data', train=True, download=True, transform=transform_train)
#     full_val_dataset = DatasetClass(root='./data', train=True, download=True, transform=transform_base)
#     test_dataset = DatasetClass(root='./data', train=False, download=True, transform=transform_base)
#
#
#     num_train = len(full_train_dataset)
#     indices = list(range(num_train))
#     split = int(np.floor(val_split * num_train))
#
#     # np.random.seed(seed)
#     # np.random.shuffle(indices)
#     rs = np.random.RandomState(seed)
#     rs.shuffle(indices)
#
#
#     train_idx, valid_idx = indices[split:], indices[:split]
#
#     #训练集使用有增强的函数，验证集不需要
#     train_dataset = Subset(full_train_dataset, train_idx)
#     val_dataset = Subset(full_val_dataset, valid_idx)
#
#     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=False, num_workers=4)
#     val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=False, num_workers=4)
#     test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=False, num_workers=4)
#     full_train_loader = DataLoader(full_train_dataset, batch_size=batch_size, shuffle=True, pin_memory=False,
#                                    num_workers=4)
#
#     return train_loader, val_loader, test_loader,full_train_loader, num_classes

class TinyImageNetValDataset(Dataset):
    """
    读取 Tiny-ImageNet 官方 val 划分：
    val/
      images/
      val_annotations.txt
    """
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform

        val_dir = os.path.join(root, 'val')
        img_dir = os.path.join(val_dir, 'images')
        anno_path = os.path.join(val_dir, 'val_annotations.txt')
        wnids_path = os.path.join(root, 'wnids.txt')

        # 读取类别顺序，建立 class_to_idx
        with open(wnids_path, 'r') as f:
            classes = [line.strip() for line in f if line.strip()]
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}

        self.samples = []
        with open(anno_path, 'r') as f:
            for line in f:
                # 格式: img_name \t class \t x0 \t y0 \t x1 \t y1
                parts = line.strip().split('\t')
                img_name, cls_name = parts[0], parts[1]
                img_path = os.path.join(img_dir, img_name)
                target = self.class_to_idx[cls_name]
                self.samples.append((img_path, target))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, target = self.samples[idx]
        img = Image.open(img_path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, target


def get_tiny_imagenet_official_loaders(
    data_root='./data/tiny-imagenet-200',
    batch_size=128,
    num_workers=4
):
    """
    Tiny-ImageNet 官方划分
    Train: train/
    Eval : val/   （官方验证集，平时本地实验一般把它当测试集）
    """

    # 你可以先用 ImageNet 的均值方差，也可以后面自己重新统计 Tiny-ImageNet 的
    stats = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

    transform_train = transforms.Compose([
        transforms.RandomCrop(64, padding=8),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(*stats),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(*stats),
    ])

    # 1. 训练集：train/ 下已经按类别分文件夹
    trainset = datasets.ImageFolder(
        root=os.path.join(data_root, 'train'),
        transform=transform_train
    )

    trainloader = DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    # 2. 验证集：官方 val/，读取 val_annotations.txt
    testset = TinyImageNetValDataset(
        root=data_root,
        transform=transform_test
    )

    testloader = DataLoader(
        testset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return trainloader, testloader
def set_seed(seed):
    """
    set random seed for all function
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # 这两项设置可以确保CUDA操作的确定性，但可能会降低一点性能
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print(f"\n[Seed Set to {seed}]")

# ==========3. 模型与优化器加载模块 ==============
def get_model(model_name, num_classes):
    # -----------------------------------------------------------
    # DenseNet-121 适配 CIFAR-100
    # -----------------------------------------------------------
    if model_name == 'densenet121':
        # 1. 加载模型结构
        model = densenet121(weights=None, num_classes=num_classes)

        # 2. 修改第一层卷积 (注意：DenseNet 把层放在 .features 里，且命名为 conv0)
        # 将 7x7 stride=2 改为 3x3 stride=1
        model.features.conv0 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)

        # 3. 移除第一层最大池化
        # (注意：DenseNet 命名为 pool0)
        model.features.pool0 = nn.Identity()

    # -----------------------------------------------------------
    # ResNet 适配 CIFAR-100 (保持你原有的逻辑)
    # -----------------------------------------------------------
    elif model_name == 'resnet18':
        model = resnet18(weights=None, num_classes=num_classes)
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()

    elif model_name == 'resnet50':
        # for TinyImage
        model = resnet50(weights=None, num_classes=num_classes)
        # for CIFAR
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()

    else:
        raise ValueError(f'Model {model_name} not supported')
    return model.to(device)

# def get_optimizer(model, optimizer_name, hparams):
#     # hparams:  a dict including all hyper-parameters
#     if optimizer_name == 'SGD':
#         return optim.SGD(model.parameters(), **hparams)
#     elif optimizer_name == 'Adam':
#         return optim.Adam(model.parameters(), **hparams)
#     elif optimizer_name == 'AdamW':
#         return optim.AdamW(model.parameters(), **hparams)
#     elif optimizer_name == 'SAM':
#         base_optimizer = optim.SGD
#         # 从hparams中弹出SAM专属的rho参数，剩下的自动成为base_optimizer的参数
#         sam_specific_hparams = {'rho': hparams.pop('rho', 0.05)}
#         return SAM(model.parameters(), base_optimizer, **sam_specific_hparams, **hparams)
#     elif optimizer_name == 'HHDyn':
#         return HHDyn(model.parameters(), **hparams)
#     else:
#         raise ValueError(f'Optimizer {optimizer_name} not supported')


# ==========4. 核心训练与评估函数 ==========
def evaluate(model, data_loader, criterion):
    model.eval()
    total_loss = 0.0
    correct, total = 0, 0
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            total_loss += loss.item() * data.size(0)
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    if total ==0:
        return 0.0, 0.0
    avg_loss = total_loss / total
    accuracy = 100 * correct / total
    return  avg_loss, accuracy


def diagnostic_run(model, optimizer, scheduler, train_loader, val_loader, epochs):
    criterion = nn.CrossEntropyLoss()
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'avg_spike_rate': [],
        'cos_itot': [],   # 每个 epoch 的平均余弦 (Itot, -g),放电前“合力方向”与最速下降方向的一致性（邻域力/非线性是否把方向带偏）
        'cos_dW': [],   # 每个 epoch 的平均余弦 (ΔW, -g),放电后“真实稀疏更新”与最速下降方向的一致性（门控是否选对了位置）
        'energy_ratio': [] # 每个 epoch 的平均能量比例 ≈ (cos_dW)^2, 相对密集 SGD，本步保留的梯度 L2 能量占比（ΔW 对 ||g||^2 的保留率的一阶近似）
        }
    print("\n--- Starting Diagnostic Run for EventGrad ---")
    global_step = 0

    for epoch in range(epochs):
        model.train()
        epoch_spikes_temp = []
        epoch_cos_itot = []
        epoch_cos_dW = []

        total_loss = 0.0
        total_data, correct = 0, 0

        for data, target in train_loader:
            # print('I am Here!')
            data, target = data.to(device), target.to(device)

            output = model(data)
            loss = criterion(output, target)

            total_loss += loss.item() * data.size(0)
            total_data += data.size(0)
            _, predicted = torch.max(output.data, 1)
            correct += (predicted == target).sum().item()

            loss.backward()
            optimizer.step()

            try:
                cos_itot = optimizer.cosine_itot_vs_neggrad()
            except Exception:
                cos_itot = float('nan')

            try:
                cos_dW = optimizer.cosine_update_vs_neggrad()
            except Exception:
                cos_dW = float('nan')

            try:
                global_spiking_rate, _, _, _ = optimizer.get_spiking_rate()
            except Exception:
                global_spiking_rate = float('nan')

            if not math.isnan(cos_itot):
                epoch_cos_itot.append(float(cos_itot))
            if not math.isnan(cos_dW):
                epoch_cos_dW.append(float(cos_dW))
            if not math.isnan(global_spiking_rate):
                epoch_spikes_temp.append(float(global_spiking_rate))

            global_step += 1
            optimizer.zero_grad()

        train_loss = total_loss / max(total_data,1)
        train_acc = 100 * correct / max(total_data,1)
        def _nanmean(xs):
            arr = np.array(xs, dtype=float)
            return float(np.nanmean(arr)) if arr.size > 0 else float('nan')

        val_acc, val_loss = evaluate(model, val_loader, criterion)

        avg_epoch_rate = _nanmean(epoch_spikes_temp)
        mean_cos_itot = _nanmean(epoch_cos_itot)
        mean_cos_dW = _nanmean(epoch_cos_dW)
        mean_energy_ratio = (mean_cos_dW ** 2) if not math.isnan(mean_cos_dW) else float('nan')

        history['train_loss'].append(float(train_loss))
        history['train_acc'].append(float(train_acc))
        history['val_loss'].append(float(val_loss))
        history['val_acc'].append(float(val_acc))
        history['avg_spike_rate'].append(float(avg_epoch_rate))
        history['cos_itot'].append(float(mean_cos_itot))
        history['cos_dW'].append(float(mean_cos_dW))
        history['energy_ratio'].append(float(mean_energy_ratio))

        scheduler.step()

        print(f'  Epoch {epoch + 1}/{epochs}: '
              f'Train loss:{train_loss:.4f}, Val loss:{val_loss:.4f}, '
              f'Train acc:{train_acc: .2f}%, Val acc:{val_acc:.2f}%, '
              f'Avg rho:{avg_epoch_rate:.6f}, '
              f'⟨cos(Itot,-g)⟩:{mean_cos_itot:.3f}, ⟨cos(dW,-g)⟩:{mean_cos_dW:.3f}')
    print("--- Diagnostic Run Complete ---")
    return history


# def objective(trial, base_params,neighbor_map, seeds):
#     """
#     这是Optuna每次尝试需要运行的函数。
#     它会从Optuna获取一组超参数，运行一次实验，并返回一个评估指标。
#     """
#     # Optuna会在每次调用此函数时，使用一个独立的随机状态，无需手动设置seed
#     # --- a. 定义超参数的搜索空间 ---
#     # Optuna会从这个空间中智能地进行采样
#     grad_gain = trial.suggest_float("grad_gain", 1e5, 2e7, log=True)  # 在10万到2000万之间进行对数采样
#     lr = trial.suggest_float("lr", 1e-4, 1e-1, log=True)  # 在0.0001到0.1之间进行对数采样
#
#     print(f"\n--- [Trial #{trial.number}] Testing: grad_gain={grad_gain:.2f}, lr={lr:.5f} ---")
#
#     # --- b. 运行一次标准的、20个epoch的训练 ---
#
#     histories = []
#
#     for seed in seeds:
#         set_seed(seed)
#
#         train_loader, val_loader,_,_, num_classes = get_dataloaders(
#             CONFIG['dataset'], CONFIG['batch_size'], CONFIG['val_split'],
#             seed=seed
#         )
#         model = get_model(CONFIG['model'], num_classes)  # .to(device)
#
#         # --- 使用HHDyn和所有正确的参数 ---
#         optimizer_params = base_params.copy()
#         optimizer_params['grad_gain'] = grad_gain
#         optimizer_params['lr'] = lr
#         optimizer_params['map'] = neighbor_map
#
#         optimizer = HHDyn(model.parameters(), **optimizer_params) # 这是您真实的优化器
#         scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
#         history = diagnostic_run(model, optimizer,scheduler, train_loader, val_loader, CONFIG['tuning_epochs'])
#         histories.append(history)
#     # --- c. 返回您想要最大化的指标 ---
#     final_accuracies = [h['val_acc'][-1] for h in histories]
#     if not final_accuracies:
#         print(f"  > Trial Summary: FAILED. No valid accuracies were returned from any run.")
#         return 0.0  # 或者 -1.0
#     # all_epoch_accuracies = [acc for h in histories for acc in h['val_acc']]
#     # mean_overall_accuracy = np.mean(all_epoch_accuracies)
#     mean_accuracy = np.mean(final_accuracies)
#     std_accuracy = np.std(final_accuracies)
#     lower_quartile_accuracy = np.quantile(final_accuracies, 0.25)
#
#     # 将所有详细结果都存入trial，方便后续分析
#     # trial.set_user_attr("mean_overall_accuracy", mean_overall_accuracy)
#     trial.set_user_attr("mean_accuracy", mean_accuracy)
#     trial.set_user_attr("std_accuracy", std_accuracy)
#     trial.set_user_attr("all_final_accuracies", final_accuracies)
#     trial.set_user_attr("all_histories", histories)
#     return lower_quartile_accuracy


def train_and_validate(model, optimizer, scheduler, train_loader, val_loader, test_loader, epochs, save_path):
    """
    运行完整的训练流程：
    1. 训练每个 Epoch
    2. 在验证集上评估
    3. 保存验证集精度最高的模型 (Best Model)
    4. 训练结束后，加载最佳模型，在测试集上跑一次最终结果
    """
    criterion = nn.CrossEntropyLoss()
    best_val_acc = 0.0
    best_epoch = 0

    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [],
        'test_acc': 0.0,
        'write_volumes': [],'compute_volumes': []
    }

    print(f"Start Training for {epochs} epochs...")

    # --- Loop Epochs ---
    for epoch in range(epochs):
        # 1. Train
        model.train()
        train_loss_meter = 0
        correct = 0
        total = 0

        is_sam = (optimizer.__class__.__name__ == 'SAM')  # 简单判断是否是 SAM

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            if is_sam:
                # SAM Step
                # P1: Forward + Backward
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.first_step(zero_grad=True)

                # P2: Forward + Backward again
                criterion(model(inputs), targets).backward()
                optimizer.second_step(zero_grad=True)
            else:
                # Standard / HHDyn Step
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

            train_loss_meter += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        train_loss = train_loss_meter / total
        train_acc = 100. * correct / total

        # 2. Validation
        val_loss, val_acc = evaluate(model, val_loader, criterion)

        # 3. Schedule
        if scheduler:
            scheduler.step()

        # 4. Check Best Model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            # 保存最佳模型权重
            torch.save(model.state_dict(), save_path)

        # Log
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)


        if (epoch + 1) % 1 == 0:
            print(
                f"  Ep {epoch + 1}/{epochs} | Train loss: {train_loss:.2f},  Train_Acc: {train_acc:.2f}% | "
                f" Val loss: {val_loss:.2f}, Val_Acc: {val_acc:.2f}% (Best: {best_val_acc:.2f}%)")

        # 膜电位快照，给base_tau的选择一个参考
        # if epoch == 0:
        #    analyze_initial_tau_criteria(optimizer)
        #    break

    print(f"Training Finished. Best Validation Acc: {best_val_acc:.2f}% at Epoch {best_epoch + 1}")

    # --- Final Test ---
    # 加载验证集上表现最好的那个模型，去跑测试集
    # 这才是论文 Table 1 应该填的数据
    print("Loading best model for Final Testing...")
    model.load_state_dict(torch.load(save_path))
    test_loss, test_acc = evaluate(model, test_loader, criterion)
    history['test_acc'] = test_acc
    print(f"Final Test Accuracy: {test_acc:.2f}%")

    return history

import math
import torch


def _safe_mean(x_sum, x_cnt):
    if x_cnt == 0:
        return None
    return x_sum / x_cnt


def _compute_full_grad_norm(model):
    """
    全模型梯度范数（L2）。
    在 loss.backward() 之后、optimizer.step() 之前调用。
    """
    total_sq = 0.0
    for p in model.parameters():
        if p.grad is not None:
            g = p.grad.detach()
            total_sq += float((g * g).sum().item())
    return math.sqrt(total_sq)


def _compute_param_norm(model):
    """
    全模型参数范数（L2）。
    建议每个 epoch 末调用一次。
    """
    total_sq = 0.0
    for p in model.parameters():
        if p.requires_grad:
            w = p.detach()
            total_sq += float((w * w).sum().item())
    return math.sqrt(total_sq)


def _snapshot_init_params_cpu(model):
    """
    保存训练开始时的初始参数快照到 CPU。
    用于后续计算 delta_from_init。
    """
    return {
        id(p): p.detach().cpu().clone()
        for p in model.parameters()
        if p.requires_grad
    }


def _compute_delta_from_init_cpu(model, init_params_cpu):
    """
    计算当前参数相对初始参数的整体偏移量（L2）。
    注意这里会把当前参数搬到 CPU 上比较，建议每个 epoch 末调用一次。
    """
    total_sq = 0.0
    for p in model.parameters():
        if not p.requires_grad:
            continue
        p0 = init_params_cpu.get(id(p), None)
        if p0 is None:
            continue
        cur = p.detach().cpu()
        diff = cur - p0
        total_sq += float((diff * diff).sum().item())
    return math.sqrt(total_sq)


def _compute_E_diff_proxy(model, optimizer):
    """
    计算一个与当前 diff 构造一致的结构能量 proxy：

        E_diff = mean( Laplacian(p)^2 )

    只对 optimizer.map 中支持 diff 的参数统计：
      - intra_kernel_conv (4D)
      - intra_vector_fc   (2D)

    返回:
        float 或 None
    """
    if not hasattr(optimizer, "map"):
        return None
    if not hasattr(optimizer, "_lap2d_tensor"):
        return None
    if not hasattr(optimizer, "_lap1d_row"):
        return None
    if not hasattr(optimizer, "ipd_config"):
        return None

    cfg = optimizer.ipd_config
    if cfg is None:
        return None

    lap_r = cfg.get("lap_r", 1)
    lap_neigh = cfg.get("lap_neigh", "l1")
    lap_padding = cfg.get("lap_padding", "reflect")

    total_sq = 0.0
    total_n = 0

    for p in model.parameters():
        if not p.requires_grad:
            continue

        info = optimizer.map.get(id(p), {"type": "none"})
        layer_type = info.get("type", "none")

        raw = None
        if layer_type == "intra_kernel_conv" and p.dim() == 4:
            raw = optimizer._lap2d_tensor(
                p.detach(), lap_r, lap_neigh, lap_padding
            )
        elif layer_type == "intra_vector_fc" and p.dim() == 2:
            raw = optimizer._lap1d_row(
                p.detach(), lap_r, lap_padding
            )

        if raw is None:
            continue

        total_sq += float((raw * raw).sum().item())
        total_n += raw.numel()

    if total_n == 0:
        return None
    return total_sq / total_n

# def final_train(model, optimizer, scheduler, optimizer_name, train_loader, test_loader, epochs, save_path):
#
#     criterion = nn.CrossEntropyLoss()
#     device = next(model.parameters()).device
#     best_test_acc = 0.0
#     train_start_time = time.time()
#
#     # -----------------------------
#     # init snapshot for delta_from_init
#     # -----------------------------
#     init_params_cpu = _snapshot_init_params_cpu(model)
#
#     # -----------------------------
#     # history
#     # -----------------------------
#     history = {
#         'train_loss': [],
#         'train_acc': [],
#         'test_loss': [],
#         'test_acc': [],
#         'best_test_acc': 0.0,
#         'last_test_acc': 0.0,
#
#         # ---- epoch-level extra stats ----
#         'lr': [],
#         'grad_norm_full': [],          # 全模型 grad norm（epoch内step平均）
#         'grad_norm_active': [],        # 进入 IPD force 的 active/support 子集 grad norm（epoch内step平均）
#         'diff_force_norm': [],
#         'coul_force_norm': [],
#         'force_norm': [],
#         'rho_diff': [],
#         'rho_force': [],
#         'cos_F_vs_-g': [],
#         'cos_geff_prev': [],
#         'param_norm': [],
#         'delta_from_init': [],
#         'E_diff': [],
#         'force_steps': [],
#         'force_params': [],
#         'ipd_n_steps': [],
#         'epoch_time_sec': [],
#         'train_steps': [],
#         'train_samples': [],
#         'total_train_time_sec': 0.0,
#     }
#
#     print(f'\nStarting Final training with {optimizer_name}...')
#
#     for epoch in range(epochs):
#         epoch_start_time = time.time()
#         num_train_steps = len(train_loader)
#         model.train()
#
#         # ---- reset IPD stats each epoch ----
#         if hasattr(optimizer, "reset_ipd_stats"):
#             optimizer.reset_ipd_stats()
#         else:
#             if hasattr(optimizer, "reset_rho_stats"):
#                 optimizer.reset_rho_stats()
#             if hasattr(optimizer, "reset_traj_stats"):
#                 optimizer.reset_traj_stats()
#
#         running_loss = 0.0
#         correct = 0
#         total_samples = 0
#
#         # ---- step accumulators for this epoch ----
#         full_grad_norm_sum = 0.0
#         full_grad_norm_cnt = 0
#
#         active_grad_norm_sum = 0.0
#         active_grad_norm_cnt = 0
#
#         diff_force_norm_sum = 0.0
#         diff_force_norm_cnt = 0
#
#         coul_force_norm_sum = 0.0
#         coul_force_norm_cnt = 0
#
#         force_norm_sum = 0.0
#         force_norm_cnt = 0
#
#         rho_diff_sum = 0.0
#         rho_diff_cnt = 0
#
#         rho_force_sum = 0.0
#         rho_force_cnt = 0
#
#         cos_Fg_sum = 0.0
#         cos_Fg_cnt = 0
#
#         cos_geff_sum = 0.0
#         cos_geff_cnt = 0
#
#         # -----------------------------
#         # train loop
#         # -----------------------------
#         for data, target in train_loader:
#             data, target = data.to(device), target.to(device)
#             current_batch_size = data.size(0)
#
#             if optimizer_name == 'SAM':
#                 optimizer.zero_grad()
#                 output = model(data)
#                 loss = criterion(output, target)
#                 loss.backward()
#
#                 # 全模型 grad norm：记录 first backward 的结果
#                 full_grad_norm = _compute_full_grad_norm(model)
#                 full_grad_norm_sum += full_grad_norm
#                 full_grad_norm_cnt += 1
#
#                 optimizer.first_step(zero_grad=True)
#
#                 criterion(model(data), target).backward()
#                 optimizer.second_step(zero_grad=True)
#
#             else:
#                 optimizer.zero_grad()
#                 output = model(data)
#                 loss = criterion(output, target)
#                 loss.backward()
#
#                 # ---- full-model grad norm (before optimizer.step) ----
#                 full_grad_norm = _compute_full_grad_norm(model)
#                 full_grad_norm_sum += full_grad_norm
#                 full_grad_norm_cnt += 1
#
#                 optimizer.step()
#
#                 # ---- read IPD step stats if available ----
#                 if hasattr(optimizer, "get_last_step_stats"):
#                     step_stats = optimizer.get_last_step_stats()
#
#                     v = step_stats.get("grad_norm_active", None)
#                     if v is not None:
#                         active_grad_norm_sum += float(v)
#                         active_grad_norm_cnt += 1
#
#                     v = step_stats.get("diff_force_norm", None)
#                     if v is not None:
#                         diff_force_norm_sum += float(v)
#                         diff_force_norm_cnt += 1
#
#                     v = step_stats.get("coul_force_norm", None)
#                     if v is not None:
#                         coul_force_norm_sum += float(v)
#                         coul_force_norm_cnt += 1
#
#                     v = step_stats.get("force_norm", None)
#                     if v is not None:
#                         force_norm_sum += float(v)
#                         force_norm_cnt += 1
#
#                     v = step_stats.get("rho_diff", None)
#                     if v is not None:
#                         rho_diff_sum += float(v)
#                         rho_diff_cnt += 1
#
#                     v = step_stats.get("rho_force", None)
#                     if v is not None:
#                         rho_force_sum += float(v)
#                         rho_force_cnt += 1
#
#                     v = step_stats.get("cos_F_vs_-g", None)
#                     if v is not None:
#                         cos_Fg_sum += float(v)
#                         cos_Fg_cnt += 1
#
#                     v = step_stats.get("cos_geff_prev", None)
#                     if v is not None:
#                         cos_geff_sum += float(v)
#                         cos_geff_cnt += 1
#
#             # ---- standard running metrics ----
#             running_loss += loss.item() * current_batch_size
#             pred = output.argmax(dim=1, keepdim=True)
#             correct += pred.eq(target.view_as(pred)).sum().item()
#             total_samples += current_batch_size
#
#         # -----------------------------
#         # train metrics
#         # -----------------------------
#         epoch_time = time.time() - epoch_start_time
#         epoch_train_acc = 100.0 * correct / total_samples
#         epoch_train_loss = running_loss / total_samples
#
#         history['train_acc'].append(epoch_train_acc)
#         history['train_loss'].append(epoch_train_loss)
#
#
#         # -----------------------------
#         # evaluate
#         # -----------------------------
#         test_loss, test_acc = evaluate(model, test_loader, criterion)
#         history['test_loss'].append(test_loss)
#         history['test_acc'].append(test_acc)
#
#         if test_acc > best_test_acc:
#             best_test_acc = test_acc
#             torch.save(model.state_dict(), save_path)
#
#         if scheduler is not None:
#             scheduler.step()
#             current_lr = scheduler.get_last_lr()[0]
#         else:
#             current_lr = optimizer.param_groups[0]["lr"]
#
#         # -----------------------------
#         # epoch-end extra stats
#         # -----------------------------
#         epoch_full_grad_norm = _safe_mean(full_grad_norm_sum, full_grad_norm_cnt)
#         epoch_active_grad_norm = _safe_mean(active_grad_norm_sum, active_grad_norm_cnt)
#         epoch_diff_force_norm = _safe_mean(diff_force_norm_sum, diff_force_norm_cnt)
#         epoch_coul_force_norm = _safe_mean(coul_force_norm_sum, coul_force_norm_cnt)
#         epoch_force_norm = _safe_mean(force_norm_sum, force_norm_cnt)
#         epoch_rho_diff = _safe_mean(rho_diff_sum, rho_diff_cnt)
#         epoch_rho_force = _safe_mean(rho_force_sum, rho_force_cnt)
#         epoch_cos_Fg = _safe_mean(cos_Fg_sum, cos_Fg_cnt)
#         epoch_cos_geff = _safe_mean(cos_geff_sum, cos_geff_cnt)
#
#         epoch_param_norm = _compute_param_norm(model)
#         epoch_delta_from_init = _compute_delta_from_init_cpu(model, init_params_cpu)
#         epoch_E_diff = _compute_E_diff_proxy(model, optimizer)
#
#         # 从 optimizer 里拿 running summary（更适合 force_steps / force_params / n_steps）
#         force_steps = 0
#         force_params = 0
#         ipd_n_steps = 0
#
#         if hasattr(optimizer, "get_ipd_stats"):
#             stats = optimizer.get_ipd_stats()
#
#             # 如果你后来按我上一条改了 optimizer，这里优先用新字段
#             if epoch_rho_force is None:
#                 epoch_rho_force = stats.get("rho_force_mean", stats.get("rho_mean", None))
#             if epoch_rho_diff is None:
#                 epoch_rho_diff = stats.get("rho_diff_mean", None)
#             if epoch_cos_Fg is None:
#                 epoch_cos_Fg = stats.get("cos_F_vs_-g_mean", None)
#             if epoch_cos_geff is None:
#                 epoch_cos_geff = stats.get("cos_geff_curv_mean", None)
#             if epoch_active_grad_norm is None:
#                 epoch_active_grad_norm = stats.get("grad_norm_active_mean", None)
#             if epoch_force_norm is None:
#                 epoch_force_norm = stats.get("force_norm_mean", None)
#             if epoch_diff_force_norm is None:
#                 epoch_diff_force_norm = stats.get("diff_force_norm_mean", None)
#
#             force_steps = stats.get("force_active_steps", 0)
#             force_params = stats.get("force_active_params", 0)
#             ipd_n_steps = stats.get("n_steps", 0)
#
#         # -----------------------------
#         # save epoch stats to history
#         # -----------------------------
#
#         history['train_steps'].append(num_train_steps)
#         history['train_samples'].append(total_samples)
#         history['lr'].append(current_lr)
#         history['grad_norm_full'].append(epoch_full_grad_norm)
#         history['grad_norm_active'].append(epoch_active_grad_norm)
#         history['diff_force_norm'].append(epoch_diff_force_norm)
#         history['coul_force_norm'].append(epoch_coul_force_norm)
#         history['force_norm'].append(epoch_force_norm)
#         history['rho_diff'].append(epoch_rho_diff)
#         history['rho_force'].append(epoch_rho_force)
#         history['cos_F_vs_-g'].append(epoch_cos_Fg)
#         history['cos_geff_prev'].append(epoch_cos_geff)
#         history['param_norm'].append(epoch_param_norm)
#         history['delta_from_init'].append(epoch_delta_from_init)
#         history['E_diff'].append(epoch_E_diff)
#         history['force_steps'].append(force_steps)
#         history['force_params'].append(force_params)
#         history['ipd_n_steps'].append(ipd_n_steps)
#         history['epoch_time_sec'].append(epoch_time)
#
#         # -----------------------------
#         # logging
#         # -----------------------------
#         log_msg = (
#             f"Epoch {epoch + 1}/{epochs} | "
#             f"Train Loss: {epoch_train_loss:.4f} | "
#             f"Train Acc: {epoch_train_acc:.2f}% | "
#             f"Test Loss: {test_loss:.4f} | "
#             f"Test Acc: {test_acc:.2f}% (Best: {best_test_acc:.2f}%) | "
#             f"LR: {current_lr:.6f} |"
#             f" | epoch_time={epoch_time:.1f}s"
#         )
#
#         if epoch_full_grad_norm is None:
#             log_msg += " | grad_full=NA"
#         else:
#             log_msg += f" | grad_full={epoch_full_grad_norm:.3e}"
#
#         if epoch_active_grad_norm is None:
#             log_msg += " | grad_active=NA"
#         else:
#             log_msg += f" | grad_active={epoch_active_grad_norm:.3e}"
#
#         if epoch_diff_force_norm is None:
#             log_msg += " | F_diff=NA"
#         else:
#             log_msg += f" | F_diff={epoch_diff_force_norm:.3e}"
#
#         if epoch_rho_diff is None:
#             log_msg += " | rho_diff=NA"
#         else:
#             log_msg += f" | rho_diff={epoch_rho_diff:.3e}"
#
#         if epoch_rho_force is None:
#             log_msg += " | rho_force=NA"
#         else:
#             log_msg += f" | rho_force={epoch_rho_force:.3e}"
#
#         if epoch_cos_Fg is None:
#             log_msg += " | cos(F,-g)=NA"
#         else:
#             log_msg += f" | cos(F,-g)={epoch_cos_Fg:+.3f}"
#
#         if epoch_cos_geff is None:
#             log_msg += " | cos(geff_t,geff_t-1)=NA"
#         else:
#             log_msg += f" | cos(geff_t,geff_t-1)={epoch_cos_geff:+.3f}"
#
#         if epoch_param_norm is None:
#             log_msg += " | ||w||=NA"
#         else:
#             log_msg += f" | ||w||={epoch_param_norm:.3e}"
#
#         if epoch_delta_from_init is None:
#             log_msg += " | delta_init=NA"
#         else:
#             log_msg += f" | delta_init={epoch_delta_from_init:.3e}"
#
#         if epoch_E_diff is None:
#             log_msg += " | E_diff=NA"
#         else:
#             log_msg += f" | E_diff={epoch_E_diff:.3e}"
#
#         log_msg += f" | force_steps={force_steps} | force_params={force_params}"
#
#         print(log_msg)
#
#
#     # -----------------------------
#     # save last checkpoint
#     # -----------------------------
#     base_name, ext = os.path.splitext(save_path)
#     last_save_path = f"{base_name}_last{ext}"
#     torch.save(model.state_dict(), last_save_path)
#     total_train_time = time.time() - train_start_time
#     history['total_train_time_sec'] = total_train_time
#
#     print(f"\nTraining complete.")
#     print(f"Best model (Acc: {best_test_acc:.2f}%) saved to: {save_path}")
#     print(f"Last model (Acc: {test_acc:.2f}%) saved to: {last_save_path}")
#
#     history['best_test_acc'] = best_test_acc
#     history['last_test_acc'] = test_acc
#     return history
def final_train(model, optimizer, scheduler, optimizer_name, train_loader, test_loader, epochs, save_path):
    criterion = nn.CrossEntropyLoss()
    device = next(model.parameters()).device
    best_test_acc = 0.0
    total_train_start_time = time.time()

    init_params_cpu = _snapshot_init_params_cpu(model)

    history = {
        'train_loss': [],
        'train_acc': [],
        'test_loss': [],
        'test_acc': [],
        'best_test_acc': 0.0,
        'last_test_acc': 0.0,

        'lr': [],
        'epoch_time_sec': [],
        'train_steps': [],
        'train_samples': [],
        'total_train_time_sec': 0.0,

        'grad_norm_full': [],
        'grad_norm_active': [],
        'diff_force_norm': [],
        'coul_force_norm': [],
        'force_norm': [],
        'rho_diff': [],
        'rho_coul': [],
        'rho_force': [],
        'cos_F_vs_-g': [],
        'cos_geff_prev': [],
        'param_norm': [],
        'delta_from_init': [],
        'E_diff': [],
        'gram_offdiag_abs_mean': [],
        'avg_max_abs_cos': [],
        'force_steps': [],
        'force_params': [],
        'ipd_n_steps': [],
    }

    print(f'\nStarting Final training with {optimizer_name}...')

    for epoch in range(epochs):
        epoch_start_time = time.time()
        num_train_steps = len(train_loader)

        model.train()

        if hasattr(optimizer, "reset_ipd_stats"):
            optimizer.reset_ipd_stats()
        else:
            if hasattr(optimizer, "reset_rho_stats"):
                optimizer.reset_rho_stats()
            if hasattr(optimizer, "reset_traj_stats"):
                optimizer.reset_traj_stats()

        running_loss = 0.0
        correct = 0
        total_samples = 0

        full_grad_norm_sum = 0.0
        full_grad_norm_cnt = 0

        active_grad_norm_sum = 0.0
        active_grad_norm_cnt = 0

        diff_force_norm_sum = 0.0
        diff_force_norm_cnt = 0

        coul_force_norm_sum = 0.0
        coul_force_norm_cnt = 0

        force_norm_sum = 0.0
        force_norm_cnt = 0

        rho_diff_sum = 0.0
        rho_diff_cnt = 0

        rho_coul_sum = 0.0
        rho_coul_cnt = 0

        rho_force_sum = 0.0
        rho_force_cnt = 0

        cos_Fg_sum = 0.0
        cos_Fg_cnt = 0

        cos_geff_sum = 0.0
        cos_geff_cnt = 0

        gram_offdiag_abs_mean_sum = 0.0
        gram_offdiag_abs_mean_cnt = 0

        avg_max_abs_cos_sum = 0.0
        avg_max_abs_cos_cnt = 0

        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            current_batch_size = data.size(0)

            if optimizer_name == 'SAM':
                optimizer.zero_grad()

                output = model(data)
                loss = criterion(output, target)
                loss.backward()

                full_grad_norm = _compute_full_grad_norm(model)
                full_grad_norm_sum += full_grad_norm
                full_grad_norm_cnt += 1

                optimizer.first_step(zero_grad=True)
                criterion(model(data), target).backward()
                optimizer.second_step(zero_grad=True)

            else:
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()

                full_grad_norm = _compute_full_grad_norm(model)
                full_grad_norm_sum += full_grad_norm
                full_grad_norm_cnt += 1

                optimizer.step()

                if hasattr(optimizer, "get_last_step_stats"):
                    step_stats = optimizer.get_last_step_stats()

                    v = step_stats.get("grad_norm_active", None)
                    if v is not None:
                        active_grad_norm_sum += float(v)
                        active_grad_norm_cnt += 1

                    v = step_stats.get("diff_force_norm", None)
                    if v is not None:
                        diff_force_norm_sum += float(v)
                        diff_force_norm_cnt += 1

                    v = step_stats.get("coul_force_norm", None)
                    if v is not None:
                        coul_force_norm_sum += float(v)
                        coul_force_norm_cnt += 1

                    v = step_stats.get("force_norm", None)
                    if v is not None:
                        force_norm_sum += float(v)
                        force_norm_cnt += 1

                    v = step_stats.get("rho_diff", None)
                    if v is not None:
                        rho_diff_sum += float(v)
                        rho_diff_cnt += 1

                    v = step_stats.get("rho_coul", None)
                    if v is not None:
                        rho_coul_sum += float(v)
                        rho_coul_cnt += 1

                    v = step_stats.get("rho_force", None)
                    if v is not None:
                        rho_force_sum += float(v)
                        rho_force_cnt += 1

                    v = step_stats.get("cos_F_vs_-g", None)
                    if v is not None:
                        cos_Fg_sum += float(v)
                        cos_Fg_cnt += 1

                    v = step_stats.get("cos_geff_prev", None)
                    if v is not None:
                        cos_geff_sum += float(v)
                        cos_geff_cnt += 1

                    v = step_stats.get("gram_offdiag_abs_mean", None)
                    if v is not None:
                        gram_offdiag_abs_mean_sum += float(v)
                        gram_offdiag_abs_mean_cnt += 1

                    v = step_stats.get("avg_max_abs_cos", None)
                    if v is not None:
                        avg_max_abs_cos_sum += float(v)
                        avg_max_abs_cos_cnt += 1

            running_loss += loss.item() * current_batch_size
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total_samples += current_batch_size

        epoch_train_acc = 100.0 * correct / total_samples
        epoch_train_loss = running_loss / total_samples

        history['train_acc'].append(epoch_train_acc)
        history['train_loss'].append(epoch_train_loss)

        test_loss, test_acc = evaluate(model, test_loader, criterion)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            torch.save(model.state_dict(), save_path)

        if scheduler is not None:
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
        else:
            current_lr = optimizer.param_groups[0]["lr"]

        epoch_time = time.time() - epoch_start_time

        epoch_full_grad_norm = _safe_mean(full_grad_norm_sum, full_grad_norm_cnt)
        epoch_active_grad_norm = _safe_mean(active_grad_norm_sum, active_grad_norm_cnt)
        epoch_diff_force_norm = _safe_mean(diff_force_norm_sum, diff_force_norm_cnt)
        epoch_coul_force_norm = _safe_mean(coul_force_norm_sum, coul_force_norm_cnt)
        epoch_force_norm = _safe_mean(force_norm_sum, force_norm_cnt)
        epoch_rho_diff = _safe_mean(rho_diff_sum, rho_diff_cnt)
        epoch_rho_coul = _safe_mean(rho_coul_sum, rho_coul_cnt)
        epoch_rho_force = _safe_mean(rho_force_sum, rho_force_cnt)
        epoch_cos_Fg = _safe_mean(cos_Fg_sum, cos_Fg_cnt)
        epoch_cos_geff = _safe_mean(cos_geff_sum, cos_geff_cnt)
        epoch_gram_offdiag_abs_mean = _safe_mean(gram_offdiag_abs_mean_sum, gram_offdiag_abs_mean_cnt)
        epoch_avg_max_abs_cos = _safe_mean(avg_max_abs_cos_sum, avg_max_abs_cos_cnt)

        epoch_param_norm = _compute_param_norm(model)
        epoch_delta_from_init = _compute_delta_from_init_cpu(model, init_params_cpu)
        epoch_E_diff = _compute_E_diff_proxy(model, optimizer)

        force_steps = 0
        force_params = 0
        ipd_n_steps = 0

        if hasattr(optimizer, "get_ipd_stats"):
            stats = optimizer.get_ipd_stats()

            if epoch_rho_force is None:
                epoch_rho_force = stats.get("rho_force_mean", None)
            if epoch_rho_diff is None:
                epoch_rho_diff = stats.get("rho_diff_mean", None)
            if epoch_rho_coul is None:
                epoch_rho_coul = stats.get("rho_coul_mean", None)

            if epoch_cos_Fg is None:
                epoch_cos_Fg = stats.get("cos_F_vs_-g_mean", None)
            if epoch_cos_geff is None:
                epoch_cos_geff = stats.get("cos_geff_curv_mean", None)
            if epoch_active_grad_norm is None:
                epoch_active_grad_norm = stats.get("grad_norm_active_mean", None)
            if epoch_force_norm is None:
                epoch_force_norm = stats.get("force_norm_mean", None)
            if epoch_diff_force_norm is None:
                epoch_diff_force_norm = stats.get("diff_force_norm_mean", None)
            if epoch_coul_force_norm is None:
                epoch_coul_force_norm = stats.get("coul_force_norm_mean", None)

            if epoch_gram_offdiag_abs_mean is None:
                epoch_gram_offdiag_abs_mean = stats.get("gram_offdiag_abs_mean", None)
            if epoch_avg_max_abs_cos is None:
                epoch_avg_max_abs_cos = stats.get("avg_max_abs_cos_mean", None)

            force_steps = stats.get("force_active_steps", 0)
            force_params = stats.get("force_active_params", 0)
            ipd_n_steps = stats.get("n_steps", 0)

        history['lr'].append(current_lr)
        history['epoch_time_sec'].append(epoch_time)
        history['train_steps'].append(num_train_steps)
        history['train_samples'].append(total_samples)

        history['grad_norm_full'].append(epoch_full_grad_norm)
        history['grad_norm_active'].append(epoch_active_grad_norm)
        history['diff_force_norm'].append(epoch_diff_force_norm)
        history['coul_force_norm'].append(epoch_coul_force_norm)
        history['force_norm'].append(epoch_force_norm)
        history['rho_diff'].append(epoch_rho_diff)
        history['rho_coul'].append(epoch_rho_coul)
        history['rho_force'].append(epoch_rho_force)
        history['cos_F_vs_-g'].append(epoch_cos_Fg)
        history['cos_geff_prev'].append(epoch_cos_geff)
        history['param_norm'].append(epoch_param_norm)
        history['delta_from_init'].append(epoch_delta_from_init)
        history['E_diff'].append(epoch_E_diff)
        history['gram_offdiag_abs_mean'].append(epoch_gram_offdiag_abs_mean)
        history['avg_max_abs_cos'].append(epoch_avg_max_abs_cos)
        history['force_steps'].append(force_steps)
        history['force_params'].append(force_params)
        history['ipd_n_steps'].append(ipd_n_steps)

        log_msg = (
            f"Epoch {epoch + 1}/{epochs} | "
            f"Train Loss: {epoch_train_loss:.4f} | "
            f"Train Acc: {epoch_train_acc:.2f}% | "
            f"Test Loss: {test_loss:.4f} | "
            f"Test Acc: {test_acc:.2f}% (Best: {best_test_acc:.2f}%) | "
            f"LR: {current_lr:.6f}"
        )

        log_msg += f" | grad_full={epoch_full_grad_norm:.3e}" if epoch_full_grad_norm is not None else " | grad_full=NA"
        log_msg += f" | grad_active={epoch_active_grad_norm:.3e}" if epoch_active_grad_norm is not None else " | grad_active=NA"

        log_msg += f" | F_diff={epoch_diff_force_norm:.3e}" if epoch_diff_force_norm is not None else " | F_diff=NA"
        log_msg += f" | F_coul={epoch_coul_force_norm:.3e}" if epoch_coul_force_norm is not None else " | F_coul=NA"
        log_msg += f" | F_total={epoch_force_norm:.3e}" if epoch_force_norm is not None else " | F_total=NA"

        log_msg += f" | rho_diff={epoch_rho_diff:.3e}" if epoch_rho_diff is not None else " | rho_diff=NA"
        log_msg += f" | rho_coul={epoch_rho_coul:.3e}" if epoch_rho_coul is not None else " | rho_coul=NA"
        log_msg += f" | rho_force={epoch_rho_force:.3e}" if epoch_rho_force is not None else " | rho_force=NA"

        log_msg += f" | cos(F,-g)={epoch_cos_Fg:+.3f}" if epoch_cos_Fg is not None else " | cos(F,-g)=NA"
        log_msg += f" | cos(geff_t,geff_t-1)={epoch_cos_geff:+.3f}" if epoch_cos_geff is not None else " | cos(geff_t,geff_t-1)=NA"

        log_msg += f" | ||w||={epoch_param_norm:.3e}" if epoch_param_norm is not None else " | ||w||=NA"
        log_msg += f" | delta_init={epoch_delta_from_init:.3e}" if epoch_delta_from_init is not None else " | delta_init=NA"
        log_msg += f" | E_diff={epoch_E_diff:.3e}" if epoch_E_diff is not None else " | E_diff=NA"

        log_msg += f" | gram_offdiag={epoch_gram_offdiag_abs_mean:.3f}" if epoch_gram_offdiag_abs_mean is not None else " | gram_offdiag=NA"
        log_msg += f" | max_abs_cos={epoch_avg_max_abs_cos:.3f}" if epoch_avg_max_abs_cos is not None else " | max_abs_cos=NA"

        log_msg += f" | force_steps={force_steps} | force_params={force_params}"
        log_msg += f" | epoch_time={epoch_time:.1f}s"

        print(log_msg)

    base_name, ext = os.path.splitext(save_path)
    last_save_path = f"{base_name}_last{ext}"
    torch.save(model.state_dict(), last_save_path)

    total_train_time = time.time() - total_train_start_time

    print(f"\nTraining complete.")
    print(f"Best model (Acc: {best_test_acc:.2f}%) saved to: {save_path}")
    print(f"Last model (Acc: {test_acc:.2f}%) saved to: {last_save_path}")

    history['best_test_acc'] = float(best_test_acc)
    history['last_test_acc'] = float(test_acc)
    history['total_train_time_sec'] = float(total_train_time)

    return history


def analyze_initial_tau_criteria(optimizer):
    print("\n[Analysis] Collecting Membrane Potentials (V) to determine base_tau...")

    all_abs_v = []

    for group in optimizer.param_groups:
        for p in group['params']:
            state = optimizer.state[p]
            if 'potential' in state:
                # 获取绝对值，因为我们关心的是能量强度
                v = state['potential'].abs().detach().cpu().view(-1)
                all_abs_v.append(v)

    if not all_abs_v:
        print("Error: No potentials found. Make sure you ran at least one step.")
        return

    # 拼接所有参数的 V
    all_abs_v = torch.cat(all_abs_v).numpy()

    # === 核心：计算分位数 ===
    # P90: 只有前 10% 强的神经元能激活
    # P95: 只有前 5% 强的神经元能激活 (推荐)
    # P99: 只有前 1% 强的神经元能激活 (极度稀疏)

    p50 = np.percentile(all_abs_v, 50)
    p80 = np.percentile(all_abs_v, 80)
    p90 = np.percentile(all_abs_v, 90)
    p95 = np.percentile(all_abs_v, 95)
    p98 = np.percentile(all_abs_v, 98)
    p99 = np.percentile(all_abs_v, 99)

    print(f"\n====== V Distribution Statistics (Epoch 0) ======")
    print(f"Total Params Count: {len(all_abs_v)}")
    print(f"Max |V| observed:   {np.max(all_abs_v):.6f}")
    print(f"Mean |V| observed:  {np.mean(all_abs_v):.6f}")
    print("-" * 40)
    print("Candidates for base_tau:")
    print(f"If you want 50% active (P50): base_tau ≈ {p50:.6f}")
    print(f"If you want 20% active (P80): base_tau ≈ {p80:.6f}")
    print(f"If you want 10% active (P90): base_tau ≈ {p90:.6f}")
    print(f"If you want 5%  active (P95): base_tau ≈ {p95:.6f}  <-- 推荐起步点")
    print(f"If you want 2%  active (P98): base_tau ≈ {p98:.6f}")
    print(f"If you want 1%  active (P99): base_tau ≈ {p99:.6f}")

def plot_weight_distributions(optimizers, model_name, num_classes):
    # load saved best model and plot their weight distribution
    print("\n--- Generating Weight Distribution ---")
    num_opts = len(optimizers )
    if num_opts == 0:
        print('No Optimizer!')
        return
    cols = 2
    rows = math.ceil(num_opts / cols)
    subplot_titles = [f'Distribution of {opt_name} ' for opt_name in optimizers]
    fig = make_subplots(
        rows=rows,
        cols=cols,
        subplot_titles=subplot_titles,
        shared_xaxes=True,
        shared_yaxes=True
    )
    for i, opt_name in enumerate(optimizers):
        row = i // cols + 1
        col = i % cols + 1

        model = get_model(opt_name, num_classes)
        model_path = f'{opt_name}_final_model.pth'

        if os.path.exists(model_path):
            print(f'Loading model{opt_name} from {model_path}...')
            model.load_state_dict(torch.load(model_path, map_location=device))
            weights = []
            for param in model.parameters():
                if param.requires_grad:
                    weights.extend(param.data.cpu().numpy().flatten())

            counts, bin_edges = np.histogram(weights, bins=200, density=True)

            fig.add_trace(
                go.Scatter(
                    x=bin_edges,
                    y=np.append(counts, counts[-1]),
                    mode='lines',
                    line_shape='hv', # 阶梯状
                    name=opt_name,
                    line_width=2
                ),
                row=row,col=col
            )
        else:
            print(f'Warning: Model {opt_name} not found.')
            fig.add_annotation(
                text=f'{opt_name} is missing.',
                row=row,
                col=col,
                showarrow=False,
                font=dict(color='red',size=14)
            )
    # --- 更新整体布局 ---
    # 隐藏每个子图的单独图例，只在悬停时显示名称
    fig.update_traces(showlegend=False)

    fig.update_layout(
        title_text='Distributions of Weights in all model',
        title_x=0.5,
        height=350 * rows,  # 根据行数动态调整高度
        width=1200,
        bargap=0.01,
        yaxis_title='Density',
        xaxis_title='Weight Value'
    )
    # 仅显示最底行的x轴标题和最左列的y轴标题
    fig.update_xaxes(title_text="", showticklabels=True, row=rows)
    fig.update_yaxes(title_text="", showticklabels=True, col=1)

    # --- 保存和显示图表 ---
    output_filename = "weight_distribution_subplots.png"
    print(f"正在保存图表至 {output_filename} 并显示交互式图表...")

    # 保存静态图片需要安装 kaleido: pip install kaleido
    try:
        fig.write_image(output_filename,height=350 * rows, width=1200)
    except ValueError as e:
        print(f"\n无法保存静态图片。请先安装 kaleido 库: `pip install kaleido`")
        print(f"错误信息: {e}")

    fig.show()

def plot_diagnostic_run_results(diagnostic_history):
    """
    绘制诊断性运行期间的验证集准确率和平均放电率的交互式图表。
    """
    if not diagnostic_history or not diagnostic_history.get('val_acc'):
        print("No diagnostic history to plot.")
        return

    print("\n--- Generating Diagnostic Plots ---")
    fig = make_subplots(
        rows = 1, cols = 2,
        subplot_titles = ('Validation Accuracy Vs. Epoch', 'Average Spike Rate Vs. Epoch')
    )
    img_width = 1200
    img_height = 500
    epochs = list(range(1, len(diagnostic_history['val_acc']) + 1))

    # 子图1: 验证集准确率
    fig.add_trace(
        go.Scatter(
            x=epochs,
            y=diagnostic_history['val_acc'],
            mode='lines+markers',
            name='Validation Accuracy',
            marker=dict(symbol='circle'),
            line=dict(color='#1f77b4', width=3),
        ),
        row=1, col=1
    )

    # 子图2: 平均放电率
    fig.add_trace(
        go.Scatter(
            x=epochs,
            y=diagnostic_history['avg_spike_rate'],
            mode='lines+markers',
            name='Average Spike Rate',
            marker=dict(symbol='circle',color='red'),
            line=dict(color='red',width=3),
        ),
        row=1, col=2
    )

    fig.update_layout(
        showlegend=False,
        height=img_height,  # 调整高度，避免图像太高
        width=img_width,  # 调整宽度，确保比例协调
        margin=dict(l=80, r=40, t=100, b=80)# 手动设置边距，为Y轴标题提供恰当空间
    )

    # 坐标轴
    fig.update_xaxes(title_text='Epoch', row=1, col=1)
    fig.update_yaxes(title_text='Validation Accuracy', row=1, col=1)

    fig.update_xaxes(title_text='Epoch', row=1, col=2)
    fig.update_yaxes(title_text='Average Spike Rate', row=1, col=2)

    tick_vals = epochs if len(epochs) < 25 else np.arange(1, len(epochs)+1, 5)
    fig.update_xaxes(tickmode='array', tickvals=tick_vals, row=1, col=1)
    fig.update_xaxes(tickmode='array', tickvals=tick_vals, row=1, col=2)

    output_filename = 'diagnostic_run_plot.png'
    print(f'Saving figure to {output_filename}...')

    #保存静态图片
    try:
        fig.write_image(output_filename,width=img_width, height=img_height, scale=1)
    except ValueError as e:
        print(f'\n Can not save the figure. Please install kaleido with pip')
        print(f'\n Error: {e}')
    fig.show()


def plot_gain_tuning_results(results_data):
    if not results_data:
        print("The file is not exist。")
        return

    print("\n--- Generating figures... ---")

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Validation Accuracy with different Gain Values', 'Average Spike Rate with different Gain Values')
    )
    output_filename = "gain_tuning_results.png"
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
    markers = ['circle', 'square', 'diamond', 'cross', 'x', 'triangle-up', 'star', 'hexagon']

    for i, (gain_key, history) in enumerate(results_data.items()):
        color = colors[i % len(colors)]
        marker = markers[i % len(markers)]

        epochs = list(range(1, len(history['val_acc']) + 1))

        # --- 子图1: 验证集准确率 ---
        fig.add_trace(
            go.Scatter(
                x=epochs,
                y=history['val_acc'],
                mode='lines+markers',
                name=f'Gain={gain_key}',
                legendgroup=gain_key,
                marker=dict(symbol=marker, color=color),
                line=dict(color=color)
            ),
            row=1, col=1
        )

        # --- 子图2: 平均放电率 ---
        fig.add_trace(
            go.Scatter(
                x=epochs,
                y=history['avg_spike_rate'],
                mode='lines+markers',
                name=f'Gain={gain_key}',
                legendgroup=gain_key,
                showlegend=False,
                marker=dict(symbol=marker, color=color),
                line=dict(color=color, dash='dash')
            ),
            row=1, col=2
        )

    img_width = 1400
    img_height = 600

    fig.update_layout(
        # title_text='EventGrad Grad_Gain 参数微调对比',
        # title_x=0.5,
        height=img_height,
        width=img_width,
        margin=dict(l=40, r=20, t=40, b=20),
        plot_bgcolor='white',
        # legend_title_text='Grad Gain'
        legend=dict(
            yanchor="top",
            y=0.98,  # 将图例的顶部放在绘图区98%的高度处
            xanchor="left",
            x=0.01,  # 将图例的左边放在绘图区2%的宽度处
            bgcolor='rgba(255, 255, 255, 0.5)',  # 设置半透明背景，避免完全遮挡数据
            bordercolor="Black",
            borderwidth=1
        )
    )
    fig.update_xaxes(
        title_text="Epoch",
        title_font=dict(size=18),
        tickfont=dict(size=14),
        gridcolor='lightgrey',
        showline=True,
        linewidth=1,
        linecolor='black',
        row=1, col=1
    )
    fig.update_yaxes(
        title_text="Validation Accuracy (%)",
        title_font=dict(size=18),
        tickfont=dict(size=14),
        gridcolor='lightgrey',
        showline=True,
        linewidth=1,
        linecolor='black',
        row=1, col=1
    )

    # 更新子图2的坐标轴
    fig.update_xaxes(
        title_text="Epoch",
        title_font=dict(size=18),
        tickfont=dict(size=14),
        gridcolor='lightgrey',
        showline=True,
        linewidth=1,
        linecolor='black',
        row=1, col=2
    )
    fig.update_yaxes(
        title_text="Average Spike Rate (%)",
        title_font=dict(size=18),
        tickfont=dict(size=14),
        gridcolor='lightgrey',
        showline=True,
        linewidth=1,
        linecolor='black',
        row=1, col=2,
        type="log"
    )

    print(f"Saving and Showing {output_filename}...")

    try:
        fig.write_image(output_filename, width=img_width*0.8, height=img_height*0.9)
    except ValueError as e:
        print(f"\n Can not save。Please install kaleido: `pip install kaleido`")
        print(f"Error information: {e}")

    fig.show()


def plot_detailed_tuning_results(results_data):
    """
    为每个gain值绘制详细的对比图，每行包含acc和rate两个子图。
    """
    if not results_data:
        print("Data doesn't exist.")
        return

    print("\n--- Generating Figures... ---")
    gains = list(results_data.keys())
    num_gains = len(gains)

    # 为每个子图动态创建标题
    subplot_titles = [f'{t} (Gain={g})' for g in gains for t in ['Accuracy', 'Average Spike Rate']]

    fig = make_subplots(
        rows=num_gains,
        cols=2,
        subplot_titles=subplot_titles,
        shared_xaxes=True,
        vertical_spacing=0.05
    )

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']  # 为seed预定义颜色

    for i, gain_key in enumerate(gains):
        row = i + 1
        seed_results = results_data[gain_key]

        for j, seed_key in enumerate(sorted(seed_results.keys())):
            color = colors[j % len(colors)]
            history = seed_results[seed_key]
            epochs = list(range(1, len(history['val_acc']) + 1))
            if not epochs: continue
            show_this_legend = (i==0)

            # --- 绘制准确率曲线 ---
            fig.add_trace(
                go.Scatter(
                    x=epochs,
                    y=history['val_acc'],
                    mode='lines+markers',
                    name=f'Seed {seed_key}',
                    legendgroup=f'seed_{seed_key}',
                    showlegend=show_this_legend,
                    line=dict(color=color)
                ),
                row=row, col=1
            )

            # --- 绘制放电率曲线 ---
            fig.add_trace(
                go.Scatter(
                    x=epochs,
                    y=history['avg_spike_rate'],
                    mode='lines',
                    name=f'Seed {seed_key}',
                    legendgroup=f'seed_{seed_key}',
                    showlegend=False,  # 避免图例重复
                    line=dict(color=color, dash='dot')
                ),
                row=row, col=2
            )

    # --- 更新整体布局 ---
    img_height = 350 * num_gains
    img_width = 1400
    title_font_size = 18
    tick_font_size = 14

    fig.update_layout(
        # title_text='Gain 详细调优过程对比',
        # title_x = 0.5,
        height=img_height,
        width=img_width,
        plot_bgcolor='white',
        margin=dict(l=80, r=40, t=100, b=80),
        legend_title_text='Random Seed',
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor='rgba(255, 255, 255, 0.7)',  # 设置半透明背景
            bordercolor="Black",
            borderwidth=1
        )
    )
    fig.update_xaxes(
        title_font = dict(size=title_font_size),
        tickfont = dict(size=tick_font_size),
        gridcolor='lightgrey',
        showline=True,
        linewidth=1,
        linecolor='black'
    )
    fig.update_yaxes(
        title_font=dict(size=title_font_size),
        tickfont=dict(size=tick_font_size),
        gridcolor='lightgrey',
        showline=True,
        linewidth=1,
        linecolor='black'
    )


    # 为所有子图更新坐标轴标题
    for i in range(1, num_gains + 1):
        fig.update_yaxes(title_text="Validation Accuracy (%)", row=i, col=1)
        fig.update_yaxes(title_text="Average Spike Rate", row=i, col=2, type="log")
    # 只为最底部的子图显示X轴标题
    fig.update_xaxes(title_text="Epoch", row=num_gains, col=1)
    fig.update_xaxes(title_text="Epoch", row=num_gains, col=2)

    output_filename = "gain_tuning_detailed_plot.png"
    output_html_filename = "gain_tuning_detailed_plot.html"
    fig.write_image(output_filename, width=img_width, height=img_height)
    fig.write_html(output_html_filename)
    fig.show()

def plot_lr_tuning_results(results_data):
    """
       为每个gain值绘制详细的对比图，每行包含acc和rate两个子图。
       """
    if not results_data:
        print("Data doesn't exist.")
        return

    print("\n--- Generating Figures... ---")
    lrs = list(results_data.keys())
    num_lrs = len(lrs)

    # 为每个子图动态创建标题
    subplot_titles = [f'{t} (lr={g})' for g in lrs for t in ['Accuracy', 'Average Spike Rate']]

    fig = make_subplots(
        rows=num_lrs,
        cols=2,
        subplot_titles=subplot_titles,
        shared_xaxes=True,
        vertical_spacing=0.05
    )

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']  # 为seed预定义颜色

    for i, lr_key in enumerate(lrs):
        row = i + 1
        seed_results = results_data[lr_key]

        for j, seed_key in enumerate(sorted(seed_results.keys())):
            color = colors[j % len(colors)]
            history = seed_results[seed_key]
            epochs = list(range(1, len(history['val_acc']) + 1))
            if not epochs: continue
            show_this_legend = (i == 0)

            # --- 绘制准确率曲线 ---
            fig.add_trace(
                go.Scatter(
                    x=epochs,
                    y=history['val_acc'],
                    mode='lines+markers',
                    name=f'Seed {seed_key}',
                    legendgroup=f'seed_{seed_key}',
                    showlegend=show_this_legend,
                    line=dict(color=color)
                ),
                row=row, col=1
            )

            # --- 绘制放电率曲线 ---
            fig.add_trace(
                go.Scatter(
                    x=epochs,
                    y=history['avg_spike_rate'],
                    mode='lines',
                    name=f'Seed {seed_key}',
                    legendgroup=f'seed_{seed_key}',
                    showlegend=False,  # 避免图例重复
                    line=dict(color=color, dash='dot')
                ),
                row=row, col=2
            )

    # --- 更新整体布局 ---
    img_height = 350 * num_lrs
    img_width = 1400
    title_font_size = 18
    tick_font_size = 14

    fig.update_layout(
        # title_text='Gain 详细调优过程对比',
        # title_x = 0.5,
        height=img_height,
        width=img_width,
        plot_bgcolor='white',
        margin=dict(l=80, r=40, t=100, b=80),
        legend_title_text='Random Seed',
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor='rgba(255, 255, 255, 0.7)',  # 设置半透明背景
            bordercolor="Black",
            borderwidth=1
        )
    )
    fig.update_xaxes(
        title_font=dict(size=title_font_size),
        tickfont=dict(size=tick_font_size),
        gridcolor='lightgrey',
        showline=True,
        linewidth=1,
        linecolor='black'
    )
    fig.update_yaxes(
        title_font=dict(size=title_font_size),
        tickfont=dict(size=tick_font_size),
        gridcolor='lightgrey',
        showline=True,
        linewidth=1,
        linecolor='black'
    )

    # 为所有子图更新坐标轴标题
    for i in range(1, num_lrs + 1):
        fig.update_yaxes(title_text="Validation Accuracy (%)", row=i, col=1)
        fig.update_yaxes(title_text="Average Spike Rate", row=i, col=2, type="log")
    # 只为最底部的子图显示X轴标题
    fig.update_xaxes(title_text="Epoch", row=num_lrs, col=1)
    fig.update_xaxes(title_text="Epoch", row=num_lrs, col=2)

    output_filename = "lr_tuning_detailed_plot.png"
    output_html_filename = "lr_tuning_detailed_plot.html"
    fig.write_image(output_filename, width=img_width, height=img_height)
    fig.write_html(output_html_filename)
    fig.show()

class ConstantLRScheduler:
    """
    一个“什么都不做”的伪调度器。
    它保持学习率恒定，以满足那些需要scheduler对象的函数接口。
    """
    def __init__(self, optimizer):
        self.optimizer = optimizer
        # 保存初始学习率，以便 get_last_lr() 可以返回它
        self.last_lr = [group['lr'] for group in optimizer.param_groups]

    def step(self):
        """
        step()方法什么都不做，从而保持学习率不变。
        """
        pass

    def get_last_lr(self):
        """
        返回在初始化时保存的那个固定的学习率。
        """
        return self.last_lr
# def plot_tuning_summary_violinplot(results_data):
#     """
#     绘制小提琴图，用于直观对比不同gain值下最终准确率的分布情况。
#     """
#     if not results_data:
#         print("The data doesn't exist.")
#         return
#
#     print("\n--- Generating Violin Figure... ---")
#     fig = go.Figure()
#
#     # 遍历每个gain值的结果，为其创建一个小提琴图
#     for gain_key, seed_results in results_data.items():
#         # 提取每个随机种子下最后一个epoch的准确率
#         final_accs = [h.get('val_acc_history', [0])[-1] for h in seed_results.values()]
#
#         fig.add_trace(go.Violin(
#             y=final_accs,
#             name=f'Gain={gain_key}',
#             points='all',  # 显示所有原始数据点
#             box_visible=True,  # 在小提琴内部显示一个盒形图
#             meanline_visible=True  # 在小提琴内部显示平均值线
#         ))
#
#     # --- 更新图表布局 ---
#     img_width = 1000
#     img_height = 700
#     axis_title_font_size = 18
#     axis_tick_font_size = 14
#
#     fig.update_layout(
#         yaxis_title='Validation Accuracy (%)',
#         xaxis_title='Grad Gain (g)',
#         height=img_height,
#         width=img_width,
#         plot_bgcolor='white',
#         margin=dict(l=80, r=40, t=100, b=80),
#         showlegend=False  # 小提琴图通常不需要图例
#     )
#
#     # 更新坐标轴样式
#     fig.update_xaxes(
#         title_font=dict(size=axis_title_font_size),
#         tickfont=dict(size=axis_tick_font_size),
#         gridcolor='lightgrey',
#         showline=True,
#         linewidth=1,
#         linecolor='black'
#     )
#     fig.update_yaxes(
#         title_font=dict(size=axis_title_font_size),
#         tickfont=dict(size=axis_tick_font_size),
#         gridcolor='lightgrey',
#         showline=True,
#         linewidth=1,
#         linecolor='black'
#     )
#
#     # --- 保存和显示图表 ---
#     output_png_filename = "gain_tuning_violinplot.png"
#     output_html_filename = "gain_tuning_violinplot.html"
#
#     fig.write_image(output_png_filename, width=img_width, height=img_height)
#     fig.write_html(output_html_filename)
#
#     print(f"小提琴图静态图片已保存至: {output_png_filename}")
#     print(f"小提琴图可交互网页已保存至: {output_html_filename}")
#
#     fig.show()




