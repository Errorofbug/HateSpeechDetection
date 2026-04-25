"""
训练脚本 - 基于对比学习的不当言论检测模型训练

使用方法:
    python scripts/train.py                          # 使用默认配置训练（SupCon + mini数据集）
    python scripts/train.py --full                   # 使用完整数据集训练
    python scripts/train.py --no-contrastive         # 训练基线模型（无对比学习）
    python scripts/train.py --contrastive-type infonce  # 使用InfoNCE损失
"""
import os
import sys
from datetime import datetime
import random
import numpy as np

# 获取项目根目录
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from torch.optim import AdamW

from scripts.dataset import HateSpeechDataset
from models.model import ContrastiveHateSpeechModel
from utils import train_logger, config


def set_seed(seed=42):
    """
    设置随机种子以确保实验可重复性
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def info_nce_loss(features_original, features_augmented, temperature=0.05):
    """
    计算无监督对比损失 (InfoNCE)
    """
    features_original = F.normalize(features_original, p=2, dim=1)
    features_augmented = F.normalize(features_augmented, p=2, dim=1)
    similarity_matrix = torch.matmul(features_original, features_augmented.T) / temperature
    batch_size = features_original.size(0)
    labels = torch.arange(batch_size).to(features_original.device)
    loss = F.cross_entropy(similarity_matrix, labels)
    return loss


def supcon_loss(features_original, features_augmented, labels, temperature=0.05):
    """
    计算有监督对比损失 (SupCon)
    """
    device = features_original.device
    batch_size = features_original.shape[0]

    features_original = F.normalize(features_original, p=2, dim=1)
    features_augmented = F.normalize(features_augmented, p=2, dim=1)

    features = torch.cat([features_original, features_augmented], dim=0)
    labels = torch.cat([labels, labels], dim=0)

    labels = labels.contiguous().view(-1, 1)
    mask = torch.eq(labels, labels.T).float().to(device)

    anchor_dot_contrast = torch.div(torch.matmul(features, features.T), temperature)
    logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
    logits = anchor_dot_contrast - logits_max.detach()

    logits_mask = torch.scatter(
        torch.ones_like(mask), 1,
        torch.arange(batch_size * 2).view(-1, 1).to(device), 0
    )
    mask = mask * logits_mask

    exp_logits = torch.exp(logits) * logits_mask
    log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

    mask_pos_pairs = mask.sum(1)
    mask_pos_pairs = torch.where(mask_pos_pairs < 1e-6, torch.ones_like(mask_pos_pairs), mask_pos_pairs)
    mean_log_prob_pos = (mask * log_prob).sum(1) / mask_pos_pairs

    loss = -mean_log_prob_pos.mean()
    return loss


def train(use_contrastive=None, contrastive_type=None, use_mini=None):
    """
    训练函数

    参数:
        use_contrastive: 是否使用对比学习（None则从配置读取）
        contrastive_type: 对比学习类型（None则从配置读取）
        use_mini: 是否使用mini数据集（None则从配置读取）
    """
    # 从配置获取参数
    batch_size = int(config.get('training', 'batch_size', default=16))
    epochs = int(config.get('training', 'epochs', default=3))
    learning_rate = float(config.get('training', 'learning_rate', default=2e-5))
    lambda_weight = float(config.get('training', 'lambda_weight', default=0.1))
    temperature = float(config.get('training', 'temperature', default=0.05))
    log_per_step = float(config.get('training', 'log_per_step', default=10))

    # 如果参数未指定，从配置读取
    if use_contrastive is None:
        use_contrastive = config.get('training', 'use_contrastive', default=True)
    if contrastive_type is None:
        contrastive_type = config.get('training', 'contrastive_type', default='supcon')
    if use_mini is None:
        use_mini = config.get('training', 'use_mini_dataset', default=True)

    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 初始化训练日志器
    log_dir = config.get_path('logs_dir')

    # 生成带时间戳的日志文件
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'training_{timestamp}.log')
    train_logger.set_log_file(log_file)

    # 记录配置
    train_logger.info("=" * 80)
    train_logger.info("开始训练")
    train_logger.info("=" * 80)
    train_logger.log_config({
        'Device': device,
        'Batch Size': batch_size,
        'Epochs': epochs if not use_mini else 1,
        'Learning Rate': learning_rate,
        'Lambda Weight': lambda_weight,
        'Temperature': temperature,
        'Use Contrastive': use_contrastive,
        'Contrastive Type': contrastive_type if use_contrastive else 'N/A',
        'Use Mini Dataset': use_mini
    })

    # 加载数据
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    data_processed_dir = config.get_path('data_processed_dir')

    if use_mini:
        epochs = 1  # mini数据集只用1个epoch
        data_path = os.path.join(data_processed_dir, 'mini_train.csv')
        train_logger.info("使用mini数据集进行快速测试...")
    else:
        data_path = os.path.join(data_processed_dir, 'train.csv')
        train_logger.info("使用完整训练集进行训练...")

    train_dataset = HateSpeechDataset(data_path, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    train_logger.info(f"训练集大小: {len(train_dataset)}")
    train_logger.info(f"Batch数: {len(train_loader)}")

    # 初始化模型
    model_name = config.get('model', 'model_name', default='bert-base-chinese')
    model = ContrastiveHateSpeechModel(model_name).to(device)
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    criterion_ce = nn.CrossEntropyLoss()

    train_logger.info(f"模型: {model_name}")
    train_logger.info("=" * 80)

    # 设置随机种子以确保实验可重复性
    set_seed(42)

    # 训练循环
    model.train()
    global_step = 0

    for epoch in range(epochs):
        total_train_loss = 0
        total_ce_loss = 0
        total_con_loss = 0

        train_logger.info("")
        train_logger.info(f"Epoch {epoch + 1}/{epochs}")
        train_logger.info("-" * 80)

        for step, batch in enumerate(train_loader):
            optimizer.zero_grad()

            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            input_ids_aug = batch['input_ids_aug'].to(device)
            attention_mask_aug = batch['attention_mask_aug'].to(device)
            labels = batch['labels'].to(device)

            proj_orig, logits = model(input_ids, attention_mask)
            proj_aug, _ = model(input_ids_aug, attention_mask_aug)

            loss_ce = criterion_ce(logits, labels)

            if use_contrastive:
                if contrastive_type == 'supcon':
                    loss_con = supcon_loss(proj_orig, proj_aug, labels, temperature)
                else:
                    loss_con = info_nce_loss(proj_orig, proj_aug, temperature)
                loss = loss_ce + lambda_weight * loss_con
            else:
                loss_con = torch.tensor(0.0)
                loss = loss_ce

            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            total_ce_loss += loss_ce.item()
            if use_contrastive:
                total_con_loss += loss_con.item()

            global_step += 1

            # 每log_per_step步记录一次
            if step % log_per_step == 0:
                # 记录到可读日志
                train_logger.log_metrics(
                    epoch=epoch + 1,
                    step=step,
                    total_loss=loss.item(),
                    ce_loss=loss_ce.item(),
                    con_loss=loss_con.item() if use_contrastive else None
                )
                # 记录到CSV格式的loss日志（用于绘图）
                train_logger.log_metrics_for_plot(
                    epoch=epoch + 1,
                    step=step,
                    total_loss=loss.item(),
                    ce_loss=loss_ce.item(),
                    con_loss=loss_con.item() if use_contrastive else None
                )

        # Epoch结束统计
        avg_loss = total_train_loss / len(train_loader)
        avg_ce_loss = total_ce_loss / len(train_loader)
        avg_con_loss = total_con_loss / len(train_loader) if use_contrastive else 0

        train_logger.info("-" * 80)
        train_logger.info(f"Epoch {epoch + 1} 完成:")
        train_logger.info(f"  平均总损失: {avg_loss:.6f}")
        train_logger.info(f"  平均分类损失: {avg_ce_loss:.6f}")
        if use_contrastive:
            train_logger.info(f"  平均对比损失: {avg_con_loss:.6f}")

    # 保存模型
    checkpoints_dir = config.get_path('checkpoints_dir')
    os.makedirs(checkpoints_dir, exist_ok=True)

    model_suffix = f'{contrastive_type}' if use_contrastive else 'baseline'
    model_filename = f'model_{model_suffix}.pth'
    save_path = os.path.join(checkpoints_dir, model_filename)
    torch.save(model.state_dict(), save_path)

    train_logger.info("=" * 80)
    train_logger.info("训练完成!")
    train_logger.info(f"模型已保存至: {save_path}")
    train_logger.info(f"日志已保存至: {log_file}")
    train_logger.info(f"Loss日志已保存至: {train_logger.current_loss_file}")
    train_logger.info("=" * 80)

    return save_path, log_file, train_logger.current_loss_file


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='训练对比学习不当言论检测模型')
    parser.add_argument('--no-contrastive', action='store_true',
                       help='不使用对比学习（基线模型）')
    parser.add_argument('--contrastive-type', type=str, default=None,
                       choices=['supcon', 'infonce'],
                       help='对比学习类型: supcon 或 infonce')
    parser.add_argument('--full', action='store_true',
                       help='使用完整数据集训练（默认使用mini数据集快速测试）')

    args = parser.parse_args()

    train(
        use_contrastive=not args.no_contrastive if args.no_contrastive else None,
        contrastive_type=args.contrastive_type,
        use_mini=False if args.full else None
    )