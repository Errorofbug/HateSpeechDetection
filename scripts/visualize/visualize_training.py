"""
训练曲线可视化脚本

使用方法:
    python -m scripts.visualize_training                    # 使用默认配置
    python -m scripts.visualize_training --loss-log-file path    # 指定loss日志文件
"""
import os
import sys
import argparse

# 获取项目根目录
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

import matplotlib.pyplot as plt
import matplotlib
import numpy as np

from utils.config import get_config

# 设置中文字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False


def plot_training_curves(log_file=None, loss_log_file=None, save_path='docs/figures'):
    """
    绘制训练损失曲线

    参数:
        log_file: 训练日志文件路径（可选，向后兼容）
        loss_log_file: loss日志文件路径（CSV格式，优先使用）
        save_path: 图片保存路径
    """
    # 优先使用loss日志文件
    if loss_log_file is not None and os.path.exists(loss_log_file):
        return plot_training_curves_from_csv(loss_log_file, save_path)
    # 回退到log_file（向后兼容）
    elif log_file is not None and os.path.exists(log_file):
        return plot_training_curves_from_log(log_file, save_path)
    else:
        print(f"警告: 找不到日志文件 {log_file} 或 {loss_log_file}")
        print("请先运行训练脚本生成日志")
        return


def plot_training_curves_from_csv(csv_file, save_path='docs/figures'):
    """
    从CSV格式的loss日志绘制训练曲线

    参数:
        csv_file: CSV格式的loss日志文件路径
        save_path: 图片保存路径
    """
    import pandas as pd

    # 读取CSV格式的loss日志
    # 格式: epoch,step,total_loss,ce_loss,con_loss
    df = pd.read_csv(csv_file)

    steps = df['step'].tolist()
    total_losses = df['total_loss'].tolist()
    ce_losses = df['ce_loss'].tolist()
    con_losses = df['con_loss'].tolist()

    if not steps:
        print("未能从日志中解析出训练数据")
        return

    # 创建图片保存目录
    os.makedirs(save_path, exist_ok=True)

    # 设置图表样式
    try:
        plt.style.use('seaborn-v0_8-whitegrid')
    except:
        plt.style.use('seaborn-whitegrid')

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Training Curves', fontsize=16, fontweight='bold')

    # 图1: 总损失
    axes[0, 0].plot(steps, total_losses, linewidth=2, color='#2E86AB', label='Total Loss')
    axes[0, 0].set_xlabel('Step', fontsize=12)
    axes[0, 0].set_ylabel('Loss', fontsize=12)
    axes[0, 0].set_title('Total Training Loss', fontsize=14, fontweight='bold')
    axes[0, 0].legend(loc='upper right')
    axes[0, 0].grid(True, alpha=0.3)

    # 图2: 分类损失
    axes[0, 1].plot(steps, ce_losses, linewidth=2, color='#A23B72', label='Cross-Entropy Loss')
    axes[0, 1].set_xlabel('Step', fontsize=12)
    axes[0, 1].set_ylabel('Loss', fontsize=12)
    axes[0, 1].set_title('Classification Loss', fontsize=14, fontweight='bold')
    axes[0, 1].legend(loc='upper right')
    axes[0, 1].grid(True, alpha=0.3)

    # 图3: 对比损失
    axes[1, 0].plot(steps, con_losses, linewidth=2, color='#F18F01', label='Contrastive Loss')
    axes[1, 0].set_xlabel('Step', fontsize=12)
    axes[1, 0].set_ylabel('Loss', fontsize=12)
    axes[1, 0].set_title('Contrastive Loss', fontsize=14, fontweight='bold')
    axes[1, 0].legend(loc='upper right')
    axes[1, 0].grid(True, alpha=0.3)

    # 图4: 所有损失对比
    axes[1, 1].plot(steps, total_losses, linewidth=2, color='#2E86AB', label='Total')
    axes[1, 1].plot(steps, ce_losses, linewidth=2, color='#A23B72', label='CE', alpha=0.7)
    axes[1, 1].plot(steps, con_losses, linewidth=2, color='#F18F01', label='Contrastive', alpha=0.7)
    axes[1, 1].set_xlabel('Step', fontsize=12)
    axes[1, 1].set_ylabel('Loss', fontsize=12)
    axes[1, 1].set_title('All Losses Comparison', fontsize=14, fontweight='bold')
    axes[1, 1].legend(loc='upper right')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    save_file = os.path.join(save_path, 'training_curves.png')
    plt.savefig(save_file, dpi=300, bbox_inches='tight')
    print(f"训练曲线图已保存至: {save_file}")
    plt.close()


def plot_training_curves_from_log(log_file, save_path='docs/figures'):
    """
    从人类可读的日志文件绘制训练曲线（向后兼容）

    参数:
        log_file: 日志文件路径
        save_path: 图片保存路径
    """
    # 解析日志文件
    # 旧格式: step,total_loss,ce_loss,con_loss
    # 例如: 1,1.112161,0.752760,3.594004
    steps = []
    total_losses = []
    ce_losses = []
    con_losses = []

    with open(log_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#') or line.startswith('-'):
                continue
            try:
                parts = line.split(',')
                if len(parts) >= 4:
                    steps.append(int(parts[1]))  # step在第二个位置
                    total_losses.append(float(parts[2]))
                    ce_losses.append(float(parts[3]))
                    if len(parts) >= 5:
                        con_losses.append(float(parts[4]))
            except (ValueError, IndexError):
                continue

    if not steps:
        print("未能从日志中解析出训练数据")
        return

    # 创建图片保存目录
    os.makedirs(save_path, exist_ok=True)

    # 设置图表样式
    try:
        plt.style.use('seaborn-v0_8-whitegrid')
    except:
        plt.style.use('seaborn-whitegrid')

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Training Curves', fontsize=16, fontweight='bold')

    # 图1: 总损失
    axes[0, 0].plot(steps, total_losses, linewidth=2, color='#2E86AB', label='Total Loss')
    axes[0, 0].set_xlabel('Step', fontsize=12)
    axes[0, 0].set_ylabel('Loss', fontsize=12)
    axes[0, 0].set_title('Total Training Loss', fontsize=14, fontweight='bold')
    axes[0, 0].legend(loc='upper right')
    axes[0, 0].grid(True, alpha=0.3)

    # 图2: 分类损失
    axes[0, 1].plot(steps, ce_losses, linewidth=2, color='#A23B72', label='Cross-Entropy Loss')
    axes[0, 1].set_xlabel('Step', fontsize=12)
    axes[0, 1].set_ylabel('Loss', fontsize=12)
    axes[0, 1].set_title('Classification Loss', fontsize=14, fontweight='bold')
    axes[0, 1].legend(loc='upper right')
    axes[0, 1].grid(True, alpha=0.3)

    # 图3: 对比损失
    axes[1, 0].plot(steps, con_losses, linewidth=2, color='#F18F01', label='Contrastive Loss')
    axes[1, 0].set_xlabel('Step', fontsize=12)
    axes[1, 0].set_ylabel('Loss', fontsize=12)
    axes[1, 0].set_title('Contrastive Loss', fontsize=14, fontweight='bold')
    axes[1, 0].legend(loc='upper right')
    axes[1, 0].grid(True, alpha=0.3)

    # 图4: 所有损失对比
    axes[1, 1].plot(steps, total_losses, linewidth=2, color='#2E86AB', label='Total')
    axes[1, 1].plot(steps, ce_losses, linewidth=2, color='#A23B72', label='CE', alpha=0.7)
    axes[1, 1].plot(steps, con_losses, linewidth=2, color='#F18F01', label='Contrastive', alpha=0.7)
    axes[1, 1].set_xlabel('Step', fontsize=12)
    axes[1, 1].set_ylabel('Loss', fontsize=12)
    axes[1, 1].set_title('All Losses Comparison', fontsize=14, fontweight='bold')
    axes[1, 1].legend(loc='upper right')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    save_file = os.path.join(save_path, 'training_curves.png')
    plt.savefig(save_file, dpi=300, bbox_inches='tight')
    print(f"训练曲线图已保存至: {save_file}")
    plt.close()


def plot_loss_comparison(loss_files, labels, save_path='docs/figures'):
    """
    对比不同实验的训练损失曲线

    参数:
        loss_files: loss日志文件路径列表（CSV格式）
        labels: 每个实验的标签
        save_path: 图片保存路径
    """
    os.makedirs(save_path, exist_ok=True)

    plt.figure(figsize=(12, 6))
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#3B1F2B']

    for idx, (loss_file, label) in enumerate(zip(loss_files, labels)):
        if not os.path.exists(loss_file):
            continue

        import pandas as pd
        df = pd.read_csv(loss_file)
        steps, losses = df['step'].tolist(), df['total_loss'].tolist()

        if steps:
            plt.plot(steps, losses, linewidth=2, color=colors[idx % len(colors)],
                    label=label, alpha=0.8)

    plt.xlabel('Training Step', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.title('Training Loss Comparison', fontsize=16, fontweight='bold')
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)

    save_file = os.path.join(save_path, 'loss_comparison.png')
    plt.savefig(save_file, dpi=300, bbox_inches='tight')
    print(f"损失对比图已保存至: {save_file}")
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='绘制训练曲线')
    parser.add_argument('--loss-log-file', type=str, default=None,
                       help='loss日志文件路径（CSV格式）')
    parser.add_argument('--save-path', type=str, default=None,
                       help='图表保存路径')

    args = parser.parse_args()

    # 加载配置
    config = get_config()
    save_path = args.save_path or config.get('visualization', 'figures_dir', default='docs/figures')

    # 绘制训练曲线
    plot_training_curves(
        loss_log_file=args.loss_log_file,
        save_path=save_path
    )