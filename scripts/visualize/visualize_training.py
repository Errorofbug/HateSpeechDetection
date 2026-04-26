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

from utils import config

# 设置中文字体
matplotlib.rcParams['font.sans-serif'] = ['Arial Unicode MS']
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
    else:
        print(f"警告: 找不到日志文件 {log_file} 或 {loss_log_file}")
        print("请先运行训练脚本生成日志")
        return


def plot_training_curves_from_csv(csv_file, save_path='docs/figures/'):
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

    if df.empty:
        print("CSV文件为空")
        return

    # 处理多epoch的情况：将step转换为全局step
    # 首先获取每个epoch的步数
    epoch_info = df.groupby('epoch')['step'].max().to_dict()

    # 计算全局步数
    global_steps = []
    for _, row in df.iterrows():
        epoch = int(row['epoch'])
        step = int(row['step'])
        # 全局step = 之前所有epoch的总步数 + 当前epoch的step
        steps_before = sum(epoch_info.get(e, 0) for e in range(1, epoch))
        global_step = steps_before + step
        global_steps.append(global_step)

    total_losses = df['total_loss'].tolist()
    ce_losses = df['ce_loss'].tolist()
    con_losses = df['con_loss'].tolist()

    if not global_steps:
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

    # 添加epoch边界线
    total_steps = max(global_steps)
    num_epochs = len(epoch_info)
    epoch_boundaries = []
    steps_before = 0
    for i in range(1, num_epochs + 1):
        if i in epoch_info:
            steps_before += epoch_info[i]
            if i < num_epochs:  # 不在最后一个epoch后画线
                epoch_boundaries.append(steps_before)

    # 图1: 总损失
    axes[0, 0].plot(global_steps, total_losses, linewidth=2, color='#2E86AB', label='Total Loss')
    for boundary in epoch_boundaries:
        axes[0, 0].axvline(x=boundary, color='red', linestyle='--', linewidth=1, alpha=0.5)
    axes[0, 0].set_xlabel('Step', fontsize=12)
    axes[0, 0].set_ylabel('Loss', fontsize=12)
    axes[0, 0].set_title('Total Training Loss', fontsize=14, fontweight='bold')
    axes[0, 0].legend(loc='upper right')
    axes[0, 0].grid(True, alpha=0.3)

    # 图2: 分类损失
    axes[0, 1].plot(global_steps, ce_losses, linewidth=2, color='#A23B72', label='Cross-Entropy Loss')
    for boundary in epoch_boundaries:
        axes[0, 1].axvline(x=boundary, color='red', linestyle='--', linewidth=1, alpha=0.5)
    axes[0, 1].set_xlabel('Step', fontsize=12)
    axes[0, 1].set_ylabel('Loss', fontsize=12)
    axes[0, 1].set_title('Classification Loss', fontsize=14, fontweight='bold')
    axes[0, 1].legend(loc='upper right')
    axes[0, 1].grid(True, alpha=0.3)

    # 图3: 对比损失
    axes[1, 0].plot(global_steps, con_losses, linewidth=2, color='#F18F01', label='Contrastive Loss')
    for boundary in epoch_boundaries:
        axes[1, 0].axvline(x=boundary, color='red', linestyle='--', linewidth=1, alpha=0.5)
    axes[1, 0].set_xlabel('Step', fontsize=12)
    axes[1, 0].set_ylabel('Loss', fontsize=12)
    axes[1, 0].set_title('Contrastive Loss', fontsize=14, fontweight='bold')
    axes[1, 0].legend(loc='upper right')
    axes[1, 0].grid(True, alpha=0.3)

    # 图4: 所有损失对比
    axes[1, 1].plot(global_steps, total_losses, linewidth=2, color='#2E86AB', label='Total')
    axes[1, 1].plot(global_steps, ce_losses, linewidth=2, color='#A23B72', label='CE', alpha=0.7)
    axes[1, 1].plot(global_steps, con_losses, linewidth=2, color='#F18F01', label='Contrastive', alpha=0.7)
    for boundary in epoch_boundaries:
        axes[1, 1].axvline(x=boundary, color='red', linestyle='--', linewidth=1, alpha=0.5)
    axes[1, 1].set_xlabel('Step', fontsize=12)
    axes[1, 1].set_ylabel('Loss', fontsize=12)
    axes[1, 1].set_title('All Losses Comparison', fontsize=14, fontweight='bold')
    axes[1, 1].legend(loc='upper right')
    axes[1, 1].grid(True, alpha=0.3)

    # 添加epoch标签
    if len(epoch_boundaries) > 0:
        for ax in axes.flat:
            half_epoch = epoch_boundaries[0] / 2
            ax.text(half_epoch, ax.get_ylim()[1], 'Epoch 1',
                    ha='center', va='top', fontsize=10, color='gray', alpha=0.7)
            for i, boundary in enumerate(epoch_boundaries[:-1], 2):
                mid = (boundary + epoch_boundaries[i-1]) / 2 if i > 1 else (boundary + epoch_boundaries[0]) / 2
                ax.text(mid, ax.get_ylim()[1], f'Epoch {i}',
                        ha='center', va='top', fontsize=10, color='gray', alpha=0.7)

    plt.tight_layout()
    save_file = os.path.join(save_path, 'training_curves.png')
    plt.savefig(save_file, dpi=300, bbox_inches='tight')
    print(f"训练曲线图已保存至: {save_file}")
    plt.close()


def plot_loss_comparison(loss_files, labels, save_path='docs/figures/'):
    """
    对比不同实验的训练损失曲线

    参数:
        loss_files: loss日志文件路径列表（CSV格式）
        labels: 每个实验的标签
        save_path: 图片保存路径
    """
    import pandas as pd

    os.makedirs(save_path, exist_ok=True)

    plt.figure(figsize=(12, 6))
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#3B1F2B']

    plotted = False
    for idx, (loss_file, label) in enumerate(zip(loss_files, labels)):
        print(f"  处理: {label} -> {loss_file}")

        if not os.path.exists(loss_file):
            print(f"  警告: 文件不存在 {loss_file}")
            continue

        df = pd.read_csv(loss_file)
        if df.empty:
            print(f"  警告: CSV文件为空 {loss_file}")
            continue

        print(f"  读取到 {len(df)} 条记录")

        # 处理多epoch的情况：将step转换为全局step
        epoch_info = df.groupby('epoch')['step'].max().to_dict()
        print(f"  Epoch信息: {epoch_info}")

        global_steps = []
        for _, row in df.iterrows():
            epoch = int(row['epoch'])
            step = int(row['step'])
            steps_before = sum(epoch_info.get(e, 0) for e in range(1, epoch))
            global_steps.append(steps_before + step)

        losses = df['total_loss'].tolist()

        if global_steps:
            plt.plot(global_steps, losses, linewidth=2, color=colors[idx % len(colors)],
                    label=label, alpha=0.8)
            plotted = True
            print(f"  绘制 {len(global_steps)} 个数据点")

    if not plotted:
        print("  错误: 没有成功绘制任何数据")
        plt.close()
        return

    plt.xlabel('Training Step', fontsize=14)
    plt.ylabel('Total Loss', fontsize=14)
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

    save_path = args.save_path or config.get('visualization', 'figures_dir', default='docs/figures')

    # 绘制训练曲线
    plot_training_curves(
        loss_log_file=args.loss_log_file,
        save_path=save_path
    )