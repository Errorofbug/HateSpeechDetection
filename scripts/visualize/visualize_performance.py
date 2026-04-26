import os
import sys
import numpy as np
import pandas as pd

# 获取项目根目录
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

import matplotlib.pyplot as plt
import matplotlib

from utils import config

# 设置中文字体
matplotlib.rcParams['font.sans-serif'] = ['Arial Unicode MS']
matplotlib.rcParams['axes.unicode_minus'] = False


def plot_performance_comparison(metrics_dict, save_path='documents/figures'):
    """
    绘制模型性能对比柱状图

    参数:
        metrics_dict: 字典，格式为 {模型名: {指标名: 值}}
        save_path: 图片保存路径
    """
    os.makedirs(save_path, exist_ok=True)

    # 转换为DataFrame
    df = pd.DataFrame(metrics_dict).T

    # 键名映射：小写键名转大写显示名
    key_mapping = {
        'accuracy': 'Accuracy',
        'precision': 'Precision',
        'recall': 'Recall',
        'f1': 'F1-Score'
    }

    # 创建显示用的列名（大写）
    display_metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    actual_metrics = ['accuracy', 'precision', 'recall', 'f1']

    # 确保所有指标都存在（使用小写键名检查）
    for metric in actual_metrics:
        if metric not in df.columns:
            df[metric] = [0] * len(df)

    # 创建图表
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')

    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
    x = np.arange(len(df.index))
    width = 0.35

    # 子图1: Accuracy
    axes[0, 0].bar(x, df['accuracy'], color=colors[0], alpha=0.8, width=width)
    axes[0, 0].set_ylabel('Score', fontsize=12)
    axes[0, 0].set_title('Accuracy', fontsize=14, fontweight='bold')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(df.index, rotation=15, ha='right')
    axes[0, 0].set_ylim([0, 1])
    axes[0, 0].grid(axis='y', alpha=0.3)
    # 添加数值标签
    for i, v in enumerate(df['accuracy']):
        axes[0, 0].text(i, v + 0.02, f'{v:.4f}', ha='center', va='bottom', fontsize=10)

    # 子图2: Precision
    axes[0, 1].bar(x, df['precision'], color=colors[1], alpha=0.8, width=width)
    axes[0, 1].set_ylabel('Score', fontsize=12)
    axes[0, 1].set_title('Precision', fontsize=14, fontweight='bold')
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(df.index, rotation=15, ha='right')
    axes[0, 1].set_ylim([0, 1])
    axes[0, 1].grid(axis='y', alpha=0.3)
    for i, v in enumerate(df['precision']):
        axes[0, 1].text(i, v + 0.02, f'{v:.4f}', ha='center', va='bottom', fontsize=10)

    # 子图3: Recall
    axes[1, 0].bar(x, df['recall'], color=colors[2], alpha=0.8, width=width)
    axes[1, 0].set_ylabel('Score', fontsize=12)
    axes[1, 0].set_title('Recall', fontsize=14, fontweight='bold')
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(df.index, rotation=15, ha='right')
    axes[1, 0].set_ylim([0, 1])
    axes[1, 0].grid(axis='y', alpha=0.3)
    for i, v in enumerate(df['recall']):
        axes[1, 0].text(i, v + 0.02, f'{v:.4f}', ha='center', va='bottom', fontsize=10)

    # 子图4: F1-Score
    axes[1, 1].bar(x, df['f1'], color=colors[3], alpha=0.8, width=width)
    axes[1, 1].set_ylabel('Score', fontsize=12)
    axes[1, 1].set_title('F1-Score', fontsize=14, fontweight='bold')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(df.index, rotation=15, ha='right')
    axes[1, 1].set_ylim([0, 1])
    axes[1, 1].grid(axis='y', alpha=0.3)
    for i, v in enumerate(df['f1']):
        axes[1, 1].text(i, v + 0.02, f'{v:.4f}', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    save_file = os.path.join(save_path, 'performance_comparison.png')
    plt.savefig(save_file, dpi=300, bbox_inches='tight')
    print(f"性能对比图已保存至: {save_file}")
    plt.close()


def plot_performance_grouped(metrics_dict, save_path='documents/figures'):
    """
    绘制分组柱状图（所有指标在同一图）

    参数:
        metrics_dict: 字典，格式为 {模型名: {指标名: 值}}
        save_path: 图片保存路径
    """
    os.makedirs(save_path, exist_ok=True)

    df = pd.DataFrame(metrics_dict).T
    models = df.index.tolist()

    # 使用小写键名
    actual_metrics = ['accuracy', 'precision', 'recall', 'f1']
    display_metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']

    x = np.arange(len(models))
    width = 0.2

    fig, ax = plt.subplots(figsize=(12, 6))

    for idx, (metric, color) in enumerate(zip(actual_metrics, colors)):
        offset = (idx - 1.5) * width
        bars = ax.bar(x + offset, df[metric], width, label=display_metrics[idx], color=color, alpha=0.8)

        # 添加数值标签
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=9)

    ax.set_xlabel('Model', fontsize=14)
    ax.set_ylabel('Score', fontsize=14)
    ax.set_title('Model Performance Comparison (All Metrics)', fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha='right')
    ax.set_ylim([0, 1])
    ax.legend(loc='lower right')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    save_file = os.path.join(save_path, 'performance_grouped.png')
    plt.savefig(save_file, dpi=300, bbox_inches='tight')
    print(f"分组性能对比图已保存至: {save_file}")
    plt.close()


def plot_improvement_chart(baseline_metrics, improved_metrics, save_path='documents/figures'):
    """
    绘制性能提升图（展示对比学习的改进效果）

    参数:
        baseline_metrics: 基线模型指标 {指标: 值}
        improved_metrics: 改进模型指标 {指标: 值}
        save_path: 图片保存路径
    """
    os.makedirs(save_path, exist_ok=True)

    # 使用小写键名
    actual_metrics = ['accuracy', 'precision', 'recall', 'f1']
    display_metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']

    baseline_values = [baseline_metrics.get(m, 0) for m in actual_metrics]
    improved_values = [improved_metrics.get(m, 0) for m in actual_metrics]
    improvements = [(improved - baseline) * 100 for improved, baseline in zip(improved_values, baseline_values)]

    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))

    bars1 = ax.bar(x - width/2, baseline_values, width, label='Baseline (BERT Only)',
                  color='#6c757d', alpha=0.8)
    bars2 = ax.bar(x + width/2, improved_values, width, label='Ours (BERT + Contrastive)',
                  color='#28a745', alpha=0.8)

    # 添加数值标签
    for bars, values in [(bars1, baseline_values), (bars2, improved_values)]:
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.4f}', ha='center', va='bottom', fontsize=10)

    # 添加改进百分比标注
    for i, (baseline, improved, improvement) in enumerate(zip(baseline_values, improved_values, improvements)):
        ax.annotate(f'+{improvement:.2f}%',
                  xy=(i + width/2, improved),
                  xytext=(i + width/2, improved + 0.03),
                  ha='center', fontsize=11, fontweight='bold', color='#d9534f',
                  arrowprops=dict(arrowstyle='->', color='#d9534f'))

    ax.set_ylabel('Score', fontsize=14)
    ax.set_title('Performance Improvement with Contrastive Learning', fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylim([0, 1])
    ax.legend(loc='lower right')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    save_file = os.path.join(save_path, 'performance_improvement.png')
    plt.savefig(save_file, dpi=300, bbox_inches='tight')
    print(f"性能提升图已保存至: {save_file}")
    plt.close()


if __name__ == '__main__':
    # 示例：对比不同模型的性能
    metrics = {
        'BERT + SupCon': {'Accuracy': 0.9234, 'Precision': 0.9123, 'Recall': 0.9045, 'F1-Score': 0.9084},
        'BERT + InfoNCE': {'Accuracy': 0.9112, 'Precision': 0.8987, 'Recall': 0.8934, 'F1-Score': 0.8960},
        'BERT Only': {'Accuracy': 0.8956, 'Precision': 0.8734, 'Recall': 0.8789, 'F1-Score': 0.8761}
    }

    plot_performance_comparison(metrics)
    plot_performance_grouped(metrics)

    # 示例：性能提升图
    baseline = {'Accuracy': 0.8956, 'Precision': 0.8734, 'Recall': 0.8789, 'F1-Score': 0.8761}
    improved = {'Accuracy': 0.9234, 'Precision': 0.9123, 'Recall': 0.9045, 'F1-Score': 0.9084}
    plot_improvement_chart(baseline, improved)