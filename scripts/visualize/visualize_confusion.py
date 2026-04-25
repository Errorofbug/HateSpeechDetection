import os
import sys
import numpy as np
from sklearn.metrics import confusion_matrix

# 获取项目根目录
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns

from utils import config

# 设置中文字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False


def plot_confusion_matrix(true_labels, pred_labels, classes=['正常言论', '不当言论'],
                        save_path='documents/figures'):
    """
    绘制混淆矩阵热力图

    参数:
        true_labels: 真实标签列表
        pred_labels: 预测标签列表
        classes: 类别名称列表
        save_path: 图片保存路径
    """
    os.makedirs(save_path, exist_ok=True)

    # 计算混淆矩阵
    cm = confusion_matrix(true_labels, pred_labels)

    # 计算百分比
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

    # 创建子图
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Confusion Matrix', fontsize=16, fontweight='bold')

    # 左图: 数量
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                xticklabels=classes, yticklabels=classes, cbar_kws={'label': 'Count'})
    axes[0].set_xlabel('Predicted Label', fontsize=12)
    axes[0].set_ylabel('True Label', fontsize=12)
    axes[0].set_title('Count', fontsize=14, fontweight='bold')

    # 右图: 百分比
    sns.heatmap(cm_normalized, annot=True, fmt='.1f', cmap='Reds', ax=axes[1],
                xticklabels=classes, yticklabels=classes, cbar_kws={'label': 'Percentage (%)'})
    axes[1].set_xlabel('Predicted Label', fontsize=12)
    axes[1].set_ylabel('True Label', fontsize=12)
    axes[1].set_title('Percentage', fontsize=14, fontweight='bold')

    plt.tight_layout()
    save_file = os.path.join(save_path, 'confusion_matrix.png')
    plt.savefig(save_file, dpi=300, bbox_inches='tight')
    print(f"混淆矩阵已保存至: {save_file}")
    plt.close()

    # 打印详细统计
    print("\n" + "="*50)
    print("混淆矩阵统计:")
    print("="*50)
    print(f"真阳性 (TP): {cm[1, 1]:4d} - 不当言论被正确识别")
    print(f"假阳性 (FP): {cm[0, 1]:4d} - 正常言论被误判为不当言论")
    print(f"假阴性 (FN): {cm[1, 0]:4d} - 不当言论被误判为正常言论")
    print(f"真阴性 (TN): {cm[0, 0]:4d} - 正常言论被正确识别")
    print("="*50)

    # 计算指标
    tp, fp, fn, tn = cm[1, 1], cm[0, 1], cm[1, 0], cm[0, 0]
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    print(f"准确率 (Accuracy): {accuracy:.4f}")
    print(f"精确率 (Precision): {precision:.4f}")
    print(f"召回率 (Recall): {recall:.4f}")
    print(f"F1分数 (F1-Score): {f1:.4f}")
    print("="*50 + "\n")


def plot_multiple_confusion_matrices(results_dict, save_path='documents/figures'):
    """
    绘制多个模型的混淆矩阵对比

    参数:
        results_dict: 字典，格式为 {模型名: (true_labels, pred_labels)}
        save_path: 图片保存路径
    """
    os.makedirs(save_path, exist_ok=True)

    n_models = len(results_dict)
    fig, axes = plt.subplots(2, n_models, figsize=(6*n_models, 10))
    if n_models == 1:
        axes = axes.reshape(2, 1)

    fig.suptitle('Confusion Matrix Comparison', fontsize=16, fontweight='bold')

    for idx, (model_name, (true_labels, pred_labels)) in enumerate(results_dict.items()):
        cm = confusion_matrix(true_labels, pred_labels)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

        # 数量图
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, idx],
                    xticklabels=['正常', '不当'], yticklabels=['正常', '不当'],
                    cbar_kws={'label': 'Count'} if idx == n_models-1 else {})
        axes[0, idx].set_xlabel('Predicted', fontsize=11)
        axes[0, idx].set_ylabel('True', fontsize=11)
        axes[0, idx].set_title(f'{model_name}\n(Count)', fontsize=12, fontweight='bold')

        # 百分比图
        sns.heatmap(cm_normalized, annot=True, fmt='.1f', cmap='Reds', ax=axes[1, idx],
                    xticklabels=['正常', '不当'], yticklabels=['正常', '不当'],
                    cbar_kws={'label': 'Percentage (%)'} if idx == n_models-1 else {})
        axes[1, idx].set_xlabel('Predicted', fontsize=11)
        axes[1, idx].set_ylabel('True', fontsize=11)
        axes[1, idx].set_title(f'(Percentage)', fontsize=12, fontweight='bold')

    plt.tight_layout()
    save_file = os.path.join(save_path, 'confusion_matrix_comparison.png')
    plt.savefig(save_file, dpi=300, bbox_inches='tight')
    print(f"混淆矩阵对比图已保存至: {save_file}")
    plt.close()


if __name__ == '__main__':
    # 示例：绘制单个混淆矩阵
    # 需要先运行评估脚本获取真实标签和预测标签

    # 示例数据
    true_labels = [0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0]
    pred_labels = [0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0]

    plot_confusion_matrix(true_labels, pred_labels)

    # 示例：绘制多个模型的混淆矩阵对比
    # results = {
    #     'BERT + SupCon': (true_labels1, pred_labels1),
    #     'BERT + InfoNCE': (true_labels2, pred_labels2),
    #     'BERT Only': (true_labels3, pred_labels3)
    # }
    # plot_multiple_confusion_matrices(results)