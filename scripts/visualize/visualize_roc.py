import os
import sys
import numpy as np
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

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


def plot_roc_curve(true_labels, prob_scores, model_name='BERT + Contrastive',
                  save_path='documents/figures'):
    """
    绘制ROC曲线

    参数:
        true_labels: 真实标签列表 (0/1)
        prob_scores: 预测为正类的概率列表
        model_name: 模型名称
        save_path: 图片保存路径
    """
    os.makedirs(save_path, exist_ok=True)

    # 计算ROC曲线
    fpr, tpr, thresholds = roc_curve(true_labels, prob_scores)
    roc_auc = auc(fpr, tpr)

    # 绘制ROC曲线
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='#2E86AB', lw=3, label=f'{model_name} (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random (AUC = 0.5)')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=14)
    plt.ylabel('True Positive Rate (Sensitivity)', fontsize=14)
    plt.title('ROC Curve', fontsize=16, fontweight='bold')
    plt.legend(loc="lower right", fontsize=12)
    plt.grid(True, alpha=0.3)

    save_file = os.path.join(save_path, 'roc_curve.png')
    plt.savefig(save_file, dpi=300, bbox_inches='tight')
    print(f"ROC曲线已保存至: {save_file}")
    plt.close()

    # 打印AUC值
    print(f"\n{'='*50}")
    print(f"ROC分析结果 - {model_name}")
    print(f"{'='*50}")
    print(f"AUC值: {roc_auc:.4f}")
    print(f"{'='*50}\n")


def plot_multiple_roc(results_dict, save_path='documents/figures'):
    """
    绘制多个模型的ROC曲线对比

    参数:
        results_dict: 字典，格式为 {模型名: (true_labels, prob_scores)}
        save_path: 图片保存路径
    """
    os.makedirs(save_path, exist_ok=True)

    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#3B1F2B']

    plt.figure(figsize=(10, 8))

    for idx, (model_name, (true_labels, prob_scores)) in enumerate(results_dict.items()):
        fpr, tpr, _ = roc_curve(true_labels, prob_scores)
        roc_auc = auc(fpr, tpr)
        color = colors[idx % len(colors)]
        plt.plot(fpr, tpr, color=color, lw=2,
                label=f'{model_name} (AUC = {roc_auc:.4f})')

    plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--', label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=14)
    plt.ylabel('True Positive Rate', fontsize=14)
    plt.title('ROC Curve Comparison', fontsize=16, fontweight='bold')
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(True, alpha=0.3)

    save_file = os.path.join(save_path, 'roc_comparison.png')
    plt.savefig(save_file, dpi=300, bbox_inches='tight')
    print(f"ROC对比图已保存至: {save_file}")
    plt.close()


def plot_pr_curve(true_labels, prob_scores, model_name='BERT + Contrastive',
                 save_path='documents/figures'):
    """
    绘制精确率-召回率曲线

    参数:
        true_labels: 真实标签列表 (0/1)
        prob_scores: 预测为正类的概率列表
        model_name: 模型名称
        save_path: 图片保存路径
    """
    os.makedirs(save_path, exist_ok=True)

    # 计算PR曲线
    precision, recall, _ = precision_recall_curve(true_labels, prob_scores)
    ap_score = average_precision_score(true_labels, prob_scores)

    # 绘制PR曲线
    plt.figure(figsize=(10, 8))
    plt.plot(recall, precision, color='#F18F01', lw=3,
             label=f'{model_name} (AP = {ap_score:.4f})')
    plt.xlabel('Recall', fontsize=14)
    plt.ylabel('Precision', fontsize=14)
    plt.title('Precision-Recall Curve', fontsize=16, fontweight='bold')
    plt.legend(loc="upper right", fontsize=12)
    plt.grid(True, alpha=0.3)

    save_file = os.path.join(save_path, 'pr_curve.png')
    plt.savefig(save_file, dpi=300, bbox_inches='tight')
    print(f"PR曲线已保存至: {save_file}")
    plt.close()

    print(f"\n{'='*50}")
    print(f"PR分析结果 - {model_name}")
    print(f"{'='*50}")
    print(f"平均精确率 (AP): {ap_score:.4f}")
    print(f"{'='*50}\n")


if __name__ == '__main__':
    # 示例：绘制ROC曲线
    true_labels = np.array([0, 0, 1, 1, 0, 1, 0, 0, 1, 1] * 50)
    prob_scores = np.concatenate([
        np.random.uniform(0, 0.3, 50),  # 正常类的概率
        np.random.uniform(0.7, 1, 50)    # 不当言论类的概率
    ])

    plot_roc_curve(true_labels, prob_scores)
    plot_pr_curve(true_labels, prob_scores)

    # 示例：绘制多个ROC曲线对比
    results = {
        'BERT + SupCon': (true_labels, prob_scores),
        'BERT + InfoNCE': (true_labels, np.clip(prob_scores + np.random.normal(0, 0.05, len(prob_scores)), 0, 1)),
        'BERT Only': (true_labels, np.clip(prob_scores + np.random.normal(0, 0.1, len(prob_scores)), 0, 1))
    }
    plot_multiple_roc(results)