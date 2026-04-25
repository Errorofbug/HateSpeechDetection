import os
import sys
import numpy as np
from sklearn.manifold import TSNE
import seaborn as sns
import torch
from torch.utils.data import DataLoader


# 获取项目根目录
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

import matplotlib.pyplot as plt
import matplotlib

from utils import config

# 设置中文字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False


def extract_features(model, dataloader, device='cpu'):
    """
    从模型中提取特征向量

    参数:
        model: 训练好的模型
        dataloader: 数据加载器
        device: 计算设备

    返回:
        features: 特征向量数组 [N, D]
        labels: 标签数组 [N]
    """
    model.eval()
    all_features = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels']

            # 获取投影层输出的特征
            features, _ = model(input_ids, attention_mask)
            all_features.append(features.cpu().numpy())
            all_labels.extend(labels.numpy())

    features = np.vstack(all_features)
    labels = np.array(all_labels)

    return features, labels


def plot_tsne(features, labels, title='t-SNE Visualization of Feature Space',
              save_path='documents/figures'):
    """
    使用t-SNE降维并可视化特征分布

    参数:
        features: 高维特征向量 [N, D]
        labels: 标签 [N]
        title: 图表标题
        save_path: 图片保存路径
    """
    os.makedirs(save_path, exist_ok=True)

    # 使用t-SNE降维到2维
    print("正在进行t-SNE降维，请稍候...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
    features_2d = tsne.fit_transform(features)
    print("t-SNE降维完成！")

    # 创建图表
    plt.figure(figsize=(12, 10))

    # 绘制散点图
    label_names = ['正常言论 (Normal)', '不当言论 (Toxic)']
    colors = ['#28a745', '#dc3545']
    markers = ['o', 's']

    for label in [0, 1]:
        mask = labels == label
        plt.scatter(features_2d[mask, 0], features_2d[mask, 1],
                   c=colors[label], marker=markers[label], s=80, alpha=0.6,
                   label=label_names[label], edgecolors='black', linewidth=0.5)

    plt.xlabel('t-SNE Dimension 1', fontsize=14)
    plt.ylabel('t-SNE Dimension 2', fontsize=14)
    plt.title(title, fontsize=16, fontweight='bold')
    plt.legend(fontsize=12, loc='best')
    plt.grid(True, alpha=0.3)

    save_file = os.path.join(save_path, 'tsne_visualization.png')
    plt.savefig(save_file, dpi=300, bbox_inches='tight')
    print(f"t-SNE可视化图已保存至: {save_file}")
    plt.close()

    # 计算类间距离和类内距离
    compute_cluster_separation(features_2d, labels)


def compute_cluster_separation(features_2d, labels):
    """
    计算类间分离度和类内紧密度

    参数:
        features_2d: 2维特征向量
        labels: 标签
    """
    print("\n" + "="*50)
    print("聚类分析结果")
    print("="*50)

    # 计算每个类的中心点
    class_centers = []
    for label in [0, 1]:
        class_features = features_2d[labels == label]
        center = np.mean(class_features, axis=0)
        class_centers.append(center)

        # 计算类内距离（到中心点的平均距离）
        intra_dist = np.mean(np.linalg.norm(class_features - center, axis=1))
        label_name = '正常言论' if label == 0 else '不当言论'
        print(f"{label_name} - 类内平均距离: {intra_dist:.4f}")

    # 计算类间距离（两个中心点之间的距离）
    inter_dist = np.linalg.norm(class_centers[0] - class_centers[1])
    print(f"\n类间距离 (两个类别中心): {inter_dist:.4f}")

    # 计算分离度指标（类间距离 / 类内距离）
    avg_intra = (np.mean(np.linalg.norm(features_2d[labels == 0] - class_centers[0], axis=1)) +
                 np.mean(np.linalg.norm(features_2d[labels == 1] - class_centers[1], axis=1))) / 2
    separation_ratio = inter_dist / avg_intra if avg_intra > 0 else 0

    print(f"分离度指标 (类间距离 / 类内距离): {separation_ratio:.4f}")
    print("="*50 + "\n")


def plot_multiple_tsne(results_dict, save_path='documents/figures'):
    """
    绘制多个模型的t-SNE对比

    参数:
        results_dict: 字典，格式为 {模型名: (features, labels)}
        save_path: 图片保存路径
    """
    os.makedirs(save_path, exist_ok=True)

    n_models = len(results_dict)
    fig, axes = plt.subplots(1, n_models, figsize=(7*n_models, 6))
    if n_models == 1:
        axes = [axes]

    fig.suptitle('t-SNE Visualization Comparison', fontsize=16, fontweight='bold')

    for idx, (model_name, (features, labels)) in enumerate(results_dict.items()):
        print(f"正在处理 {model_name} 的t-SNE降维...")
        tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
        features_2d = tsne.fit_transform(features)

        label_names = ['正常', '不当']
        colors = ['#28a745', '#dc3545']
        markers = ['o', 's']

        for label in [0, 1]:
            mask = labels == label
            axes[idx].scatter(features_2d[mask, 0], features_2d[mask, 1],
                            c=colors[label], marker=markers[label], s=60, alpha=0.6,
                            label=label_names[label], edgecolors='black', linewidth=0.5)

        axes[idx].set_xlabel('t-SNE Dim 1', fontsize=11)
        axes[idx].set_ylabel('t-SNE Dim 2', fontsize=11)
        axes[idx].set_title(model_name, fontsize=13, fontweight='bold')
        axes[idx].legend(fontsize=10, loc='best')
        axes[idx].grid(True, alpha=0.3)

    plt.tight_layout()
    save_file = os.path.join(save_path, 'tsne_comparison.png')
    plt.savefig(save_file, dpi=300, bbox_inches='tight')
    print(f"t-SNE对比图已保存至: {save_file}")
    plt.close()


if __name__ == '__main__':
    # 示例：生成随机数据并绘制t-SNE
    np.random.seed(42)

    # 生成两个簇的数据（模拟两类特征）
    n_samples = 100
    normal_features = np.random.randn(n_samples // 2, 128) + 2  # 类别0
    toxic_features = np.random.randn(n_samples // 2, 128) - 2   # 类别1
    features = np.vstack([normal_features, toxic_features])
    labels = np.array([0] * (n_samples // 2) + [1] * (n_samples // 2))

    plot_tsne(features, labels)

    # 示例：多个模型对比
    features_baseline = np.vstack([
        np.random.randn(50, 128) + 1,    # 基线模型：类间距离小
        np.random.randn(50, 128) - 1
    ])
    labels_baseline = np.array([0] * 50 + [1] * 50)

    features_contrastive = np.vstack([
        np.random.randn(50, 128) + 3,    # 对比学习：类间距离大
        np.random.randn(50, 128) - 3
    ])
    labels_contrastive = np.array([0] * 50 + [1] * 50)

    results = {
        'Baseline (BERT Only)': (features_baseline, labels_baseline),
        'Ours (BERT + Contrastive)': (features_contrastive, labels_contrastive)
    }

    plot_multiple_tsne(results)