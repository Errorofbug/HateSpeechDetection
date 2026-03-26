import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'  # 启用 Hugging Face 国内镜像

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from torch.optim import AdamW

from dataset import HateSpeechDataset
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from models.model import ContrastiveHateSpeechModel


def info_nce_loss(features_original, features_augmented, temperature=0.05):
    """
    计算对比损失 (InfoNCE)
    features_original: 原句经过投影层后的向量 [batch_size, dim]
    features_augmented: 增强句经过投影层后的向量 [batch_size, dim]
    """
    # 1. 对向量进行 L2 归一化，这样点乘就等于余弦相似度
    features_original = F.normalize(features_original, p=2, dim=1)
    features_augmented = F.normalize(features_augmented, p=2, dim=1)

    # 2. 计算相似度矩阵 (Batch 内两两相乘)
    # 结果矩阵的对角线元素就是正样本对的相似度，非对角线是负样本对
    similarity_matrix = torch.matmul(features_original, features_augmented.T) / temperature

    # 3. 构造标签：对角线上的索引 (0, 1, 2... batch_size-1) 就是正样本的正确位置
    batch_size = features_original.size(0)
    labels = torch.arange(batch_size).to(features_original.device)

    # 4. 用交叉熵来优化这个相似度矩阵
    loss = F.cross_entropy(similarity_matrix, labels)
    return loss


def train():
    # --- 1. 基础配置 ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"当前使用的计算设备: {device}")

    batch_size = 16
    # epochs = 3
    epochs = 1 # 这里本地跑的时候用1
    learning_rate = 2e-5
    lambda_weight = 0.1  # 对比损失的权重

    # --- 2. 加载数据和模型 ---
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

    # train_dataset = HateSpeechDataset('../data/processed/train.csv', tokenizer)
    train_dataset = HateSpeechDataset('../data/processed/mini_train.csv', tokenizer) # 本地跑使用
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    model = ContrastiveHateSpeechModel('bert-base-chinese').to(device)
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    criterion_ce = nn.CrossEntropyLoss()

    # --- 3. 开始训练循环 ---
    model.train()
    for epoch in range(epochs):
        total_train_loss = 0

        for step, batch in enumerate(train_loader):
            optimizer.zero_grad()

            # 将数据推送到 GPU/CPU
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            input_ids_aug = batch['input_ids_aug'].to(device)
            attention_mask_aug = batch['attention_mask_aug'].to(device)
            labels = batch['labels'].to(device)

            # 前向传播：原句
            proj_orig, logits = model(input_ids, attention_mask)
            # 前向传播：增强句 (为了算对比损失)
            proj_aug, _ = model(input_ids_aug, attention_mask_aug)

            # 计算分类损失
            loss_ce = criterion_ce(logits, labels)
            # 计算对比损失
            loss_con = info_nce_loss(proj_orig, proj_aug)

            # 联合损失！这就是你论文的核心！
            loss = loss_ce + lambda_weight * loss_con

            # 反向传播与参数更新
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

            if step % 10 == 0:
                print(f"Epoch [{epoch + 1}/{epochs}], Step [{step}/{len(train_loader)}], "
                      f"Loss: {loss.item():.4f} (CE: {loss_ce.item():.4f}, Con: {loss_con.item():.4f})")

        avg_loss = total_train_loss / len(train_loader)
        print(f"=== Epoch {epoch + 1} 完成, 平均 Loss: {avg_loss:.4f} ===")

    # --- 4. 保存模型 ---
    os.makedirs('../checkpoints', exist_ok=True)
    torch.save(model.state_dict(), '../checkpoints/best_model.pth')
    print("模型训练完毕并已保存至 checkpoints 目录！")


if __name__ == "__main__":
    train()