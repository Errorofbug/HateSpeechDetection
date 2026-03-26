import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'  # 启用 Hugging Face 国内镜像

import torch
import pandas as pd
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import sys

# 导入你写好的 dataset 和 model
from dataset import HateSpeechDataset

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from models.model import ContrastiveHateSpeechModel


def evaluate():
    # 1. 基础设置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"正在使用 {device} 进行评估...")

    # 2. 加载测试数据 (请确保路径指向你之前划分好的 test.csv)
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    test_dataset = HateSpeechDataset('../data/processed/test.csv', tokenizer)
    # 评估时不需要打乱数据
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    # 3. 加载训练好的模型权重
    model = ContrastiveHateSpeechModel('bert-base-chinese').to(device)
    model_path = '../checkpoints/best_model.pth'

    if not os.path.exists(model_path):
        print(f"错误：找不到模型权重文件 {model_path}，请先运行 train.py")
        return

    # 将保存的权重加载到模型中
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()  # 切换到评估模式，关闭 Dropout 等机制

    # 4. 开始预测
    all_preds = []
    all_labels = []

    print("开始对测试集进行预测，请稍候...")
    with torch.no_grad():  # 评估时不需要计算梯度，节省内存和算力
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].numpy()

            # 我们只需要分类头的输出 logits，不需要投影层输出
            _, logits = model(input_ids, attention_mask)

            # 取概率最大的类别作为预测结果
            preds = torch.argmax(logits, dim=1).cpu().numpy()

            all_preds.extend(preds)
            all_labels.extend(labels)

    # 5. 计算并打印评估指标
    acc = accuracy_score(all_labels, all_preds)
    # 因为是不当言论检测，我们重点关注 label=1 的召回和精准，average='binary'
    prec = precision_score(all_labels, all_preds, average='binary', zero_division=0)
    rec = recall_score(all_labels, all_preds, average='binary', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='binary', zero_division=0)

    print("\n" + "=" * 30)
    print("模型评估结果 (测试集)")
    print("=" * 30)
    print(f"准确率 (Accuracy) : {acc:.4f}")
    print(f"精确率 (Precision): {prec:.4f}")
    print(f"召回率 (Recall)   : {rec:.4f}")
    print(f"F1 分数 (F1-Score): {f1:.4f}")
    print("=" * 30)

    # 打印详细的分类报告，论文凑字数/写分析的神器
    print("\n详细分类报告:")
    print(classification_report(all_labels, all_preds, target_names=['正常言论(0)', '不当言论(1)'], zero_division=0))


if __name__ == "__main__":
    evaluate()