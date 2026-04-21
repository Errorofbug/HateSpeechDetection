"""
评估脚本 - 评估训练好的模型

使用方法:
    python scripts/evaluate.py                      # 评估默认模型
    python scripts/evaluate.py --model model_baseline  # 评估指定模型
"""
import os
import sys
from datetime import datetime

# 获取项目根目录
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import json

from scripts.dataset import HateSpeechDataset
from models.model import ContrastiveHateSpeechModel
from utils.logger import get_logger
from utils.config import get_config


def evaluate(model_name=None, save_results=None):
    """
    评估函数

    参数:
        model_name: 要加载的模型名称（None则从配置读取）
        save_results: 是否保存评估结果（None则从配置读取）

    返回:
        labels, preds, probs, metrics
    """
    # 加载配置
    config = get_config()

    # 从配置获取参数
    if model_name is None:
        model_name = config.get('evaluation', 'model_name', default='model_supcon')
    if save_results is None:
        save_results = config.get('evaluation', 'save_results', default=True)

    batch_size = config.get('evaluation', 'batch_size', default=16)
    results_dir = config.get('evaluation', 'results_dir', default='checkpoints/evaluation_results')

    # 初始化日志器
    log_dir = config.get_path('logs_dir')
    logger = get_logger('evaluate', log_dir=log_dir)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'evaluate_{timestamp}.log')
    logger.set_log_file(log_file)

    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"正在使用 {device} 进行评估...")
    logger.info(f"评估模型: {model_name}")

    # 加载测试数据
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    data_processed_dir = config.get_path('data_processed_dir')
    test_path = os.path.join(data_processed_dir, 'test.csv')

    test_dataset = HateSpeechDataset(test_path, tokenizer)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    logger.info(f"测试集大小: {len(test_dataset)}")

    # 加载模型
    model_config_name = config.get('model', 'model_name', default='bert-base-chinese')
    model = ContrastiveHateSpeechModel(model_config_name).to(device)

    checkpoints_dir = config.get_path('checkpoints_dir')
    model_path = os.path.join(checkpoints_dir, f'{model_name}.pth')
    backup_path = os.path.join(checkpoints_dir, 'best_model.pth')

    if not os.path.exists(model_path):
        model_path = backup_path
        if not os.path.exists(model_path):
            logger.error(f"找不到模型权重文件")
            logger.error(f"尝试的路径: {model_path}")
            return None, None, None, None

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    logger.info(f"模型加载成功: {model_path}")

    # 开始预测
    all_preds = []
    all_labels = []
    all_probs = []

    logger.info("开始对测试集进行预测...")
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].numpy()

            _, logits = model(input_ids, attention_mask)

            probs = torch.softmax(logits, dim=1)
            prob_positive = probs[:, 1].cpu().numpy()

            preds = torch.argmax(logits, dim=1).cpu().numpy()

            all_preds.extend(preds)
            all_labels.extend(labels)
            all_probs.extend(prob_positive)

    # 计算评估指标
    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, average='binary', zero_division=0)
    rec = recall_score(all_labels, all_preds, average='binary', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='binary', zero_division=0)

    metrics = {
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1
    }

    # 记录结果
    logger.info("=" * 60)
    logger.info(f"模型评估结果 - {model_name}")
    logger.info("=" * 60)
    logger.info(f"准确率 (Accuracy) : {acc:.4f}")
    logger.info(f"精确率 (Precision): {prec:.4f}")
    logger.info(f"召回率 (Recall)   : {rec:.4f}")
    logger.info(f"F1 分数 (F1-Score): {f1:.4f}")
    logger.info("=" * 60)

    # 打印详细分类报告
    print("\n详细分类报告:")
    print(classification_report(all_labels, all_preds, target_names=['正常言论(0)', '不当言论(1)'], zero_division=0))

    # 保存结果
    if save_results:
        results_dir_abs = os.path.join(PROJECT_ROOT, results_dir)
        os.makedirs(results_dir_abs, exist_ok=True)

        results = {
            'model_name': model_name,
            'timestamp': timestamp,
            'labels': [int(x) for x in all_labels],
            'preds': [int(x) for x in all_preds],
            'probs': [float(x) for x in all_probs],
            'metrics': {
                'accuracy': float(acc),
                'precision': float(prec),
                'recall': float(rec),
                'f1': float(f1)
            }
        }

        results_file = os.path.join(results_dir_abs, f'{model_name}_results.json')
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        logger.info(f"评估结果已保存至: {results_file}")

    logger.info(f"日志已保存至: {log_file}")

    return all_labels, all_preds, all_probs, metrics


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='评估对比学习不当言论检测模型')
    parser.add_argument('--model', type=str, default=None,
                       help='模型名称 (model_supcon, model_infonce, model_baseline)')
    parser.add_argument('--no-save', action='store_true',
                       help='不保存评估结果')

    args = parser.parse_args()

    evaluate(
        model_name=args.model,
        save_results=not args.no_save
    )