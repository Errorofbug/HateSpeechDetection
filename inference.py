import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import torch
from transformers import BertTokenizer
from models.model import ContrastiveHateSpeechModel


class HateSpeechInference:
    """仇恨言论检测推理引擎"""

    def __init__(self, model_path: str = 'checkpoints/model_supcon.pth', device: str = None):
        """
        初始化推理引擎

        Args:
            model_path: 模型权重路径
            device: 指定设备 ('cuda' 或 'cpu')，默认自动检测
        """
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))

        # 加载分词器
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

        # 加载模型
        self.model = ContrastiveHateSpeechModel('bert-base-chinese').to(self.device)

        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            print(f"[推理] 成功加载权重: {model_path}")
        else:
            print(f"[推理] 警告: 未找到 {model_path}，模型处于初始化状态")

        self.model.eval()

    def predict(self, text: str, max_length: int = 128) -> dict:
        """
        预测单个文本

        Args:
            text: 待预测文本
            max_length: 最大序列长度

        Returns:
            包含预测结果的字典:
            - label: 预测标签 (0=正常, 1=不当言论)
            - label_name: 标签名称
            - confidence: 置信度 (百分比)
            - probabilities: 各类别概率
        """
        if not text or not text.strip():
            return {'error': '文本不能为空'}

        # 文本编码
        inputs = self.tokenizer(
            text.strip(),
            return_tensors='pt',
            max_length=max_length,
            truncation=True,
            padding='max_length'
        )
        input_ids = inputs['input_ids'].to(self.device)
        attention_mask = inputs['attention_mask'].to(self.device)

        # 推理
        with torch.no_grad():
            _, logits = self.model(input_ids, attention_mask)
            probabilities = torch.softmax(logits, dim=1)[0]
            pred_label = torch.argmax(probabilities).item()
            confidence = probabilities[pred_label].item()

        return {
            'label': int(pred_label),
            'label_name': '不当言论 🚨' if pred_label == 1 else '正常言论 ✅',
            'confidence': confidence,
            'probabilities': probabilities.cpu().tolist()
        }

    def predict_batch(self, texts: list, max_length: int = 128) -> list:
        """
        批量预测

        Args:
            texts: 待预测文本列表
            max_length: 最大序列长度

        Returns:
            预测结果列表
        """
        results = []
        for text in texts:
            results.append(self.predict(text, max_length))
        return results