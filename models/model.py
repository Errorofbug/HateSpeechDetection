import torch
import torch.nn as nn
from transformers import BertModel


class ContrastiveHateSpeechModel(nn.Module):
    def __init__(self, model_name='bert-base-chinese', projection_dim=128):
        super(ContrastiveHateSpeechModel, self).__init__()

        # 1. 文本编码器 (Text Encoder)
        self.encoder = BertModel.from_pretrained(model_name)

        # 提取 BERT 输出的维度（通常 base 版本是 768 维）
        hidden_size = self.encoder.config.hidden_size

        # 2. 对比学习模块 / 投影层 (Projector)
        # 将 768 维映射到低维空间 (如 128 维) 用于计算对比损失，这能提升对比学习的泛化性
        self.projector = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, projection_dim)
        )

        # 3. 分类头 (Classifier Head)
        # 用于最终检测任务，判断 0 (正常) 还是 1 (不当言论)
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),  # 防止过拟合
            nn.Linear(hidden_size, 2)
        )

    def forward(self, input_ids, attention_mask):
        """
        前向传播函数
        """
        # 将文本输入 BERT 编码器
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)

        # 获取 [CLS] 标记的向量表示，它代表了整个句子的全局语义
        cls_embedding = outputs.pooler_output

        # 经过投影层，得到用于对比学习的特征向量
        projected_feature = self.projector(cls_embedding)

        # 经过分类头，得到预测类别的对数几率 (Logits)
        logits = self.classifier(cls_embedding)

        return projected_feature, logits