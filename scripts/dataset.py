import torch
from torch.utils.data import Dataset
import pandas as pd

class HateSpeechDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_len=128):
        """
        :param file_path: 你的 train.csv 或 dev.csv 的路径
        :param tokenizer: BERT 的分词器
        :param max_len: 句子截断的最大长度
        """
        self.data = pd.read_csv(file_path)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # 提取原句、增强句和标签
        text = str(self.data.iloc[index]['text'])
        text_aug = str(self.data.iloc[index]['text_aug'])
        label = int(self.data.iloc[index]['label'])

        # 对原句进行编码
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,    # 自动添加 [CLS] 和 [SEP]
            max_length=self.max_len,
            padding='max_length',       # 不够长的补齐
            truncation=True,            # 太长的截断
            return_attention_mask=True,
            return_tensors='pt'         # 返回 PyTorch 的 Tensor 格式
        )

        # 对增强句（正样本）进行相同的编码
        encoding_aug = self.tokenizer(
            text_aug,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        # 返回模型需要的字典格式
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'input_ids_aug': encoding_aug['input_ids'].flatten(),
            'attention_mask_aug': encoding_aug['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }