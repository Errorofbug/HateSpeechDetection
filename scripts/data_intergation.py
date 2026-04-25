"""
数据整合脚本 - 读取多个数据源，增强后统一划分并保存

使用方法:
    python scripts/data_intergation.py                    # 使用默认配置
    python scripts/data_intergation.py --stopwords path   # 指定停用词路径
"""
import sys
import os

# 获取项目根目录
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

# 必须先导入项目 utils，避免与 synonyms.utils 冲突
from utils import logger, config

import pandas as pd
import re
import random
import jieba
import synonyms
import nltk
from nltk.corpus import wordnet
from sklearn.model_selection import train_test_split


class DataProcessor:
    def __init__(self, stopwords_path=None):
        """
        初始化，加载停用词表

        参数:
            stopwords_path: 停用词表路径（None则使用默认路径）
        """
        self.config = config

        # 获取停用词路径
        if stopwords_path is None:
            stopwords_path = os.path.join(PROJECT_ROOT, 'resources/baidu_stopwords.txt')
        elif not os.path.isabs(stopwords_path):
            stopwords_path = os.path.join(PROJECT_ROOT, stopwords_path)

        self.cn_stopwords = self._load_stopwords(stopwords_path)
        self.logger = logger

        self.logger.info(f"成功加载中文停用词 {len(self.cn_stopwords)} 个")

    def _load_stopwords(self, path):
        """加载停用词表"""
        stopwords = set()
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    word = line.strip()
                    if word:
                        stopwords.add(word)
        else:
            self.logger.warning(f"未找到停用词文件 {path}，使用默认停用词")
            stopwords = {"的", "了", "在", "是", "我", "你", "他", "这", "那"}
        return stopwords

    def _clean_text(self, text, lang='zh'):
        """基础清洗（保留完整句子给BERT处理）"""
        text = str(text)
        text = re.sub(r'@\S+', '[USER]', text)  # 脱敏
        text = re.sub(r'https?://\S+', '[URL]', text)  # 去链接
        return text.strip().lower() if lang == 'en' else text.strip()

    def augment_text(self, text, lang='zh'):
        """
        核心数据增强逻辑
        """
        if lang == 'zh':
            raw_words = list(jieba.cut(text))
            valid_words = [w for w in raw_words if w not in self.cn_stopwords and w.strip()]
        else:
            raw_words = nltk.word_tokenize(text)
            valid_words = [w for w in raw_words if len(w) > 2]

        if len(valid_words) < 2:
            return text  # 句子太短或全是停用词，放弃增强

        new_words = raw_words.copy()

        # 随机挑选1-2个有效实词进行同义词替换
        replace_count = min(2, max(1, int(len(valid_words) * 0.2)))
        targets = random.sample(valid_words, replace_count)

        for i, word in enumerate(new_words):
            if word in targets:
                if lang == 'zh':
                    nearby = synonyms.nearby(word)
                    if nearby and len(nearby[0]) > 1:
                        new_words[i] = nearby[0][1]
                else:
                    syns = wordnet.synsets(word)
                    if syns:
                        for l in syns[0].lemmas():
                            if l.name() != word:
                                new_words[i] = l.name().replace('_', ' ')
                                break

        return "".join(new_words) if lang == 'zh' else " ".join(new_words)

    def process_and_split(self, config_list, output_dir=None):
        """
        读取多个数据源，增强后统一划分并保存

        参数:
            config_list: 数据配置列表，格式为 [(路径, 文本列, 标签列, 语言), ...]
            output_dir: 输出目录（None则从配置读取）
        """
        if output_dir is None:
            output_dir = self.config.get_path('data_processed_dir')

        all_dfs = []

        for path, t_col, l_col, lang in config_list:
            # 转换为绝对路径
            if not os.path.isabs(path):
                path = os.path.join(PROJECT_ROOT, path)

            self.logger.info(f"读取文件: {path}")
            df = pd.read_json(path) if path.endswith('.json') else pd.read_csv(path)

            temp_df = pd.DataFrame()
            temp_df['text'] = df[t_col].apply(lambda x: self._clean_text(x, lang))
            temp_df['label'] = df[l_col].astype(int)
            temp_df['lang'] = lang

            self.logger.info(f"正在为 {lang} 数据构造对比样本对...")
            temp_df['text_aug'] = temp_df['text'].apply(lambda x: self.augment_text(x, lang))
            all_dfs.append(temp_df)

        full_df = pd.concat(all_dfs, ignore_index=True)
        full_df = full_df[full_df['text'].str.len() > 0]

        self.logger.info(f"总数据量: {len(full_df)}，开始按 8:1:1 划分数据集...")
        train_val, test_df = train_test_split(full_df, test_size=0.1, stratify=full_df['label'], random_state=42)
        train_df, dev_df = train_test_split(train_val, test_size=0.111, stratify=train_val['label'], random_state=42)

        os.makedirs(output_dir, exist_ok=True)
        train_df.to_csv(os.path.join(output_dir, 'train.csv'), index=False, encoding='utf-8-sig')
        dev_df.to_csv(os.path.join(output_dir, 'dev.csv'), index=False, encoding='utf-8-sig')
        test_df.to_csv(os.path.join(output_dir, 'test.csv'), index=False, encoding='utf-8-sig')

        self.logger.info(f"数据集已保存至: {output_dir}")

        # 生成mini数据集（用于快速测试）
        self._create_mini_dataset(train_df, output_dir)

        return train_df, dev_df, test_df

    def _create_mini_dataset(self, train_df, output_dir, sample_size=1600):
        """创建mini数据集用于快速测试"""
        if len(train_df) > sample_size:
            mini_train = train_df.groupby('label', group_keys=False).apply(
                lambda x: x.sample(n=min(len(x), sample_size // 2), random_state=42)
            )
            mini_train.to_csv(os.path.join(output_dir, 'mini_train.csv'), index=False, encoding='utf-8-sig')
            self.logger.info(f"Mini数据集已保存至: {output_dir}/mini_train.csv ({len(mini_train)} 条)")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='数据整合脚本')
    parser.add_argument('--stopwords', type=str, default=None,
                       help='停用词文件路径')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='输出目录')

    args = parser.parse_args()

    processor = DataProcessor(stopwords_path=args.stopwords)

    # 数据配置：请根据实际数据位置修改
    # 格式：(文件路径, 文本列名, 标签列名, 语言)
    configs = [
        ('data/raw/COLDataset/dev.csv', 'TEXT', 'label', 'zh'),
        ('data/raw/toxicndata/dev.json', 'content', 'toxic', 'zh')
    ]

    processor.process_and_split(configs, output_dir=args.output_dir)