import pandas as pd
import os
import re
import random
import jieba
import synonyms
import nltk
from nltk.corpus import wordnet
from sklearn.model_selection import train_test_split


class DataProcessor:
    def __init__(self, stopwords_path="resources/baidu_stopwords.txt"):
        """初始化，加载停用词表"""
        self.cn_stopwords = self._load_stopwords(stopwords_path)
        print(f"[初始化] 成功加载中文停用词 {len(self.cn_stopwords)} 个")

    def _load_stopwords(self, path):
        stopwords = set()
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    word = line.strip()
                    if word: stopwords.add(word)
        else:
            print(f"[警告] 未找到 {path}，使用基础保底停用词。")
            stopwords = {"的", "了", "在", "是", "我", "你", "他", "这", "那"}
        return stopwords

    def _clean_text(self, text, lang='zh'):
        """基础清洗（注意：这里不分词，保留完整句子给后续的BERT）"""
        text = str(text)
        text = re.sub(r'@\S+', '[USER]', text)  # 脱敏
        text = re.sub(r'https?://\S+', '[URL]', text)  # 去链接
        return text.strip().lower() if lang == 'en' else text.strip()

    def augment_text(self, text, lang='zh'):
        """
        核心数据增强逻辑
        """
        if lang == 'zh':
            # 1. 词级别切分
            raw_words = list(jieba.cut(text))
            # 2. 去停用词，提取核心实词用于增强
            valid_words = [w for w in raw_words if w not in self.cn_stopwords and w.strip()]
        else:
            raw_words = nltk.word_tokenize(text)
            valid_words = [w for w in raw_words if len(w) > 2]  # 英文简单过滤短词

        if len(valid_words) < 2:
            return text  # 句子太短或全是停用词，放弃增强

        new_words = raw_words.copy()  # 复制原句列表，保证生成的新句子结构完整

        # 策略：随机挑选 1-2 个有效实词进行同义词替换
        replace_count = min(2, max(1, int(len(valid_words) * 0.2)))
        targets = random.sample(valid_words, replace_count)

        for i, word in enumerate(new_words):
            if word in targets:
                if lang == 'zh':
                    nearby = synonyms.nearby(word)
                    if nearby and len(nearby[0]) > 1:
                        new_words[i] = nearby[0][1]  # 替换为最相近的词
                else:
                    syns = wordnet.synsets(word)
                    if syns:
                        for l in syns[0].lemmas():
                            if l.name() != word:
                                new_words[i] = l.name().replace('_', ' ')
                                break

        # 3. 将增强后的词列表重新无缝拼接成字符串（交给将来的BERT处理）
        return "".join(new_words) if lang == 'zh' else " ".join(new_words)

    def process_and_split(self, config_list, output_dir="data/processed"):
        """读取多个数据源，增强后统一划分并保存"""
        all_dfs = []
        for path, t_col, l_col, lang in config_list:
            print(f"[处理中] 读取文件: {path}")
            df = pd.read_json(path) if path.endswith('.json') else pd.read_csv(path)

            temp_df = pd.DataFrame()
            temp_df['text'] = df[t_col].apply(lambda x: self._clean_text(x, lang))
            temp_df['label'] = df[l_col].astype(int)
            temp_df['lang'] = lang

            print(f"[处理中] 正在为 {lang} 数据构造对比样本对...")
            temp_df['text_aug'] = temp_df['text'].apply(lambda x: self.augment_text(x, lang))
            all_dfs.append(temp_df)

        full_df = pd.concat(all_dfs, ignore_index=True)
        # 剔除清洗后变成空文本的脏数据
        full_df = full_df[full_df['text'].str.len() > 0]

        print(f"[整合] 总数据量: {len(full_df)}，开始按 8:1:1 划分数据集...")
        train_val, test_df = train_test_split(full_df, test_size=0.1, stratify=full_df['label'], random_state=42)
        train_df, dev_df = train_test_split(train_val, test_size=0.111, stratify=train_val['label'], random_state=42)

        os.makedirs(output_dir, exist_ok=True)
        train_df.to_csv(f"{output_dir}/train.csv", index=False, encoding='utf-8-sig')
        dev_df.to_csv(f"{output_dir}/dev.csv", index=False, encoding='utf-8-sig')
        test_df.to_csv(f"{output_dir}/test.csv", index=False, encoding='utf-8-sig')

        print("[完成] 数据集已保存至 data/processed/ 目录。")


if __name__ == "__main__":
    processor = DataProcessor()
    # 请根据你实际存放的数据修改这里的路径
    configs = [
        ('data/raw/COLDataset/dev.csv', 'TEXT', 'label', 'zh'),
        ('data/raw/toxicndata/dev.json', 'content', 'toxic', 'zh')
    ]
    processor.process_and_split(configs)