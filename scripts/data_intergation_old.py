import pandas as pd
import json
import os
import re
import random
import jieba


class DataProcessor:
    def __init__(self, output_path="data/processed/full_train.csv"):
        self.output_path = output_path
        self.all_data = []

    def clean_chinese(self, text):
        """中文清洗：脱敏、去符号"""
        text = re.sub(r'@\S+', '[USER]', str(text))  # 脱敏
        text = re.sub(r'https?://\S+', '[URL]', text)  # 去链接
        return text.strip()

    def clean_english(self, text):
        """英文清洗：转小写、脱敏"""
        text = str(text).lower()
        text = re.sub(r'@\S+', '[USER]', text)
        return text.strip()

    def augment_text(self, text, lang='zh'):
        """
        构造对比学习正样本 (简单策略：随机删词)
        更高级的可以用同义词替换或回译
        """
        if lang == 'zh':
            words = list(jieba.cut(text))
        else:
            words = text.split()

        if len(words) > 5:
            # 随机删除一个词
            idx = random.randint(0, len(words) - 1)
            words.pop(idx)

        return "".join(words) if lang == 'zh' else " ".join(words)

    def load_file(self, file_path, text_col, label_col, lang='zh'):
        """加载单个文件并标准化"""
        print(f"正在处理: {file_path}")
        ext = os.path.splitext(file_path)[-1].lower()

        try:
            if ext == '.csv':
                df = pd.read_csv(file_path)
            elif ext == '.json':
                df = pd.read_json(file_path)
            else:
                print(f"不支持的格式: {ext}")
                return

            # 提取需要的列并统一命名
            standard_df = pd.DataFrame()
            standard_df['text'] = df[text_col].apply(self.clean_chinese if lang == 'zh' else self.clean_english)
            standard_df['label'] = df[label_col].astype(int)
            standard_df['lang'] = lang

            # 生成对比学习样本对
            standard_df['text_aug'] = standard_df['text'].apply(lambda x: self.augment_text(x, lang))

            self.all_data.append(standard_df)
        except Exception as e:
            print(f"处理文件 {file_path} 出错: {e}")

    def save_integrated_data(self):
        """合并所有数据并保存"""
        if not self.all_data:
            print("没有可合并的数据！")
            return

        final_df = pd.concat(self.all_data, ignore_index=True)
        # 打乱顺序，防止模型学习到文件的顺序特征
        final_df = final_df.sample(frac=1).reset_index(drop=True)

        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        final_df.to_csv(self.output_path, index=False, encoding='utf-8-sig')
        print(f"所有数据已整合并保存至: {self.output_path}")
        print(f"总数据量: {len(final_df)} 条")


if __name__ == "__main__":
    processor = DataProcessor()

    # --- 这里根据你下载的实际文件路径和列名进行配置 ---

    # 示例1：处理 COLD (中文 CSV)
    processor.load_file(
        file_path='data/raw/COLDataset/dev.csv',
        text_col='TEXT',
        label_col='label',
        lang='zh'
    )

    processor.load_file(
        file_path='data/raw/toxicndata/dev.json',
        text_col='content',
        label_col='toxic',
        lang='zh'
    )

    # 示例2：处理某个英文 JSON (假设列名是 'comment' 和 'is_toxic')
    # processor.load_file(
    #     file_path='data/raw/EnglishData/test.json',
    #     text_col='comment',
    #     label_col='is_toxic',
    #     lang='en'
    # )

    # 执行合并
    processor.save_integrated_data()