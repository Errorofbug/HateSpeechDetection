import pandas as pd
# 读取你的全量训练集
df = pd.read_csv('data/processed/train.csv') # 或者 full_train.csv，取决于你上一步的名字
# 取前 100 条，保存为测试用的 mini 版本
df.head(1600).to_csv('data/processed/mini_train.csv', index=False, encoding='utf-8-sig')