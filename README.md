# 基于对比学习的不当言论检测研究

## 环境搭建

1. 安装conda

   参考：https://www.anaconda.com/docs/getting-started/miniconda/install/overview

2. 在conda中创建虚拟环境：hate_speech_det

   ```shell
   # 选择的是python3.9下的latest版本
   conda create -n hate_speech_det python=3.9
   ```

3. 安装相关深度学习包

## 项目结构

```
HateSpeechDetection/
├── data/                   # 存放原始数据和处理后的数据
│   ├── raw/                # 下载的原始 COLD 数据集 (train.csv, dev.csv等)
│   └── processed/          # 运行预处理脚本后生成的规范化数据
├── models/                 # 存放模型架构定义
│   ├── encoder.py          # 文本编码器（如基于 BERT 的实现） 
│   ├── contrastive.py      # 对比学习模块与损失函数 
│   └── classifier.py       # 分类头模块 [cite: 4]
├── scripts/                # 存放各种功能脚本
│   ├── preprocess.py       # 数据清洗、脱敏与增强脚本 
│   ├── train.py            # 模型训练脚本 
│   └── evaluate.py         # 实验验证与性能评估脚本 
├── app/                    # 系统开发相关
│   ├── api.py              # Flask 后端 RESTful API 
│   └── web_ui.py           # Gradio 交互式前端界面 
├── utils/                  # 存放通用工具函数（如日志管理、配置读取）
├── requirements.txt        # 项目依赖清单
└── checkpoints/            # 存放训练好的模型权重文件
```

## 数据准备与处理

### 数据集来源

#### 中文数据集

| 数据集名称                                | 描述                                                         | 数据规模             | 数据格式 | 来源/下载地址                                                |
| ----------------------------------------- | ------------------------------------------------------------ | -------------------- | -------- | :----------------------------------------------------------- |
| COLD (Chinese Offensive Language Dataset) | 目前最权威的中文攻击性语言数据集，涵盖性别、种族、地域等维度的攻击。 | 3.7万条 (知乎、微博) | CSV      | https://github.com/thu-coai/COLDataset                       |
| TOXICN                                    | 包含知乎和贴吧数据，针对中文语境下的毒性言论（Toxicity）进行了精细分类。 | 约1.2万条            | JSON     | https://www.scidb.cn/en/detail?dataSetId=32236889f4c54c07a044fea962cb2043 |

#### 英文数据集

| 数据集名称                                    | 描述                                                      | 数据规模          | 数据格式 | 来源/下载地址                                                |
| --------------------------------------------- | --------------------------------------------------------- | ----------------- | -------- | :----------------------------------------------------------- |
| Hate Speech and Offensive Language (Davidson) | NLP领域的经典数据集，分为 Hate、Offensive、Neither 三类。 | 2.5万条 (Twitter) | CSV      | https://www.kaggle.com/datasets/mrmorj/hate-speech-and-offensive-language-dataset |

### 数据集处理

核心思路：利用`jieba（中文）`和`nltk（英文）`分词，进行同义词替换后得到增强后的数据`text_aug`

脚本主要包含：

1. 词级别句子划分
2. 去除停用词：提取核心实词
3. 同义词替换：
   - 中文使用`synonyms`库：https://blog.csdn.net/jcjy_baiyang/article/details/138375629
   - 英文使用`nltk`库

核心代码：

```python
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
```

## 模型开发与训练

基线模型：BERT

联合损失函数

