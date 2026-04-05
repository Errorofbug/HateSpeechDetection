# 基于对比学习的不当言论检测研究

## 环境搭建

1. **安装conda**

   参考：https://www.anaconda.com/docs/getting-started/miniconda/install/overview

2. **在conda中创建虚拟环境**：hate_speech_det

   ```shell
   # 选择的是python3.9下的latest版本
   conda create -n hate_speech_det python=3.9
   ```

3. **安装相关深度学习包**

## 项目结构

```
hate-speech-det/
├── data/                    # 数据目录
│   ├── raw/                # 原始数据集 (COLD, TOXICN, Davidson)
│   └── processed/          # 处理后的数据 (含增强文本 text_aug)
├── models/                 # 模型定义
│   └── model.py           # 对比学习不当言论检测模型
├── scripts/               # 核心脚本
│   ├── dataset.py         # 数据加载与预处理
│   ├── train.py           # 模型训练脚本
│   ├── evaluate.py        # 评估脚本
│   └── data_integration.py # 数据整合脚本
├── checkpoints/           # 训练好的模型权重
├── templates/             # Flask 前端模板
│   └── index.html         # 检测系统网页界面
├── app.py                 # Flask 后端服务器
├── resources/             # 停用词等资源文件
└── README.md              # 项目说明
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

### 数据增强

核心思路：利用`jieba（中文）`和`nltk（英文）`分词，进行同义词替换后得到增强后的数据`text_aug`

`scripts/data_intergation.py`脚本主要包含：

1. **词级别句子划分**
2. **去除停用词**：提取核心实词。
3. **同义词替换**：
   - 中文使用`synonyms`库：https://blog.csdn.net/jcjy_baiyang/article/details/138375629。
   - 英文使用`nltk`库。

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

### 模型架构

本项目采用**对比学习**与**BERT 文本编码器**相结合的架构，核心模型定义在 `models/model.py` 中：

```python
class ContrastiveHateSpeechModel(nn.Module):
    def __init__(self, model_name='bert-base-chinese', projection_dim=128):
        super(ContrastiveHateSpeechModel, self).__init__()
        # 1. 文本编码器 (Text Encoder)
        self.encoder = BertModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size
        # 2. 对比学习模块 / 投影层 (Projector)
        self.projector = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, projection_dim)
        )
        # 3. 分类头 (Classifier Head)
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(hidden_size, 2)
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = outputs.pooler_output
        projected_feature = self.projector(cls_embedding)  # 用于对比学习的特征
        logits = self.classifier(cls_embedding)            # 用于分类的 logits
        return projected_feature, logits
```

模型包含三个核心组件：
1. **文本编码器**：基于预训练的 BERT（`bert-base-chinese` 或 `bert-base-uncased`），将输入文本映射为语义向量。
2. **投影层**：将 BERT 输出的高维向量（768 维）映射到低维空间（如 128 维），用于计算对比损失，提升特征的判别性。
3. **分类头**：在 BERT 输出的 [CLS] 向量上添加一个 Dropout 和线性层，输出二分类（正常言论/不当言论）的 logits。

### 对比学习策略

本项目采用了**有监督对比学习（Supervised Contrastive Learning，SupCon）** 策略，以充分利用标注信息提升模型性能：

- **正样本构造**：每个样本由**原句-增强句对**组成，通过同义词替换得到语义相近的变体（详见数据增强部分）。
- **监督信号利用**：与传统无监督对比学习（InfoNCE）仅拉近原句与增强句不同，SupCon 额外利用了类别标签信息：
  - 同一类别的样本（如都是不当言论或都是正常言论）的特征向量应相互接近。
  - 不同类别的样本特征向量应相互远离。
  - 原句与增强句作为同一语义的正样本对，其特征向量应更加接近。

这种策略使得模型不仅能学习到语义不变性（同义句应具有相似特征），还能学习到类内紧凑性和类间可分性，从而提升分类性能。

### 损失函数

总损失由**分类损失**和**对比损失**加权组成，联合损失公式如下：

$$
\mathcal{L} = \mathcal{L}_{\text{CE}} + \lambda \cdot \mathcal{L}_{\text{con}}
$$

其中：
- $\mathcal{L}_{\text{CE}}$ 是分类任务的交叉熵损失。
- $\mathcal{L}_{\text{con}}$ 是对比损失，可采用无监督（InfoNCE）或有监督（SupCon）形式。
- $\lambda$ 是对比损失的权重（默认 0.1）。

#### 1. 无监督对比损失（InfoNCE）

InfoNCE（Info Noise-Contrastive Estimation）是无监督对比学习的经典损失，仅利用原句-增强句对作为正样本：

```python
def info_nce_loss(features_original, features_augmented, temperature=0.05):
    # 归一化
    features_original = F.normalize(features_original, p=2, dim=1)
    features_augmented = F.normalize(features_augmented, p=2, dim=1)
    # 计算相似度矩阵
    similarity_matrix = torch.matmul(features_original, features_augmented.T) / temperature
    batch_size = features_original.size(0)
    labels = torch.arange(batch_size).to(features_original.device)
    # 交叉熵损失
    loss = F.cross_entropy(similarity_matrix, labels)
    return loss
```

该损失强制**同一句子的原句与增强句在特征空间接近**，而与batch内其他句子的特征远离。

#### 2. 有监督对比损失（SupCon）

SupCon（Supervised Contrastive Loss）扩展了InfoNCE，额外利用类别标签信息，使**同一类别的样本相互接近，不同类别的样本相互远离**：

```python
def supcon_loss(features_original, features_augmented, labels, temperature=0.05):
    device = features_original.device
    batch_size = features_original.shape[0]

    # 1. L2 归一化
    features_original = F.normalize(features_original, p=2, dim=1)
    features_augmented = F.normalize(features_augmented, p=2, dim=1)

    # 2. 将原句和增强句拼接，得到 2N 个样本
    features = torch.cat([features_original, features_augmented], dim=0)
    labels = torch.cat([labels, labels], dim=0)

    # 3. 构造掩码矩阵：标签相同的样本对为 1
    labels = labels.contiguous().view(-1, 1)
    mask = torch.eq(labels, labels.T).float().to(device)

    # 4. 计算相似度矩阵并减去最大值（数值稳定）
    anchor_dot_contrast = torch.matmul(features, features.T) / temperature
    logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
    logits = anchor_dot_contrast - logits_max.detach()

    # 5. 消除自身对比（对角线置零）
    logits_mask = torch.scatter(
        torch.ones_like(mask),
        1,
        torch.arange(batch_size * 2).view(-1, 1).to(device),
        0
    )
    mask = mask * logits_mask

    # 6. 计算对数概率
    exp_logits = torch.exp(logits) * logits_mask
    log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

    # 7. 计算正样本平均损失
    mask_pos_pairs = mask.sum(1)
    mask_pos_pairs = torch.where(mask_pos_pairs < 1e-6, torch.ones_like(mask_pos_pairs), mask_pos_pairs)
    mean_log_prob_pos = (mask * log_prob).sum(1) / mask_pos_pairs

    # 8. 最终损失
    loss = - mean_log_prob_pos.mean()
    return loss
```

**当前训练中采用的是 SupCon 损失**，因为它能同时利用数据增强产生的正样本对和类别监督信号，提升模型的判别能力。

### 训练流程

训练脚本 `scripts/train.py` 的主要步骤：

1. **配置**：设置设备（GPU/CPU）、批次大小、学习率、训练轮数等。
2. **数据加载**：使用 `HateSpeechDataset` 加载已增强的数据（原句 `text` 与增强句 `text_aug`）。
3. **模型初始化**：加载 `ContrastiveHateSpeechModel` 和优化器（AdamW）。
4. **训练循环**：
   - 对每个 batch，分别计算原句和增强句的投影特征及分类 logits。
   - 计算分类损失和对比损失，并按权重相加得到总损失。
   - 反向传播并更新参数。
5. **模型保存**：训练完成后将模型权重保存至 `checkpoints/best_model.pth`。

### 快速开始训练

```bash
# 进入项目根目录
cd /path/to/hate-speech-det

# 运行训练脚本（默认使用 mini_train.csv 进行快速测试）
python scripts/train.py
```

如需完整训练，请修改 `train.py` 中的数据集路径并将 `epochs` 调整为 3‑5。

## 端到端的检测系统

本项目提供了一个基于 **Flask** 的 Web 应用，允许用户通过浏览器实时检测文本是否为不当言论。

### 核心推理逻辑

**模型加载**：启动时加载预训练的 `ContrastiveHateSpeechModel` 和 BERT 分词器。

核心推理代码：

```python
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    text = data.get('text', '').strip()
    # 分词与编码
    inputs = tokenizer(text, return_tensors='pt', max_length=128,
                       truncation=True, padding='max_length')
    # 模型推理
    with torch.no_grad():
        _, logits = model(input_ids, attention_mask)
        probabilities = torch.softmax(logits, dim=1)[0]
        pred_label = torch.argmax(probabilities).item()
        confidence = probabilities[pred_label].item()
    # 返回 JSON 结果
    result = {
        'label': int(pred_label),
        'label_name': '不当言论 🚨' if pred_label == 1 else '正常言论 ✅',
        'confidence': f"{confidence * 100:.2f}%"
    }
    return jsonify(result)
```

### 启动检测系统

1. **确保模型已训练**：检查 `checkpoints/best_model.pth` 是否存在，若不存在请先运行训练脚本。

2. **启动Flask服务器**：
   
   ```bash
   cd /path/to/hate-speech-det
   python app.py
   ```
   服务器将在 `http://127.0.0.1:5000` 启动。

### 注意事项

- 首次启动时会从 Hugging Face 镜像下载 BERT 分词器和模型参数（若未缓存），请确保网络通畅。
- 默认使用 `bert-base-chinese` 模型，如需检测英文文本，请修改 `app.py` 中的模型名称并重新训练。
