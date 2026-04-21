# 基于对比学习的不当言论检测研究

## 项目概述

本项目实现了一个基于对比学习的不当言论检测系统，使用BERT作为文本编码器，结合有监督对比学习（SupCon）提升模型性能。

## 环境搭建

### 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 首次运行会初始化配置文件
python scripts/train.py --help

# 3. 训练模型（默认使用完整数据集）
python scripts/train.py

# 4. 评估模型
python scripts/evaluate.py

# 5. 生成实验图表
python scripts/generate_figures.py
```

### 详细安装步骤

1. **安装conda**

   参考：https://www.anaconda.com/docs/getting-started/miniconda/install/overview

2. **在conda中创建虚拟环境**：hate_speech_det

   ```shell
   conda create -n hate_speech_det python=3.9
   conda activate hate_speech_det
   ```

3. **安装依赖**

   ```bash
   pip install -r requirements.txt
   ```

## 项目结构

```
hate-speech-det/
├── config.yaml              # 配置文件（统一管理所有参数）
├── data/                    # 数据目录
│   ├── raw/                # 原始数据集 (COLD, TOXICN, Davidson)
│   └── processed/          # 处理后的数据 (含增强文本 text_aug)
├── models/                 # 模型定义
│   └── model.py           # 对比学习不当言论检测模型
├── scripts/               # 核心脚本
│   ├── dataset.py         # 数据加载与预处理
│   ├── train.py           # 训练脚本（支持配置、时间戳日志）
│   ├── evaluate.py        # 评估脚本（支持配置、结果保存）
│   ├── data_intergation.py # 数据整合脚本
│   └── visualize_*.py     # 可视化脚本（支持配置）
├── utils/                 # 工具模块
│   ├── __init__.py       # 工具包初始化
│   ├── logger.py          # 统一日志模块
│   └── config.py           # 配置管理模块
├── checkpoints/           # 训练好的模型权重
├── logs/                  # 训练和评估日志（带时间戳）
├── docs/                  # 文档目录
│   └── figures/          # 生成的实验图表
├── templates/             # Flask 前端模板
│   └── index.html         # 检测系统网页界面
├── app.py                 # Flask 后端服务器
├── resources/             # 停用词等资源文件
└── requirements.txt       # 依赖清单
```

## 配置说明

项目使用 `config.yaml` 统一管理所有配置，避免代码中写死路径。

### 配置项

| 配置项 | 默认值 | 说明 |
|---------|---------|------|
| `paths.data_dir` | data | 数据目录 |
| `paths.checkpoints_dir` | checkpoints | 模型权重目录 |
| `paths.logs_dir` | logs | 日志目录 |
| `paths.figures_dir` | docs/figures | 图表输出目录 |
| `training.batch_size` | 16 | 批次大小 |
| `training.epochs` | 3 | 训练轮数 |
| `training.learning_rate` | 2e-5 | 学习率 |
| `training.lambda_weight` | 0.1 | 对比损失权重 |
| `training.temperature` | 0.05 | 温度参数 |
| `training.use_mini_dataset` | false | 是否使用mini数据集 |
| `training.use_contrastive` | true | 是否使用对比学习 |
| `training.contrastive_type` | supcon | 对比学习类型 (supcon/infonce) |
| `model.model_name` | bert-base-chinese | BERT模型名称 |
| `evaluation.model_name` | model_supcon | 默认评估的模型 |
| `visualization.dpi` | 300 | 图表分辨率 |

### 修改配置

直接编辑 `config.yaml` 文件即可，例如：

```yaml
# 使用完整数据集训练
training:
  use_mini_dataset: false
  epochs: 5
```

## 日志系统

### 日志目录结构

```
logs/
├── training_20260421_175858.log  # 训练日志（带时间戳）
├── evaluate_20260421_180015.log  # 评估日志（带时间戳）
└── .gitkeep
```

### 日志格式

每次运行都会生成带时间戳的日志文件，不会覆盖之前的日志。

训练日志格式：
```
[timestamp] [level] [module] 配置信息
[timestamp] [level] [module] Epoch 1/3
[timestamp] [level] [module] Step 10 | Total Loss: 0.5123 | CE Loss: 0.4567 | Con Loss: 0.0556
```

## 数据准备与处理

### 数据集来源

#### 中文数据集

| 数据集名称 | 描述 | 数据规模 | 数据格式 | 来源 |
|-----------|------|---------|---------|------|
| COLD | 最权威的中文攻击性语言数据集 | 3.7万条 (知乎、微博) | CSV | https://github.com/thu-coai/COLDataset |
| TOXICN | 针对中文语境下的毒性言论 | 约1.2万条 | JSON | scidb.cn |

#### 英文数据集

| 数据集名称 | 描述 | 数据规模 | 数据格式 | 来源 |
|-----------|------|---------|---------|------|
| Davidson | NLP领域的经典数据集 | 2.5万条 (Twitter) | CSV | Kaggle |

### 数据增强

核心思路：利用`jieba（中文）`和`nltk（英文）`分词，进行同义词替换后得到增强后的数据`text_aug`。

**脚本位置**: `scripts/data_intergation.py`

**实现**:
1. 词级别句子划分
2. 去除停用词，提取核心实词
3. 同义词替换：
   - 中文使用`synonyms`库
   - 英文使用`nltk`库

## 模型开发与训练

### 模型架构

本项目采用**对比学习**与**BERT 文本编码器**相结合的架构。

**脚本位置**: `models/model.py`

模型包含三个核心组件：
1. **文本编码器**：基于预训练的 BERT
2. **投影层**：768维 → 128维，用于对比学习
3. **分类头**：输出二分类的logits

### 对比学习策略

采用**有监督对比学习（SupCon）**策略：
- 正样本：原句-增强句对（同义词替换）
- 监督信号：将同类别的样本拉近，异类别推远

### 损失函数

联合损失公式：

$$
\mathcal{L} = \mathcal{L}_{\text{CE}} + \lambda \cdot \mathcal{L}_{\text{con}}$$

- $\mathcal{L}_{\text{CE}}$: 交叉熵损失（分类）
- $\mathcal{L}_{\text{con}}$: SupCon损失（对比学习）
- $\lambda$: 对比损失权重（默认0.1）

### 训练命令

```bash
# 查看帮助信息
python scripts/train.py --help

# 训练模型（使用配置文件中的参数）
python scripts/train.py

# 训练指定模型
python scripts/train.py --contrastive-type supcon --full

# 训练基线模型（无对比学习）
python scripts/train.py --no-contrastive --full

# 使用InfoNCE损失
python scripts/train.py --contrastive-type infonce --full
```

### 训练配置

通过修改 `config.yaml` 调整训练参数：

```yaml
training:
  batch_size: 16
  epochs: 5
  learning_rate: 2e-5
  lambda_weight: 0.1
  use_contrastive: true
  contrastive_type: supcon
  use_mini_dataset: false
```

## 端到端的检测系统

本项目提供了一个基于 **Flask** 的 Web 应用，允许用户通过浏览器实时检测文本是否为不当言论。

### 启动检测系统

1. **确保模型已训练**：检查 `checkpoints/model_*.pth` 是否存在

2. **启动Flask服务器**：

   ```bash
   python app.py
   ```

   服务器将在 `http://127.0.0.1:5000` 启动。

### 注意事项

- 首次启动时会从 Hugging Face 镜像下载 BERT 分词器和模型参数
- 默认使用 `bert-base-chinese` 模型，如需检测英文文本，请修改 `config.yaml` 并重新训练

## 实验可视化

项目提供完整的实验可视化工具，用于生成论文所需的图表。

### 可用图表

| 图表名称 | 说明 | 脚本 |
|---------|------|------|
| 训练曲线 | 展示训练过程中Loss的变化趋势 | `scripts/visualize_training.py` |
| 混淆矩阵 | 展示TP/FP/FN/TN的详细分布 | `scripts/visualize_confusion.py` |
| 性能对比 | 对比不同模型的性能指标 | `scripts/visualize_performance.py` |
| ROC曲线 | 展示模型的分类能力 | `scripts/visualize_roc.py` |
| t-SNE图 | 可视化特征空间的分布 | `scripts/visualize_tsne.py` |

### 生成图表

```bash
# 查看帮助信息
python scripts/generate_figures.py --help

# 生成所有图表（使用默认配置）
python scripts/generate_figures.py

# 指定结果目录
python scripts/generate_figures.py --results-dir checkpoints/evaluation_results

# 指定输出目录
python scripts/generate_figures.py --figures-dir docs/figures
```

### 图表输出位置

所有生成的图表默认保存在 `docs/figures/` 目录下，格式为PNG（300 DPI）。

## 评估命令

```bash
# 评估默认模型
python scripts/evaluate.py

# 评估指定模型
python scripts/evaluate.py --model model_supcon
python scripts/evaluate.py --model model_baseline
python scripts/evaluate.py --model model_infonce

# 不保存评估结果
python scripts/evaluate.py --no-save
```

### 评估结果

评估结果保存在 `checkpoints/evaluation_results/` 目录下，格式为JSON。

```json
{
  "model_name": "model_supcon",
  "timestamp": "20260421_175858",
  "labels": [0, 0, 1, ...],
  "preds": [0, 1, 1, ...],
  "probs": [0.12, 0.85, ...],
  "metrics": {
    "accuracy": 0.8113,
    "precision": 0.8147,
    "recall": 0.8129,
    "f1": 0.8138
  }
}
```

## 工具模块

### 日志模块

**位置**: `utils/logger.py`

提供统一的日志记录功能：
- 带时间戳的日志文件
- 支持不同日志级别（DEBUG, INFO, WARN, ERROR）
- 同时输出到文件和控制台
- 格式化的日志消息

使用方法：

```python
from utils.logger import get_logger

logger = get_logger('train')
logger.info("训练开始")
logger.log_config({'batch_size': 16})
logger.log_metrics(epoch=1, step=10, total_loss=0.5, ce_loss=0.4, con_loss=0.1)
```

### 配置模块

**位置**: `utils/config.py`

统一管理项目配置：
- 读取YAML配置文件
- 提供默认配置
- 路径管理（自动转换为绝对路径）
- 配置保存和打印

使用方法：

```python
from utils.config import get_config, init_config

# 读取配置
config = get_config()

# 获取项目根目录
project_root = config.get_project_root()

# 获取路径
data_dir = config.get_path('data_dir')
checkpoints_dir = config.get_path('checkpoints_dir')

# 初始化配置文件
init_config()
```

## 快速开始（完整流程）

```bash
# 1. 准备数据（如果需要）
# 数据已处理，跳过此步骤

# 2. 训练不同模型用于对比
python scripts/train.py --contrastive-type supcon --full
python scripts/train.py --contrastive-type infonce --full
python scripts/train.py --no-contrastive --full

# 3. 评估模型
python scripts/evaluate.py --model model_supcon
python scripts/evaluate.py --model model_infonce
python scripts/evaluate.py --model model_baseline

# 4. 生成所有实验图表
python scripts/generate_figures.py
```

## 项目特色

1. **配置驱动**：所有参数通过 `config.yaml` 统一管理，无需修改代码
2. **时间戳日志**：每次运行生成带时间戳的日志，避免覆盖
3. **统一路径**：自动将相对路径转换为绝对路径，支持从任意位置运行
4. **完整可视化**：提供训练曲线、混淆矩阵、ROC、t-SNE等全套图表
5. **结构清晰**：模块化设计，便于维护和扩展

## 目录说明

| 目录 | 说明 | Git管理 |
|------|------|---------|
| `data/` | 数据目录（保留目录结构） | 保留目录，忽略文件 |
| `checkpoints/` | 模型权重（保留目录结构） | 保留目录，忽略.pth文件 |
| `logs/` | 日志目录（保留目录结构） | 保留目录，忽略.log文件 |
| `docs/` | 文档目录（保留目录结构） | 保留目录，忽略图表文件 |
| `utils/` | 工具模块 | 完全管理 |
| `models/` | 模型定义 | 完全管理 |
| `scripts/` | 核心脚本 | 完全管理 |
| `templates/` | 模板文件 | 完全管理 |

## 依赖清单

详细依赖列表请查看 `requirements.txt`，主要包括：

- 深度学习：torch, transformers
- 数据处理：pandas, numpy
- 机器学习：scikit-learn
- 中文处理：jieba, synonyms
- 英文处理：nltk
- 可视化：matplotlib, seaborn
- Web服务：flask, flask-cors
- 配置管理：pyyaml

## 许可证

本项目仅用于学术研究和教育目的。