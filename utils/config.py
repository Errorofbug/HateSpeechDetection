"""
项目配置管理模块

使用YAML配置文件统一管理项目参数，避免代码中写死路径
"""
import os
import yaml


class Config:
    """配置管理类"""

    def __init__(self, config_path='config.yaml'):
        """
        初始化配置

        参数:
            config_path: 配置文件路径
        """
        self.config_path = config_path
        self.config = self._load_config()

    def _load_config(self):
        """加载配置文件"""
        if not os.path.exists(self.config_path):
            # 使用默认配置
            return self.get_default_config()

        with open(self.config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)

    def get_default_config(self):
        """获取默认配置"""
        return {
            # 项目路径配置
            'paths': {
                'data_dir': 'data',
                'data_raw_dir': 'data/raw',
                'data_processed_dir': 'data/processed',
                'checkpoints_dir': 'checkpoints',
                'logs_dir': 'logs',
                'figures_dir': 'docs',
                'utils_dir': 'utils',
                'models_dir': 'models',
                'scripts_dir': 'scripts'
            },

            # 训练配置
            'training': {
                'batch_size': 16,
                'epochs': 1,
                'learning_rate': 2e-5,
                'lambda_weight': 0.1,
                'temperature': 0.05,
                'projection_dim': 128,
                'dropout_rate': 0.3,
                'log_per_step': 10,
                'use_mini_dataset': True,  # 默认使用mini数据集快速测试
                'use_contrastive': True,
                'contrastive_type': 'supcon'  # 'supcon' or 'infonce'
            },

            # 模型配置
            'model': {
                'model_name': 'bert-base-chinese',
                'hidden_size': 768,
                'num_labels': 2,
                'max_seq_length': 128
            },

            # 评估配置
            'evaluation': {
                'batch_size': 16,
                'model_name': 'model_supcon',  # 默认评估的模型
                'save_results': True,
                'results_dir': 'checkpoints/evaluation_results'
            },

            # 可视化配置
            'visualization': {
                'figures_dir': 'docs/figures',
                'dpi': 300,
                'figure_format': 'png',
                'font_size': 12,
                'figure_style': 'seaborn-v0_8-whitegrid'
            },

            # 日志配置
            'logging': {
                'log_dir': 'logs',
                'log_to_file': True,
                'log_to_console': True,
                'log_level': 'INFO'  # DEBUG, INFO, WARN, ERROR
            }
        }

    def get(self, *keys, default=None):
        """
        获取配置项

        参数:
            *keys: 配置键路径，如 'paths', 'data_dir'
            default: 默认值

        返回:
            配置值
        """
        value = self.config
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        return value

    def get_path(self, path_key):
        """
        获取项目路径（自动转换为绝对路径）

        参数:
            path_key: 路径键名，如 'data_dir'

        返回:
            绝对路径
        """
        # 获取项目根目录
        project_root = self.get_project_root()

        # 获取相对路径
        rel_path = self.get('paths', path_key, default=path_key)

        # 转换为绝对路径
        return os.path.join(project_root, rel_path)

    def get_project_root(self):
        """获取项目根目录"""
        # 如果config_path是相对路径，则基于脚本位置计算
        if not os.path.isabs(self.config_path):
            # config.yaml在项目根目录
            return os.path.dirname(os.path.abspath(self.config_path))
        else:
            # 如果是绝对路径，则向上查找
            config_dir = os.path.dirname(self.config_path)
            return config_dir

    def save(self):
        """保存配置到文件"""
        # 获取默认配置作为模板
        default_config = self.get_default_config()

        # 创建配置目录
        os.makedirs(os.path.dirname(self.config_path), exist_ok=True)

        with open(self.config_path, 'w', encoding='utf-8') as f:
            yaml.dump(self.config, f, allow_unicode=True, default_flow_style=False, sort_keys=False)

    def print_config(self):
        """打印当前配置"""
        print("=" * 60)
        print("当前配置:")
        print("=" * 60)
        for section, values in self.config.items():
            print(f"\n[{section}]")
            for key, value in values.items():
                print(f"  {key}: {value}")
        print("\n" + "=" * 60)


# 全局配置实例
_config = None


def get_config(config_path='config.yaml'):
    """
    获取全局配置实例

    参数:
        config_path: 配置文件路径

    返回:
        Config实例
    """
    global _config
    if _config is None:
        _config = Config(config_path)
    return _config


def init_config(config_path='config.yaml'):
    """
    初始化配置文件

    参数:
        config_path: 配置文件路径
    """
    global _config
    _config = Config(config_path)
    _config.save()
    print(f"配置文件已创建: {config_path}")
    return _config


# 全局配置实例（可直接导入使用）
config = get_config()