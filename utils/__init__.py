"""
工具模块

提供日志、配置等通用功能

使用方式:
    from utils import logger, train_logger, config
"""
from .config import Config, get_config, init_config, config
from .logger import Logger, get_logger, logger as _logger_placeholder, train_logger as _train_logger_placeholder

# 初始化全局 logger
log_dir = config.get_path('logs_dir')

# 通用日志器（不创建CSV文件）
logger = Logger('hate_speech', log_dir=log_dir, enable_csv=False)
logger.set_log_file()

# 训练专用日志器（创建CSV文件用于绘图）
train_logger = Logger('train', log_dir=log_dir, enable_csv=True)
# train_logger 的日志文件在训练开始时设置

__all__ = [
    'Logger', 'get_logger', 'logger', 'train_logger',
    'Config', 'get_config', 'init_config', 'config'
]