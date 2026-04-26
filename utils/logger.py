"""
项目日志模块

提供统一的日志记录功能，支持时间戳日志、文件输出和不同级别

使用方式:
    from utils.logger import logger, train_logger
"""
import os
import sys
from datetime import datetime


class Logger:
    """统一的日志记录器"""

    def __init__(self, name, log_dir='logs', log_to_file=True, log_to_console=True, enable_csv=False):
        """
        初始化日志器

        参数:
            name: 日志名称（通常为模块名）
            log_dir: 日志文件保存目录
            log_to_file: 是否输出到文件
            log_to_console: 是否输出到控制台
            enable_csv: 是否创建CSV格式的loss日志文件（仅train_logger需要）
        """
        self.name = name
        self.log_to_file = log_to_file
        self.log_to_console = log_to_console
        self.log_dir = log_dir
        self.enable_csv = enable_csv

        # 创建日志目录
        if log_to_file:
            os.makedirs(log_dir, exist_ok=True)

        # 当前日志文件（带时间戳）
        self.current_log_file = None
        # 当前loss日志文件（用于绘图，CSV格式）
        self.current_loss_file = None

    def get_log_filename(self):
        """生成日志文件名"""
        return os.path.join(self.log_dir, f'{self.name}.log')

    def set_log_file(self, log_file=None):
        """
        设置当前日志文件

        同时创建loss日志文件（CSV格式，用于绘图）
        """
        if log_file is None:
            log_file = self.get_log_filename()
        self.current_log_file = log_file

        if self.log_to_file:
            with open(log_file, 'a', encoding='utf-8') as f:  # 追加模式，避免覆盖
                f.write(f"# Log created at: {datetime.now()}\n")
                f.write("=" * 80 + "\n")

            # 仅当启用CSV时才创建loss日志文件（用于绘图）
            if self.enable_csv:
                loss_log_file = log_file.replace('.log', '_loss.csv')
                self.current_loss_file = loss_log_file

                # 写入CSV表头
                with open(loss_log_file, 'w', encoding='utf-8') as f:
                    f.write("epoch,step,total_loss,ce_loss,con_loss\n")

    def _format_message(self, level, message):
        """格式化日志消息"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        return f"[{timestamp}] [{level:8}] [{self.name:12}] {message}"

    def info(self, message):
        """输出信息级别日志"""
        formatted = self._format_message('INFO', message)
        if self.log_to_console:
            print(formatted)
        if self.log_to_file and self.current_log_file:
            with open(self.current_log_file, 'a', encoding='utf-8') as f:
                f.write(formatted + '\n')

    def warning(self, message):
        """输出警告级别日志"""
        formatted = self._format_message('WARN', message)
        if self.log_to_console:
            print(formatted)
        if self.log_to_file and self.current_log_file:
            with open(self.current_log_file, 'a', encoding='utf-8') as f:
                f.write(formatted + '\n')

    def error(self, message):
        """输出错误级别日志"""
        formatted = self._format_message('ERROR', message)
        if self.log_to_console:
            print(formatted, file=sys.stderr)
        if self.log_to_file and self.current_log_file:
            with open(self.current_log_file, 'a', encoding='utf-8') as f:
                f.write(formatted + '\n')

    def debug(self, message):
        """输出调试级别日志"""
        formatted = self._format_message('DEBUG', message)
        if self.log_to_console:
            print(formatted)
        if self.log_to_file and self.current_log_file:
            with open(self.current_log_file, 'a', encoding='utf-8') as f:
                f.write(formatted + '\n')

    def log_config(self, config_dict):
        """记录配置信息"""
        self.info("Configuration:")
        for key, value in config_dict.items():
            self.info(f"  {key}: {value}")
        self.info("=" * 80)

    def log_metrics(self, epoch, step, total_loss, ce_loss=None, con_loss=None):
        """
        记录训练指标

        参数:
            epoch: 当前epoch
            step: 当前step
            total_loss: 总损失
            ce_loss: 分类损失
            con_loss: 对比损失
        """
        # 构造清晰的日志行
        parts = [f"Epoch {epoch}", f"Step {step}"]
        parts.append(f"Total Loss: {total_loss:.6f}")

        if ce_loss is not None:
            parts.append(f"CE Loss: {ce_loss:.6f}")
        if con_loss is not None:
            parts.append(f"Con Loss: {con_loss:.6f}")

        message = " | ".join(parts)
        self.info(message)

    def log_metrics_for_plot(self, epoch, step, total_loss, ce_loss=None, con_loss=None):
        """
        记录训练指标到专门的绘图日志（CSV格式）

        参数:
            epoch: 当前epoch
            step: 当前step
            total_loss: 总损失
            ce_loss: 分类损失
            con_loss: 对比损失
        """
        if not self.log_to_file or self.current_loss_file is None:
            return

        # CSV格式：epoch,step,total_loss,ce_loss,con_loss
        line = f"{epoch},{step},{total_loss:.6f}"
        if ce_loss is not None:
            line += f",{ce_loss:.6f}"
        else:
            line += ","
        if con_loss is not None:
            line += f",{con_loss:.6f}"
        else:
            line += ","
        line += "\n"

        with open(self.current_loss_file, 'a', encoding='utf-8') as f:
            f.write(line)

    def log_evaluation_results(self, metrics):
        """
        记录评估结果

        参数:
            metrics: 指标字典
        """
        self.info("Evaluation Results:")
        self.info("=" * 80)
        for key, value in metrics.items():
            self.info(f"  {key:20}: {value:.6f}" if isinstance(value, float) else f"  {key:20}: {value}")
        self.info("=" * 80)


# 预定义的日志器
def get_logger(name, log_dir='logs'):
    """获取日志器实例"""
    return Logger(name, log_dir=log_dir)


# 全局日志器实例（由 __init__.py 初始化）
logger = None
train_logger = None