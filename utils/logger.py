import os
import sys
import time
import logging
import shutil
from datetime import datetime
from typing import Optional, Dict, Any, List, Union


class ProgressBar:
    """
    终端进度条显示类，自适应终端窗口宽度
    """
    # 保存当前活动的进度条实例，用于全局访问
    _active_instance = None
    
    def __init__(self, total: int, prefix: str = '', suffix: str = '', decimals: int = 1,
                 length: int = None, fill: str = '█', print_end: str = '\r'):
        """
        初始化进度条
        
        Args:
            total: 总迭代次数
            prefix: 前缀字符串
            suffix: 后缀字符串
            decimals: 百分比小数位数
            length: 进度条长度，如果为None则自动根据终端宽度调整
            fill: 进度条填充字符
            print_end: 打印结束字符
        """
        self.total = total
        self.prefix = prefix
        self.suffix = suffix
        self.decimals = decimals
        self.fill = fill
        self.print_end = print_end
        self.iteration = 0
        self.start_time = time.time()
        self.batch_info = ""
        self.last_output = ""
        self.visible = True
        
        # 设置为当前活动实例
        ProgressBar._active_instance = self
        
        # 获取终端宽度并设置进度条长度
        self._set_bar_length(length)
        self._print_progress()
    
    def update(self, iteration: Optional[int] = None, suffix: Optional[str] = None, batch_info: Optional[str] = None):
        """
        更新进度条
        
        Args:
            iteration: 当前迭代次数，如果为None则自增1
            suffix: 更新后缀信息
            batch_info: 批次信息，例如当前批次/总批次
        """
        if iteration is not None:
            self.iteration = iteration
        else:
            self.iteration += 1
            
        if suffix is not None:
            self.suffix = suffix
            
        if batch_info is not None:
            self.batch_info = batch_info
        
        # 设置为当前活动实例（确保在更新时重新激活）
        ProgressBar._active_instance = self
        # 确保进度条可见
        self.visible = True
        # 更新进度条显示
        self._print_progress()
    
    def _get_terminal_width(self) -> int:
        """
        获取终端窗口宽度
        
        Returns:
            终端窗口宽度，如果无法获取则返回默认值80
        """
        try:
            # 使用shutil获取终端宽度
            terminal_width = shutil.get_terminal_size().columns
            return max(terminal_width, 40)  # 确保最小宽度为40
        except Exception:
            # 如果无法获取终端宽度，返回默认值
            return 80
    
    def _set_bar_length(self, length: Optional[int] = None):
        """
        设置进度条长度，根据终端宽度自适应调整
        
        Args:
            length: 指定的进度条长度，如果为None则自动计算
        """
        if length is not None:
            self.length = length
            return
            
        # 获取终端宽度
        terminal_width = self._get_terminal_width()
        
        # 计算其他元素占用的宽度
        # 格式: [前缀] |进度条| 百分比% [后缀] [批次信息] [时间信息]
        other_elements_width = len(self.prefix) + 3 + 7 + len(self.suffix) + 20  # 估计值
        
        # 计算可用于进度条的宽度
        available_width = terminal_width - other_elements_width
        
        # 设置进度条长度，确保在合理范围内
        self.length = max(10, min(available_width, 100))
    
    def _print_progress(self):
        """
        打印进度条
        """
        # 如果进度条不可见，则不打印
        if not self.visible:
            return
            
        # 重新检查终端宽度并调整进度条长度
        self._set_bar_length()
        
        percent = ("{0:." + str(self.decimals) + "f}").format(100 * (self.iteration / float(self.total)))
        filled_length = int(self.length * self.iteration // self.total)
        bar = self.fill * filled_length + '-' * (self.length - filled_length)
        
        # 计算剩余时间
        elapsed_time = time.time() - self.start_time
        if self.iteration > 0:
            eta = elapsed_time * (self.total / self.iteration - 1)
            time_info = f"| ETA: {self._format_time(eta)} | 用时: {self._format_time(elapsed_time)}"
        else:
            time_info = ""
            
        # 批次信息
        batch_display = f"| {self.batch_info}" if self.batch_info else ""
        
        # 根据终端宽度动态调整显示内容
        terminal_width = self._get_terminal_width()
        output = f'\r{self.prefix} |{bar}| {percent}% {self.suffix} {batch_display} {time_info}'
        
        # 如果输出内容超过终端宽度，进行裁剪
        if len(output) > terminal_width:
            # 优先保留进度条和百分比，裁剪其他信息
            base_output = f'\r{self.prefix} |{bar}| {percent}%'
            remaining_width = terminal_width - len(base_output)
            
            if remaining_width > 10:  # 确保有足够空间显示其他信息
                # 按优先级添加其他信息
                if self.suffix and remaining_width > len(self.suffix) + 1:
                    base_output += f' {self.suffix}'
                    remaining_width -= (len(self.suffix) + 1)
                
                # 添加批次信息
                if batch_display and remaining_width > len(batch_display) + 1:
                    base_output += f' {batch_display}'
                    remaining_width -= (len(batch_display) + 1)
                
                # 添加时间信息
                if time_info and remaining_width > len(time_info) + 1:
                    base_output += f' {time_info}'
            
            output = base_output
        
        # 保存最后的输出，用于恢复进度条
        self.last_output = output
            
        # 打印进度条 - 确保在Windows环境下也能正确显示
        print(output, end='', flush=True)
        
        # 如果完成则换行
        if self.iteration == self.total:
            print('\n', end='', flush=True)
            ProgressBar._active_instance = None  # 完成后清除活动实例
    
    def clear(self):
        """
        清除进度条显示
        """
        if self.visible:
            # 计算需要清除的字符数
            terminal_width = self._get_terminal_width()
            # 打印足够多的空格来覆盖整行，然后回到行首
            print('\r' + ' ' * terminal_width + '\r', end='', flush=True)
            self.visible = False
    
    def restore(self):
        """
        恢复进度条显示
        """
        if not self.visible and self.last_output:
            print(self.last_output, end='', flush=True)
            self.visible = True
    
    @classmethod
    def clear_current(cls):
        """
        清除当前活动的进度条（如果有）
        """
        if cls._active_instance is not None:
            cls._active_instance.clear()
    
    @classmethod
    def restore_current(cls):
        """
        恢复当前活动的进度条（如果有）
        """
        if cls._active_instance is not None:
            cls._active_instance.restore()
    
    def _format_time(self, seconds: float) -> str:
        """
        格式化时间显示
        
        Args:
            seconds: 秒数
            
        Returns:
            格式化后的时间字符串
        """
        m, s = divmod(int(seconds), 60)
        h, m = divmod(m, 60)
        return f"{h:02d}:{m:02d}:{s:02d}" if h > 0 else f"{m:02d}:{s:02d}"


class Logger:
    """
    日志记录类，支持终端输出和文件保存
    """
    # 日志级别映射
    LEVEL_MAP = {
        'debug': logging.DEBUG,
        'info': logging.INFO,
        'warning': logging.WARNING,
        'error': logging.ERROR,
        'critical': logging.CRITICAL
    }
    
    def __init__(self, name: str = 'TF', log_dir: str = 'logs', 
                 console_level: str = 'info', file_level: str = 'info',
                 log_file: Optional[str] = None):
        """
        初始化日志记录器
        
        Args:
            name: 日志记录器名称
            log_dir: 日志文件保存目录
            console_level: 控制台日志级别
            file_level: 文件日志级别
            log_file: 日志文件名，如果为None则自动生成
        """
        self.name = name
        self.log_dir = log_dir
        
        # 创建日志目录
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
            
        # 设置日志文件名
        if log_file is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            log_file = f"{name}_{timestamp}.log"
        self.log_file = os.path.join(log_dir, log_file)
        
        # 创建日志记录器
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)  # 设置为最低级别，具体过滤由handler决定
        self.logger.propagate = False  # 避免日志重复输出
        
        # 清除已有的处理器
        if self.logger.handlers:
            self.logger.handlers.clear()
            
        # 创建自定义的控制台处理器，用于处理进度条和日志的协调显示
        self._create_console_handler(console_level)
        
        # 创建文件处理器
        file_handler = logging.FileHandler(self.log_file, encoding='utf-8')
        file_handler.setLevel(self.LEVEL_MAP.get(file_level.lower(), logging.INFO))
        file_formatter = logging.Formatter(
            '%(asctime)s [%(levelname)s] - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        self.logger.addHandler(file_handler)
        
        # 记录初始化信息
        self.info(f"日志系统初始化完成，日志文件：{self.log_file}")
    
    def _create_console_handler(self, console_level: str):
        """
        创建自定义的控制台处理器，处理进度条和日志的协调显示
        
        Args:
            console_level: 控制台日志级别
        """
        class ProgressBarStreamHandler(logging.StreamHandler):
            """
            自定义的StreamHandler，在输出日志前清除进度条，输出后恢复进度条
            """
            def emit(self, record):
                # 在输出日志前清除当前进度条
                ProgressBar.clear_current()
                
                # 调用父类的emit方法输出日志
                super().emit(record)
                
                # 输出日志后恢复进度条
                ProgressBar.restore_current()
        
        # 使用自定义的处理器
        console_handler = ProgressBarStreamHandler()
        console_handler.setLevel(self.LEVEL_MAP.get(console_level.lower(), logging.INFO))
        console_formatter = logging.Formatter(
            '\n%(asctime)s [%(levelname)s] - %(message)s',  # 添加换行符确保日志从新行开始
            datefmt='%H:%M:%S'
        )
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)
    
    def debug(self, message: str):
        """记录调试信息"""
        self.logger.debug(message)
    
    def info(self, message: str):
        """记录一般信息"""
        self.logger.info(message)
    
    def warning(self, message: str):
        """记录警告信息"""
        self.logger.warning(message)
    
    def error(self, message: str):
        """记录错误信息"""
        self.logger.error(message)
    
    def critical(self, message: str):
        """记录严重错误信息"""
        self.logger.critical(message)
    
    def log_metrics(self, metrics: Dict[str, Any], prefix: str = ''):
        """
        记录指标信息
        
        Args:
            metrics: 指标字典
            prefix: 指标前缀
        """
        prefix = f"{prefix} " if prefix else ""
        message = prefix + " | ".join([f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}" for k, v in metrics.items()])
        self.info(message)
    
    def create_progress_bar(self, total: int, prefix: str = '', suffix: str = '', **kwargs) -> ProgressBar:
        """
        创建进度条
        
        Args:
            total: 总迭代次数
            prefix: 前缀字符串
            suffix: 后缀字符串
            **kwargs: 其他进度条参数
            
        Returns:
            进度条对象
        """
        return ProgressBar(total, prefix, suffix, **kwargs)
        
    def pause_progress_bar(self):
        """
        暂停当前活动的进度条显示
        用于在需要输出其他信息时临时清除进度条
        """
        ProgressBar.clear_current()
        
    def resume_progress_bar(self):
        """
        恢复当前活动的进度条显示
        用于在输出其他信息后恢复进度条
        """
        ProgressBar.restore_current()


# 创建全局日志记录器实例
def create_logger(name: str = 'TF', log_dir: str = 'logs', **kwargs) -> Logger:
    """
    创建日志记录器
    
    Args:
        name: 日志记录器名称
        log_dir: 日志文件保存目录
        **kwargs: 其他日志记录器参数
        
    Returns:
        日志记录器对象
    """
    return Logger(name, log_dir, **kwargs)

def ensure_log_dirs():
    """
    确保日志目录存在
    """
    # 创建预训练日志目录
    pretrain_log_dir = 'checkpoint/pretrain/logs'
    os.makedirs(pretrain_log_dir, exist_ok=True)
    
    # 创建微调日志目录
    finetune_log_dir = 'checkpoint/finetune/logs'
    os.makedirs(finetune_log_dir, exist_ok=True)
    
    # 创建其他可能需要的目录
    os.makedirs('checkpoint/pretrain', exist_ok=True)
    os.makedirs('checkpoint/finetune', exist_ok=True)