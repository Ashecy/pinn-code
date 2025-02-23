import sys
import psutil
import time
import torch


# -------------------------------
# 定义同时输出到控制台和文件的 Logger 类
# -------------------------------
class Logger(object):
    def __init__(self, filename="log.txt"):
        self.terminal = sys.stdout
        self.log = open(filename, "w")  # 每次运行覆盖旧文件

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()


def log_system_info(device, start_time, iteration=None, label=None):
    """
    打印当前的系统监控信息，包括 CPU 内存使用、GPU 内存占用（如果可用）及经过时间。
    参数:
      device: 当前使用的 torch.device
      start_time: 程序开始的时间（time.time()）
      iteration: 当前迭代次数（可选）
      label: 日志标签（可选，例如 "Adam Final"）
    """
    process = psutil.Process()
    memory_usage = process.memory_info().rss / (1024 ** 2)
    elapsed_time = time.time() - start_time
    info = f"Memory Usage -> CPU: {memory_usage:.2f} MB"
    if torch.cuda.is_available():
        gpu_memory_allocated = torch.cuda.memory_allocated(device) / (1024 ** 2)
        gpu_memory_reserved = torch.cuda.memory_reserved(device) / (1024 ** 2)
        info += f", GPU Allocated: {gpu_memory_allocated:.2f} MB, GPU Reserved: {gpu_memory_reserved:.2f} MB"
    info += f", Elapsed Time: {elapsed_time:.2f} seconds"
    # if iteration is not None:
    #    info = f"Iteration: {iteration}. " + info
    if label is not None:
        info = f"{label}: " + info
    print(info)
