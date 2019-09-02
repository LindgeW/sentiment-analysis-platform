import logging
import sys
'''
自定义日志模块
'''
class MyLogger():
    #日志级别关系映射
    level_dict = {
        'debug': logging.DEBUG,
        'info': logging.INFO,
        'warning': logging.WARNING,
        'error': logging.ERROR,
        'crit': logging.CRITICAL
    }

    #日志信息输出格式
    LOG_FMT = "%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s"
    #日期显示格式
    DATE_FMT = "%m/%d/%Y %H:%M:%S %p"

    def __init__(self, filename='all.log', level='info', log_fmt=LOG_FMT, date_fmt=DATE_FMT):
        #设置日志输出格式
        fmt = logging.Formatter(fmt=log_fmt, datefmt=date_fmt)
        #创建一个名为filename的日志器
        self.logger = logging.getLogger(filename)
        #设置日志级别
        self.logger.setLevel(self.level_dict[level])
        #获取控制台输出的处理器
        console_handler = logging.StreamHandler(sys.stdout) #默认是sys.stderr
        #设置控制台处理器的等级为DEBUG
        console_handler.setLevel(self.level_dict['info'])
        #设置控制台输出日志的格式
        console_handler.setFormatter(fmt)

        #获取文件输出的处理器
        file_handler = logging.FileHandler(filename)
        #设置文件输出处理器的等级为INFO
        file_handler.setLevel(self.level_dict['debug'])
        #设置文件输出日志的格式
        file_handler.setFormatter(fmt)

        self.logger.addHandler(console_handler)
        self.logger.addHandler(file_handler)

    def debug(self, msg):
        self.logger.debug(msg)

    def info(self, msg):
        self.logger.info(msg)

    def warning(self, msg):
        self.logger.warning(msg)

    def error(self, msg):
        self.logger.error(msg)

    def critical(self, msg):
        self.logger.critical(msg)
