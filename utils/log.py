import logging, inspect
from colorlog import ColoredFormatter

from mode.singleton import Singleton


# 日志封装类
@Singleton
class Log():

    # 创建logger
    def __init__(self) -> None:
        '''
        设置默认DEBUG模式,DEBUG/INFO/WARNING/ERROR/CRITICAL级别的日志均可打印
        basicConfig和Formatter两者选其一即可
        logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
        '''
        logger = logging.getLogger(__name__)

        # 创建一个StreamHandler并设置格式
        handler = logging.StreamHandler()
        formatter = ColoredFormatter(
            # "%(log_color)s%(asctime)s - %(levelname)-8s%(reset)s %(white)s%(message)s - [%(filename)s:%(lineno)d]",
            "%(log_color)s%(asctime)s - %(levelname)-8s%(reset)s %(white)s%(message)s",
            datefmt='%Y-%m-%d %H:%M:%S',  # 添加日期和时间的格式
            reset=True,
            log_colors={
                'DEBUG': 'cyan',
                'INFO': 'green',
                'WARNING': 'yellow',
                'ERROR': 'red',
                'CRITICAL': 'red,bg_white',
            },
            secondary_log_colors={},
            style='%'
        )

        handler.setFormatter(formatter)

        # 将handler添加到logger中
        logger.addHandler(handler)

        self.logger = logger
        self.InitLevel()


    # 通过读取Yml配置文件设置日志等级
    def InitLevel(self):
        # 获取配置中的日志级别字符串
        LoggerLevelStr = 'INFO'

        self.SetLevel(level=LoggerLevelStr)


    # 设置日志打印级别，默认传入DEBUG
    def SetLevel(self, level='debug') -> None:
        level = level.lower() # 化为小写
        LoggerLevelMapping = {
            'debug': logging.DEBUG,
            'info': logging.INFO,
            'warning': logging.WARNING,
            'error': logging.ERROR,
            'critical': logging.CRITICAL
        }

        logger_level = LoggerLevelMapping.get(level, None)

        if logger_level is None:
            self.logger.warning('The log level of "{0}" does not exist. It has been automatically set to DEBUG mode'.format(level))
            self.logger.setLevel(level=logging.DEBUG)
        else:
            self.logger.setLevel(level=logger_level)


    def _get_caller_info(self):
        # 使用inspect模块获取调用者的信息
        frame_info = inspect.getframeinfo(inspect.stack()[2][0])
        return frame_info.filename, frame_info.lineno


    # 打印DEBUG级别的日志信息
    def Debug(self, content) -> None:
        caller_filename, caller_lineno = self._get_caller_info()
        self.logger.debug(f"{content} - [Called from: {caller_filename}:{caller_lineno}]")


    # 打印INFO级别的日志信息
    def Info(self, content) -> None:
        caller_filename, caller_lineno = self._get_caller_info()
        self.logger.info(f"{content} - [Called from: {caller_filename}:{caller_lineno}]")


    # 打印WARNING级别的日志信息
    def Warn(self, content) -> None:
        caller_filename, caller_lineno = self._get_caller_info()
        self.logger.warning(f"{content} - [Called from: {caller_filename}:{caller_lineno}]")


    # 打印ERROR级别的日志信息
    def Error(self, content) -> None:
        caller_filename, caller_lineno = self._get_caller_info()
        self.logger.error(f"{content} - [Called from: {caller_filename}:{caller_lineno}]")


    # 打印CRITICAL级别的日志信息
    def Critical(self, content) -> None:
        caller_filename, caller_lineno = self._get_caller_info()
        self.logger.critical(f"{content} - [Called from: {caller_filename}:{caller_lineno}]")
