import logging
import sys
from logging.handlers import TimedRotatingFileHandler


class BOLogFormatter(logging.Formatter):
    red = '\033[91m'
    green = '\033[92m'
    blue = '\033[94m'
    grey = "\x1b[38;20m"
    yellow = "\x1b[33;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    FORMATS = {
        'client': green + format + reset,
    }

    def __init__(self, logger_name: str):
        self.logger_name = logger_name
        logging.Formatter().__init__()

    def format(self, record):
        log_fmt = self.FORMATS.get(self.logger_name)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


def get_console_handler(logger_name: str) -> logging.StreamHandler:
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(BOLogFormatter(logger_name))
    return console_handler


def get_file_handler(logger_name: str, file_name: str) -> TimedRotatingFileHandler:
    file_handler = TimedRotatingFileHandler(file_name, when='midnight')
    file_handler.setFormatter(BOLogFormatter(logger_name))
    return file_handler


def get_logger(logger_name: str, file_name: str) -> logging.Logger:
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(get_console_handler(logger_name))
    logger.addHandler(get_file_handler(logger_name, file_name))
    logger.propagate = False
    return logger