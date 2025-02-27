import logging
from logging.handlers import RotatingFileHandler
import os

log_file = 'image_search.log'
log_dir = 'src/logs'
log_level=logging.INFO

def get_logger( ):

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    log_file_path = os.path.join(log_dir, log_file)

    logger = logging.getLogger(__name__)

    if not logger.hasHandlers():
        logger.setLevel(log_level)

        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG) 

        file_handler = RotatingFileHandler(log_file_path, maxBytes=5*1024*1024, backupCount=3)
        file_handler.setLevel(logging.INFO)

        log_format = '%(asctime)s - %(levelname)s - %(message)s'
        formatter = logging.Formatter(log_format, datefmt='%Y-%m-%d %H:%M')
        console_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)

        logger.addHandler(console_handler)
        logger.addHandler(file_handler)

    return logger