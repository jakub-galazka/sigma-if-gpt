import os
import logging
import datetime


class Logger:

    _LOG_FILES_DIR = "logs"
    _LOG_FILENAME_TIME_FORMAT = "%Y-%m-%d_%H-%M-%S"
    _LOGGING_FORMAT = "%(asctime)s | %(levelname)s | %(message)s"
    _LOGGING_TIME_FORMAT = "%Y-%m-%d %H:%M:%S"

    def __init__(self) -> None:
        os.makedirs(Logger._LOG_FILES_DIR, exist_ok=True)
        timestamp = datetime.datetime.now().strftime(Logger._LOG_FILENAME_TIME_FORMAT)
        log_filepath = os.path.join(Logger._LOG_FILES_DIR, f"log_{timestamp}.txt")

        # Handlers
        file_handler = logging.FileHandler(filename=log_filepath, encoding="utf-8")
        file_handler.setLevel(logging.INFO)

        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        # Formatter
        formatter = logging.Formatter(Logger._LOGGING_FORMAT, Logger._LOGGING_TIME_FORMAT)
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        # Logger
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
