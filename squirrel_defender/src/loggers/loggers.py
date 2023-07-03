import logging
import sys


class Logger:
    def __init__(self, logfile: str):
        """logger class to log  msg to both stdout and a specified logfile

        Args:
            logfile (str): which file to the log the msgs
        """
        self.logfile = logfile
        self.logger = logging.getLogger("main")
        self.logger.setLevel(logging.INFO)

        file_handler = logging.FileHandler(self.logfile, "w")
        formatter = logging.Formatter(
            "[%(asctime)s;%(levelname)s]    %(message)s", "%Y-%m-%d %H:%M:%S"
        )

        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

        stream_handle = logging.StreamHandler(sys.stdout)
        stream_handle.setFormatter(formatter)
        self.logger.addHandler(stream_handle)
        self.logger.propagate = False

    def info(self, msg: str) -> None:
        """for logging the msg to stdout and file

        Args:
            msg (str): msg to log
        """
        self.logger.info(msg)
