# logger_config.py
import logging

# 设置日志
logging_config = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {
            "format": "%(asctime)s - %(levelname)s - %(message)s",
        },
    },
    "handlers": {
        "default": {
            "level": "INFO",
            "class": "logging.StreamHandler",
            "formatter": "default",
        },
    },
    "root": {
        "handlers": ["default"],
        "level": "INFO",
    },
}

logging.config.dictConfig(logging_config)
logger = logging.getLogger(__name__)

