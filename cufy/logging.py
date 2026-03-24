import logging

logger: logging.Logger = logging.getLogger("cufy")
logger.addHandler(logging.NullHandler())
