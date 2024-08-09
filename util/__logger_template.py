"""
import logging
formatter = logging.Formatter(fmt=f'[%(asctime)s|%(name)s] %(levelname)s - %(message)s',
                              datefmt='%m/%d/%Y %I:%M:%S')
console_handler = logging.StreamHandler()
log = logging.getLogger(__name__)
log.addHandler(console_handler)
console_handler.setFormatter(formatter)
"""

"""
import logging

fpath_logging_file = <fpath_log>

formatter = logging.Formatter(fmt=f'[%(asctime)s|%(name)s] %(levelname)s - %(message)s',
                              datefmt='%m/%d/%Y %I:%M:%S')
console_handler = logging.StreamHandler()
file_handler = logging.FileHandler(f"{fpath_logging_file}", mode="a", encoding="utf-8")
log = logging.getLogger(__name__)
log.addHandler(console_handler)
log.addHandler(file_handler)
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

"""