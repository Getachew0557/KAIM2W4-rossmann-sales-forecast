import os
import logging

# Ensure the 'logs' directory exists
log_dir = os.path.join(os.path.dirname(__file__), '../logs')
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# Set up logging
log_file_info = os.path.join(log_dir, 'info.log')
log_file_error = os.path.join(log_dir, 'error.log')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

info_logger = logging.getLogger('info')
info_handler = logging.FileHandler(log_file_info)
info_handler.setLevel(logging.INFO)
info_logger.addHandler(info_handler)

error_logger = logging.getLogger('error')
error_handler = logging.FileHandler(log_file_error)
error_handler.setLevel(logging.ERROR)
error_logger.addHandler(error_handler)
