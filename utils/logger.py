import logging
import sys
from logging.handlers import RotatingFileHandler

class ColorFormatter(logging.Formatter):
    COLORS = {
        logging.DEBUG: "\033[37m",   # White
        logging.INFO: "\033[36m",    # Cyan
        logging.WARNING: "\033[33m", # Yellow
        logging.ERROR: "\033[31m",   # Red
        logging.CRITICAL: "\033[41m" # Red background
    }
    RESET = "\033[0m"

    def format(self, record):
        log_color = self.COLORS.get(record.levelno, self.RESET)
        message = super().format(record)
        return f"{log_color}{message}{self.RESET}"

# Create main logger
logger = logging.getLogger("video_qa")
logger.setLevel(logging.DEBUG)

# Formatter
formatter = ColorFormatter("[%(asctime)s] [%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S")

# Console handler (color output)
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.DEBUG)
console_handler.setFormatter(formatter)

# Rotating file handler (logs stored for debugging)
file_handler = RotatingFileHandler("video_qa.log", maxBytes=2_000_000, backupCount=5)
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s"))

# Attach handlers
logger.addHandler(console_handler)
logger.addHandler(file_handler)

# Example log messages for testing
if __name__ == "__main__":
    logger.debug("Debug message for developers.")
    logger.info("System initialized successfully.")
    logger.warning("Low confidence detection detected.")
    logger.error("Frame processing error.")
    logger.critical("System crash imminent.")
