import logging
import sys

def setup_logging(level=logging.INFO):
    """Configures basic logging.

    Args:
        level: The logging level (e.g., logging.INFO, logging.DEBUG).

    Returns:
        The configured logger instance.
    """
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=level, format=log_format, stream=sys.stdout)
    logger = logging.getLogger(__name__) # Get logger for this module
    # If you want a specific logger name across the app, use:
    # logger = logging.getLogger('conclave_simulation')
    return logger
