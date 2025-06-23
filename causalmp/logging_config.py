import logging

def configure_logging(level=logging.WARNING, log_file=None, log_format=None):
    """Configure logging for the package."""
    if isinstance(level, str):
        level = getattr(logging, level.upper())
    
    format_str = log_format or '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Configure root logger
    logging.basicConfig(
        level=level,
        format=format_str,
        filename=log_file
    )
    
    # Configure package logger
    logger = logging.getLogger('causalmp')
    logger.setLevel(level)
    
    # If a file is specified, add a file handler
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter(format_str))
        logger.addHandler(file_handler)
    
    # Prevent propagation to root logger to avoid duplicate logs
    logger.propagate = False
    
    return logger 