import sys
from loguru import logger
from pathlib import Path
import os
from datetime import datetime

def setup_logging():
    # Create logs directory if it doesn't exist
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    # Generate log filename with timestamp
    log_filename = f"relationship_assessment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    log_path = log_dir / log_filename

    # Configure loguru
    config = {
        "handlers": [
            {
                "sink": sys.stdout,
                "format": "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
                         "<level>{level: <8}</level> | "
                         "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
                         "<level>{message}</level>",
                "level": "INFO",
            },
            {
                "sink": str(log_path),
                "format": "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | "
                         "{name}:{function}:{line} | {message}",
                "level": "DEBUG",
                "rotation": "500 MB",
                "retention": "10 days",
            },
        ],
    }

    # Remove default logger and apply new configuration
    logger.remove()
    for handler in config["handlers"]:
        logger.add(**handler)

    return logger

def log_agent_action(agent_id: str, action: str, details: dict):
    logger.info(f"Agent {agent_id} performing {action}: {details}")

def log_error(error: Exception, context: dict = None):
    if context is None:
        context = {}
    logger.error(f"Error: {str(error)}, Context: {context}")
    logger.exception(error)

def log_api_request(method: str, endpoint: str, params: dict = None):
    if params is None:
        params = {}
    logger.info(f"API Request - Method: {method}, Endpoint: {endpoint}, Params: {params}")

def log_api_response(endpoint: str, status_code: int, response_time: float):
    logger.info(f"API Response - Endpoint: {endpoint}, Status: {status_code}, Time: {response_time:.3f}s")