"""Logging configuration for Hartford AI agents."""

import logging
import json
import sys
from datetime import datetime
from typing import Optional
from pathlib import Path


class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging."""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON.
        
        Args:
            record: Log record
            
        Returns:
            JSON formatted log string
        """
        log_obj = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }
        
        if hasattr(record, 'agent_name'):
            log_obj['agent_name'] = record.agent_name
        
        if hasattr(record, 'session_id'):
            log_obj['session_id'] = record.session_id
        
        if record.exc_info:
            log_obj['exception'] = self.formatException(record.exc_info)
        
        return json.dumps(log_obj)


def get_logger(
    name: str,
    level: Optional[str] = None,
    log_format: Optional[str] = None
) -> logging.Logger:
    """Get configured logger instance.
    
    Args:
        name: Logger name
        level: Log level (default from config)
        log_format: Log format ('json' or 'text')
        
    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    
    # Only configure if not already configured
    if not logger.handlers:
        from ..config.config import get_config
        config = get_config()
        
        # Set log level
        log_level = level or config.log_level
        logger.setLevel(getattr(logging, log_level.upper()))
        
        # Create console handler
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(getattr(logging, log_level.upper()))
        
        # Set formatter
        format_type = log_format or config.log_format
        if format_type == "json":
            formatter = JSONFormatter()
        else:
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
        
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        # Add file handler for persistent logging
        log_dir = Path("/tmp/hartford_agents/logs")
        log_dir.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(
            log_dir / f"{name.replace('.', '_')}.log"
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger