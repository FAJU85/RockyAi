"""
Enhanced logging configuration for Rocky AI
Structured logging with correlation IDs and multiple handlers
"""
import logging
import logging.handlers
import sys
import json
import uuid
from typing import Dict, Any, Optional
from datetime import datetime
from contextvars import ContextVar
from apps.api.app.config import get_settings

# Context variable for correlation ID
correlation_id: ContextVar[Optional[str]] = ContextVar('correlation_id', default=None)


class CorrelationFilter(logging.Filter):
    """Add correlation ID to log records"""
    
    def filter(self, record: logging.LogRecord) -> bool:
        record.correlation_id = correlation_id.get() or "no-correlation-id"
        return True


class StructuredFormatter(logging.Formatter):
    """Structured JSON formatter for logs"""
    
    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "correlation_id": getattr(record, 'correlation_id', 'no-correlation-id'),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        
        # Add exception info if present
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)
        
        # Add extra fields
        for key, value in record.__dict__.items():
            if key not in ['name', 'msg', 'args', 'levelname', 'levelno', 'pathname', 
                          'filename', 'module', 'exc_info', 'exc_text', 'stack_info',
                          'lineno', 'funcName', 'created', 'msecs', 'relativeCreated',
                          'thread', 'threadName', 'processName', 'process', 'getMessage',
                          'correlation_id']:
                log_entry[key] = value
        
        return json.dumps(log_entry, default=str)


class RockyLogger:
    """Enhanced logger for Rocky AI with correlation tracking"""
    
    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
        self._setup_logger()
    
    def _setup_logger(self):
        """Setup logger with handlers and formatters"""
        settings = get_settings()
        
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # Set log level
        self.logger.setLevel(getattr(logging, settings.logging.level.upper()))
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        
        if settings.environment.value == "production":
            # JSON formatter for production
            console_formatter = StructuredFormatter()
        else:
            # Human-readable formatter for development
            console_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - [%(correlation_id)s] - %(message)s'
            )
        
        console_handler.setFormatter(console_formatter)
        console_handler.addFilter(CorrelationFilter())
        self.logger.addHandler(console_handler)
        
        # File handler (if configured)
        if settings.logging.file_path:
            file_handler = logging.handlers.RotatingFileHandler(
                settings.logging.file_path,
                maxBytes=settings.logging.max_file_size,
                backupCount=settings.logging.backup_count
            )
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(StructuredFormatter())
            file_handler.addFilter(CorrelationFilter())
            self.logger.addHandler(file_handler)
        
        # Prevent propagation to root logger
        self.logger.propagate = False
    
    def set_correlation_id(self, correlation_id: str):
        """Set correlation ID for current context"""
        correlation_id.set(correlation_id)
    
    def get_correlation_id(self) -> Optional[str]:
        """Get current correlation ID"""
        return correlation_id.get()
    
    def generate_correlation_id(self) -> str:
        """Generate new correlation ID"""
        new_id = str(uuid.uuid4())
        self.set_correlation_id(new_id)
        return new_id
    
    def debug(self, message: str, **kwargs):
        """Log debug message with extra context"""
        self.logger.debug(message, extra=kwargs)
    
    def info(self, message: str, **kwargs):
        """Log info message with extra context"""
        self.logger.info(message, extra=kwargs)
    
    def warning(self, message: str, **kwargs):
        """Log warning message with extra context"""
        self.logger.warning(message, extra=kwargs)
    
    def error(self, message: str, **kwargs):
        """Log error message with extra context"""
        self.logger.error(message, extra=kwargs)
    
    def critical(self, message: str, **kwargs):
        """Log critical message with extra context"""
        self.logger.critical(message, extra=kwargs)
    
    def log_analysis_start(self, query: str, user_id: Optional[str] = None):
        """Log analysis start"""
        self.info(
            "Analysis started",
            event="analysis_start",
            query=query,
            user_id=user_id
        )
    
    def log_analysis_complete(self, query: str, execution_time: float, 
                            success: bool, user_id: Optional[str] = None):
        """Log analysis completion"""
        self.info(
            "Analysis completed",
            event="analysis_complete",
            query=query,
            execution_time=execution_time,
            success=success,
            user_id=user_id
        )
    
    def log_executor_start(self, language: str, code_length: int):
        """Log executor start"""
        self.info(
            "Code execution started",
            event="executor_start",
            language=language,
            code_length=code_length
        )
    
    def log_executor_complete(self, language: str, execution_time: float, 
                            success: bool, memory_used: float):
        """Log executor completion"""
        self.info(
            "Code execution completed",
            event="executor_complete",
            language=language,
            execution_time=execution_time,
            success=success,
            memory_used_mb=memory_used
        )
    
    def log_error(self, error: Exception, context: Dict[str, Any] = None):
        """Log error with context"""
        self.error(
            f"Error occurred: {str(error)}",
            event="error",
            error_type=type(error).__name__,
            context=context or {}
        )


def get_logger(name: str) -> RockyLogger:
    """Get logger instance"""
    return RockyLogger(name)


def set_correlation_id(correlation_id: str):
    """Set correlation ID for current context"""
    correlation_id.set(correlation_id)


def get_correlation_id() -> Optional[str]:
    """Get current correlation ID"""
    return correlation_id.get()


def generate_correlation_id() -> str:
    """Generate new correlation ID"""
    new_id = str(uuid.uuid4())
    correlation_id.set(new_id)
    return new_id
