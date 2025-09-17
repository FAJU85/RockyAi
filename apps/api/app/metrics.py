"""
Prometheus metrics for Rocky AI
Comprehensive monitoring and observability
"""
from prometheus_client import Counter, Histogram, Gauge, Summary, Info, generate_latest, CONTENT_TYPE_LATEST
from prometheus_client.core import CollectorRegistry
from fastapi import Response
from typing import Dict, Any
import time
from apps.api.app.logging_config import get_logger

logger = get_logger(__name__)

# Create a custom registry
registry = CollectorRegistry()

# Analysis metrics
analysis_requests_total = Counter(
    'rocky_analysis_requests_total',
    'Total number of analysis requests',
    ['analysis_type', 'language', 'status'],
    registry=registry
)

analysis_duration_seconds = Histogram(
    'rocky_analysis_duration_seconds',
    'Time spent on analysis requests',
    ['analysis_type', 'language'],
    buckets=[0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0],
    registry=registry
)

analysis_errors_total = Counter(
    'rocky_analysis_errors_total',
    'Total number of analysis errors',
    ['error_type', 'analysis_type'],
    registry=registry
)

# Code execution metrics
code_execution_duration_seconds = Histogram(
    'rocky_code_execution_duration_seconds',
    'Time spent executing code',
    ['language', 'status'],
    buckets=[0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0],
    registry=registry
)

code_execution_memory_usage_bytes = Histogram(
    'rocky_code_execution_memory_usage_bytes',
    'Memory usage during code execution',
    ['language'],
    buckets=[1024*1024, 5*1024*1024, 10*1024*1024, 25*1024*1024, 50*1024*1024, 100*1024*1024, 250*1024*1024, 500*1024*1024],
    registry=registry
)

# Database metrics
database_connections_active = Gauge(
    'rocky_database_connections_active',
    'Number of active database connections',
    registry=registry
)

database_query_duration_seconds = Histogram(
    'rocky_database_query_duration_seconds',
    'Time spent on database queries',
    ['query_type'],
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
    registry=registry
)

# Cache metrics
cache_operations_total = Counter(
    'rocky_cache_operations_total',
    'Total number of cache operations',
    ['operation', 'status'],
    registry=registry
)

cache_hit_ratio = Gauge(
    'rocky_cache_hit_ratio',
    'Cache hit ratio',
    registry=registry
)

# WebSocket metrics
websocket_connections_active = Gauge(
    'rocky_websocket_connections_active',
    'Number of active WebSocket connections',
    registry=registry
)

websocket_messages_total = Counter(
    'rocky_websocket_messages_total',
    'Total number of WebSocket messages',
    ['message_type'],
    registry=registry
)

# System metrics
system_cpu_usage_percent = Gauge(
    'rocky_system_cpu_usage_percent',
    'System CPU usage percentage',
    registry=registry
)

system_memory_usage_bytes = Gauge(
    'rocky_system_memory_usage_bytes',
    'System memory usage in bytes',
    registry=registry
)

system_disk_usage_bytes = Gauge(
    'rocky_system_disk_usage_bytes',
    'System disk usage in bytes',
    registry=registry
)

# AI model metrics
ai_model_requests_total = Counter(
    'rocky_ai_model_requests_total',
    'Total number of AI model requests',
    ['model_name', 'status'],
    registry=registry
)

ai_model_tokens_used_total = Counter(
    'rocky_ai_model_tokens_used_total',
    'Total number of AI model tokens used',
    ['model_name'],
    registry=registry
)

ai_model_response_duration_seconds = Histogram(
    'rocky_ai_model_response_duration_seconds',
    'Time spent waiting for AI model responses',
    ['model_name'],
    buckets=[0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0],
    registry=registry
)

# Application info
application_info = Info(
    'rocky_application_info',
    'Application information',
    registry=registry
)

# Set application info
application_info.info({
    'version': '0.2.0',
    'name': 'Rocky AI',
    'description': 'AI Research Assistant'
})


class MetricsCollector:
    """Centralized metrics collection and management"""
    
    def __init__(self):
        self.registry = registry
        self.start_time = time.time()
    
    def record_analysis_request(self, analysis_type: str, language: str, status: str):
        """Record an analysis request"""
        analysis_requests_total.labels(
            analysis_type=analysis_type,
            language=language,
            status=status
        ).inc()
    
    def record_analysis_duration(self, analysis_type: str, language: str, duration: float):
        """Record analysis duration"""
        analysis_duration_seconds.labels(
            analysis_type=analysis_type,
            language=language
        ).observe(duration)
    
    def record_analysis_error(self, error_type: str, analysis_type: str):
        """Record an analysis error"""
        analysis_errors_total.labels(
            error_type=error_type,
            analysis_type=analysis_type
        ).inc()
    
    def record_code_execution(self, language: str, duration: float, memory_bytes: int, status: str):
        """Record code execution metrics"""
        code_execution_duration_seconds.labels(
            language=language,
            status=status
        ).observe(duration)
        
        code_execution_memory_usage_bytes.labels(
            language=language
        ).observe(memory_bytes)
    
    def record_database_operation(self, query_type: str, duration: float):
        """Record database operation metrics"""
        database_query_duration_seconds.labels(
            query_type=query_type
        ).observe(duration)
    
    def record_cache_operation(self, operation: str, status: str):
        """Record cache operation metrics"""
        cache_operations_total.labels(
            operation=operation,
            status=status
        ).inc()
    
    def update_cache_hit_ratio(self, ratio: float):
        """Update cache hit ratio"""
        cache_hit_ratio.set(ratio)
    
    def update_websocket_connections(self, count: int):
        """Update WebSocket connection count"""
        websocket_connections_active.set(count)
    
    def record_websocket_message(self, message_type: str):
        """Record WebSocket message"""
        websocket_messages_total.labels(
            message_type=message_type
        ).inc()
    
    def record_ai_model_request(self, model_name: str, status: str, tokens: int = 0, duration: float = 0):
        """Record AI model request metrics"""
        ai_model_requests_total.labels(
            model_name=model_name,
            status=status
        ).inc()
        
        if tokens > 0:
            ai_model_tokens_used_total.labels(
                model_name=model_name
            ).inc(tokens)
        
        if duration > 0:
            ai_model_response_duration_seconds.labels(
                model_name=model_name
            ).observe(duration)
    
    def update_system_metrics(self, cpu_percent: float, memory_bytes: int, disk_bytes: int):
        """Update system metrics"""
        system_cpu_usage_percent.set(cpu_percent)
        system_memory_usage_bytes.set(memory_bytes)
        system_disk_usage_bytes.set(disk_bytes)
    
    def get_metrics(self) -> str:
        """Get all metrics in Prometheus format"""
        return generate_latest(registry)
    
    def get_uptime_seconds(self) -> float:
        """Get application uptime in seconds"""
        return time.time() - self.start_time


# Global metrics collector
metrics_collector = MetricsCollector()


def get_metrics_collector() -> MetricsCollector:
    """Get the global metrics collector"""
    return metrics_collector


# Context managers for automatic metric collection
class AnalysisTimer:
    """Context manager for timing analysis operations"""
    
    def __init__(self, analysis_type: str, language: str):
        self.analysis_type = analysis_type
        self.language = language
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time:
            duration = time.time() - self.start_time
            metrics_collector.record_analysis_duration(self.analysis_type, self.language, duration)
            
            if exc_type:
                metrics_collector.record_analysis_error(str(exc_type), self.analysis_type)


class DatabaseTimer:
    """Context manager for timing database operations"""
    
    def __init__(self, query_type: str):
        self.query_type = query_type
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time:
            duration = time.time() - self.start_time
            metrics_collector.record_database_operation(self.query_type, duration)


class CodeExecutionTimer:
    """Context manager for timing code execution"""
    
    def __init__(self, language: str):
        self.language = language
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time:
            duration = time.time() - self.start_time
            status = 'success' if not exc_type else 'error'
            metrics_collector.record_code_execution(self.language, duration, 0, status)
