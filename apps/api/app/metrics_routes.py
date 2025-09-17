"""
Metrics routes for Rocky AI
Prometheus metrics and health monitoring endpoints
"""
from fastapi import APIRouter, Depends, Response
from fastapi.responses import PlainTextResponse
from typing import Dict, Any
import psutil
import time
from apps.api.app.metrics import get_metrics_collector, MetricsCollector
from apps.api.app.logging_config import get_logger
from apps.api.app.websocket import get_connection_manager

logger = get_logger(__name__)
router = APIRouter()


@router.get("/metrics", response_class=PlainTextResponse)
async def get_metrics(metrics_collector: MetricsCollector = Depends(get_metrics_collector)):
    """Get Prometheus metrics"""
    try:
        # Update system metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        metrics_collector.update_system_metrics(
            cpu_percent=cpu_percent,
            memory_bytes=memory.used,
            disk_bytes=disk.used
        )
        
        # Get WebSocket connection count
        websocket_manager = await get_connection_manager()
        connection_stats = websocket_manager.get_connection_stats()
        metrics_collector.update_websocket_connections(connection_stats['total_connections'])
        
        # Generate metrics
        metrics_data = metrics_collector.get_metrics()
        
        return Response(
            content=metrics_data,
            media_type="text/plain; version=0.0.4; charset=utf-8"
        )
        
    except Exception as e:
        logger.error(f"Failed to generate metrics: {e}")
        return Response(
            content="# Error generating metrics\n",
            media_type="text/plain"
        )


@router.get("/health/detailed")
async def detailed_health_check(metrics_collector: MetricsCollector = Depends(get_metrics_collector)):
    """Detailed health check with metrics"""
    try:
        # System metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        # WebSocket metrics
        websocket_manager = await get_connection_manager()
        websocket_stats = websocket_manager.get_connection_stats()
        
        # Application metrics
        uptime_seconds = metrics_collector.get_uptime_seconds()
        
        health_data = {
            "status": "healthy",
            "timestamp": time.time(),
            "uptime_seconds": uptime_seconds,
            "system": {
                "cpu_percent": cpu_percent,
                "memory": {
                    "used_bytes": memory.used,
                    "total_bytes": memory.total,
                    "percent": memory.percent
                },
                "disk": {
                    "used_bytes": disk.used,
                    "total_bytes": disk.total,
                    "percent": (disk.used / disk.total) * 100
                }
            },
            "websocket": websocket_stats,
            "application": {
                "version": "0.2.0",
                "environment": "production"
            }
        }
        
        return health_data
        
    except Exception as e:
        logger.error(f"Detailed health check failed: {e}")
        return {
            "status": "unhealthy",
            "timestamp": time.time(),
            "error": str(e)
        }


@router.get("/health/readiness")
async def readiness_check():
    """Kubernetes readiness probe"""
    try:
        # Check if all critical services are ready
        checks = {
            "database": True,  # Would check database connectivity
            "cache": True,     # Would check Redis connectivity
            "ai_model": True,  # Would check DMR connectivity
            "websocket": True  # Would check WebSocket service
        }
        
        all_ready = all(checks.values())
        
        if all_ready:
            return {"status": "ready", "checks": checks}
        else:
            return Response(
                content={"status": "not_ready", "checks": checks},
                status_code=503
            )
            
    except Exception as e:
        logger.error(f"Readiness check failed: {e}")
        return Response(
            content={"status": "not_ready", "error": str(e)},
            status_code=503
        )


@router.get("/health/liveness")
async def liveness_check():
    """Kubernetes liveness probe"""
    try:
        # Basic liveness check
        return {"status": "alive", "timestamp": time.time()}
        
    except Exception as e:
        logger.error(f"Liveness check failed: {e}")
        return Response(
            content={"status": "dead", "error": str(e)},
            status_code=503
        )


@router.get("/metrics/summary")
async def metrics_summary(metrics_collector: MetricsCollector = Depends(get_metrics_collector)):
    """Get a summary of key metrics"""
    try:
        # This would typically query the metrics registry for specific values
        # For now, return a basic summary
        return {
            "uptime_seconds": metrics_collector.get_uptime_seconds(),
            "system": {
                "cpu_percent": psutil.cpu_percent(interval=1),
                "memory_percent": psutil.virtual_memory().percent,
                "disk_percent": (psutil.disk_usage('/').used / psutil.disk_usage('/').total) * 100
            },
            "websocket": await get_connection_manager().then(lambda m: m.get_connection_stats()),
            "timestamp": time.time()
        }
        
    except Exception as e:
        logger.error(f"Failed to get metrics summary: {e}")
        return {"error": str(e)}


@router.get("/metrics/export")
async def export_metrics(metrics_collector: MetricsCollector = Depends(get_metrics_collector)):
    """Export metrics in JSON format for external monitoring systems"""
    try:
        # This would export metrics in a format suitable for external systems
        # like DataDog, New Relic, etc.
        return {
            "metrics": {
                "uptime_seconds": metrics_collector.get_uptime_seconds(),
                "system_cpu_percent": psutil.cpu_percent(interval=1),
                "system_memory_percent": psutil.virtual_memory().percent,
                "system_disk_percent": (psutil.disk_usage('/').used / psutil.disk_usage('/').total) * 100
            },
            "timestamp": time.time(),
            "source": "rocky-ai"
        }
        
    except Exception as e:
        logger.error(f"Failed to export metrics: {e}")
        return {"error": str(e)}
