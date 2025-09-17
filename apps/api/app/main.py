from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Dict, Optional, Any, List
import os
import httpx
import asyncio
from contextlib import asynccontextmanager

from apps.api.app.config import get_settings
from apps.api.app.logging_config import get_logger
from apps.api.app.database import init_database, check_database_health
from apps.api.app.cache import get_cache
from apps.api.app.orchestrator import get_orchestrator, close_orchestrator
from apps.api.app.websocket_routes import router as websocket_router
from apps.api.app.metrics_routes import router as metrics_router
from apps.api.app.analytics_routes import router as analytics_router
from apps.api.app.security_routes import router as security_router
from apps.api.app.security import (
    RateLimitMiddleware,
    SecurityHeadersMiddleware,
    InputValidationMiddleware,
    IPWhitelistMiddleware,
    SecurityAuditMiddleware
)
from apps.api.app.metrics import get_metrics_collector, AnalysisTimer

# Configure logging
logger = get_logger(__name__)
settings = get_settings()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    logger.info("Starting Rocky AI API...")
    
    # Initialize database
    if settings.enable_database:
        try:
            init_database()
            logger.info("Database initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
    
    # Initialize cache
    if settings.enable_caching:
        try:
            cache = await get_cache()
            logger.info("Cache initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize cache: {e}")
    
    # Initialize orchestrator
    try:
        orchestrator = await get_orchestrator()
        logger.info("Orchestrator initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize orchestrator: {e}")
    
    logger.info("Rocky AI API started successfully")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Rocky AI API...")
    await close_orchestrator()
    logger.info("Rocky AI API shutdown complete")


app = FastAPI(
    title="Rocky AI API",
    version=settings.version,
    description="AI Research Assistant API with statistical analysis capabilities",
    lifespan=lifespan
)

# Security middleware
if settings.is_production():
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=["localhost", "127.0.0.1", "*.yourdomain.com"]
    )

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.security.allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# Security middleware
app.add_middleware(SecurityAuditMiddleware)
app.add_middleware(InputValidationMiddleware)
app.add_middleware(SecurityHeadersMiddleware)
app.add_middleware(RateLimitMiddleware, calls_per_minute=60)

# IP whitelist middleware (if configured)
if settings.security.allowed_ips:
    app.add_middleware(IPWhitelistMiddleware, allowed_ips=settings.security.allowed_ips)

# Include WebSocket routes
app.include_router(websocket_router)

# Include metrics routes
app.include_router(metrics_router)

# Include analytics routes
app.include_router(analytics_router)

# Include security routes
app.include_router(security_router)

# Pydantic models for request/response
class AnalysisRequest(BaseModel):
    query: str
    data_path: Optional[str] = None
    context: Optional[Dict[str, Any]] = None

class AnalysisResponse(BaseModel):
    query: str
    plan: Optional[Dict[str, Any]] = None
    code: Optional[str] = None
    status: str
    error: Optional[str] = None

class HealthResponse(BaseModel):
    status: str
    services: Dict[str, str]

@app.get("/health", response_model=HealthResponse)
async def health():
    """Enhanced health check endpoint"""
    services = {
        "api": "ok",
        "dmr": "unknown",
        "database": "unknown",
        "cache": "unknown"
    }
    
    # Check DMR status
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{settings.dmr.base_url}/health")
            if response.status_code == 200:
                services["dmr"] = "ok"
            else:
                services["dmr"] = "error"
    except Exception as e:
        logger.warning(f"DMR health check failed: {e}")
        services["dmr"] = "unavailable"
    
    # Check database status
    if settings.enable_database:
        try:
            if check_database_health():
                services["database"] = "ok"
            else:
                services["database"] = "error"
        except Exception as e:
            logger.warning(f"Database health check failed: {e}")
            services["database"] = "unavailable"
    else:
        services["database"] = "disabled"
    
    # Check cache status
    if settings.enable_caching:
        try:
            cache = await get_cache()
            stats = await cache.get_cache_stats()
            if stats:
                services["cache"] = "ok"
            else:
                services["cache"] = "error"
        except Exception as e:
            logger.warning(f"Cache health check failed: {e}")
            services["cache"] = "unavailable"
    else:
        services["cache"] = "disabled"
    
    # Determine overall status
    overall_status = "ok" if all(status in ["ok", "disabled"] for status in services.values()) else "degraded"
    
    return HealthResponse(status=overall_status, services=services)

@app.post("/ai/generate")
async def ai_generate(payload: dict):
    """Legacy AI generation endpoint"""
    url = f"{DMR_BASE_URL}/v1/chat/completions"
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            r = await client.post(url, json=payload)
            r.raise_for_status()
            return r.json()
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=str(e))

@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_data(request: AnalysisRequest, background_tasks: BackgroundTasks):
    """Enhanced analysis endpoint with full pipeline integration and metrics"""
    metrics_collector = get_metrics_collector()
    
    try:
        logger.info(f"Received analysis request: {request.query[:100]}...")
        
        orchestrator = await get_orchestrator()
        
        # Record analysis request
        metrics_collector.record_analysis_request(
            analysis_type="unknown",  # Will be updated after planning
            language="unknown",       # Will be updated after planning
            status="started"
        )
        
        result = await orchestrator.execute_analysis(
            query=request.query,
            data_path=request.data_path,
            context=request.context,
            user_id=getattr(request, 'user_id', None),
            dataset_id=getattr(request, 'dataset_id', None)
        )
        
        # Record analysis completion
        if result.get("plan"):
            metrics_collector.record_analysis_request(
                analysis_type=result["plan"].get("analysis_type", "unknown"),
                language=result["plan"].get("language", "unknown"),
                status=result.get("status", "unknown")
            )
        
        # Add background task for cleanup if needed
        if result.get("analysis_id"):
            background_tasks.add_task(cleanup_analysis, result["analysis_id"])
        
        return AnalysisResponse(**result)
        
    except Exception as e:
        logger.log_error(e, {"query": request.query})
        
        # Record analysis error
        metrics_collector.record_analysis_error(
            error_type=type(e).__name__,
            analysis_type="unknown"
        )
        
        return AnalysisResponse(
            query=request.query,
            status="failed",
            error=str(e)
        )


async def cleanup_analysis(analysis_id: str):
    """Background task to cleanup analysis resources"""
    try:
        # Implement cleanup logic here
        logger.info(f"Cleaning up analysis {analysis_id}")
    except Exception as e:
        logger.error(f"Failed to cleanup analysis {analysis_id}: {e}")

@app.get("/models")
async def list_models():
    """List available AI models"""
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(f"{DMR_BASE_URL}/v1/models")
            if response.status_code == 200:
                return response.json()
            else:
                return {"error": "Failed to fetch models"}
    except Exception as e:
        logger.error(f"Failed to list models: {e}")
        return {"error": str(e)}

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "name": "Rocky AI API",
        "version": "0.1.0",
        "description": "AI Research Assistant API",
        "endpoints": {
            "health": "/health",
            "analyze": "/analyze",
            "models": "/models",
            "docs": "/docs"
        }
    }
