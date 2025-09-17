"""
Enhanced executor service for Rocky AI
Integrates Python and R executors with the main orchestrator
"""
import asyncio
import httpx
import json
from typing import Dict, Any, Optional, Tuple
from datetime import datetime
from apps.api.app.config import get_settings
from apps.api.app.logging_config import get_logger, generate_correlation_id
from apps.api.app.cache import get_cache
from services.runners.python.executor import PythonExecutor, ExecutionResult
from services.runners.r.executor import execute_r_code

logger = get_logger(__name__)


class ExecutorService:
    """Enhanced executor service with caching and error handling"""
    
    def __init__(self):
        self.settings = get_settings()
        self.python_executor = PythonExecutor(
            max_execution_time=self.settings.executor.max_execution_time,
            max_memory_mb=self.settings.executor.max_memory_mb,
            max_output_size=self.settings.executor.max_output_size,
            working_dir=self.settings.executor.working_dir
        )
        self.cache = None
    
    async def initialize(self):
        """Initialize executor service"""
        self.cache = await get_cache()
        logger.info("Executor service initialized")
    
    async def execute_code(self, code: str, language: str, 
                          analysis_id: Optional[str] = None,
                          correlation_id: Optional[str] = None) -> Dict[str, Any]:
        """Execute code in the specified language"""
        if not correlation_id:
            correlation_id = generate_correlation_id()
        
        logger.log_executor_start(language, len(code))
        start_time = datetime.now()
        
        try:
            # Check cache first
            if self.cache and analysis_id:
                cache_key = f"execution:{analysis_id}:{hash(code)}"
                cached_result = await self.cache.get(cache_key)
                if cached_result:
                    logger.info(f"Using cached execution result for analysis {analysis_id}")
                    return cached_result
            
            # Execute code based on language
            if language.lower() == "python":
                result = await self._execute_python(code)
            elif language.lower() == "r":
                result = await self._execute_r(code)
            else:
                raise ValueError(f"Unsupported language: {language}")
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            # Log completion
            logger.log_executor_complete(
                language, 
                execution_time, 
                result["success"], 
                result.get("memory_used_mb", 0)
            )
            
            # Cache result if successful
            if self.cache and analysis_id and result["success"]:
                await self.cache.set(cache_key, result, ttl=3600)
            
            return result
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            logger.log_error(e, {
                "language": language,
                "analysis_id": analysis_id,
                "execution_time": execution_time
            })
            
            return {
                "success": False,
                "output": "",
                "error": str(e),
                "execution_time": execution_time,
                "memory_used_mb": 0,
                "language": language,
                "correlation_id": correlation_id
            }
    
    async def _execute_python(self, code: str) -> Dict[str, Any]:
        """Execute Python code"""
        try:
            result = await self.python_executor.execute(code)
            
            return {
                "success": result.success,
                "output": result.output,
                "error": result.error,
                "execution_time": result.execution_time,
                "memory_used_mb": result.memory_used_mb,
                "return_code": result.return_code,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "language": "python"
            }
            
        except Exception as e:
            logger.error(f"Python execution failed: {e}")
            return {
                "success": False,
                "output": "",
                "error": str(e),
                "execution_time": 0,
                "memory_used_mb": 0,
                "return_code": -1,
                "stdout": "",
                "stderr": str(e),
                "language": "python"
            }
    
    async def _execute_r(self, code: str) -> Dict[str, Any]:
        """Execute R code"""
        try:
            # For now, we'll use a simplified R execution
            # In production, this would call the R executor service
            result = execute_r_code(code)
            
            return {
                "success": result.get("success", False),
                "output": result.get("output", ""),
                "error": result.get("error", ""),
                "execution_time": result.get("execution_time", 0),
                "memory_used_mb": result.get("memory_used_mb", 0),
                "return_code": result.get("return_code", 0),
                "stdout": result.get("output", ""),
                "stderr": result.get("error", ""),
                "language": "r"
            }
            
        except Exception as e:
            logger.error(f"R execution failed: {e}")
            return {
                "success": False,
                "output": "",
                "error": str(e),
                "execution_time": 0,
                "memory_used_mb": 0,
                "return_code": -1,
                "stdout": "",
                "stderr": str(e),
                "language": "r"
            }
    
    async def validate_code(self, code: str, language: str) -> Tuple[bool, str]:
        """Validate code before execution"""
        try:
            if language.lower() == "python":
                return self.python_executor._validate_code(code)
            elif language.lower() == "r":
                # R validation would be implemented here
                return True, ""
            else:
                return False, f"Unsupported language: {language}"
                
        except Exception as e:
            logger.error(f"Code validation failed: {e}")
            return False, str(e)
    
    async def get_execution_stats(self) -> Dict[str, Any]:
        """Get execution statistics"""
        try:
            # This would collect stats from executors
            return {
                "python_executions": 0,  # Would be tracked
                "r_executions": 0,       # Would be tracked
                "total_executions": 0,   # Would be tracked
                "average_execution_time": 0.0,
                "success_rate": 0.0
            }
        except Exception as e:
            logger.error(f"Failed to get execution stats: {e}")
            return {}
    
    async def cleanup_old_executions(self):
        """Clean up old execution files"""
        try:
            # Cleanup would be implemented here
            logger.info("Cleaned up old execution files")
        except Exception as e:
            logger.error(f"Failed to cleanup old executions: {e}")


# Global executor service instance
executor_service = ExecutorService()


async def get_executor_service() -> ExecutorService:
    """Get executor service instance"""
    if not executor_service.cache:
        await executor_service.initialize()
    return executor_service
