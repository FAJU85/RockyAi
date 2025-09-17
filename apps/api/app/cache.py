"""
Redis caching service for Rocky AI
High-performance caching with TTL and serialization
"""
import json
import pickle
import hashlib
from typing import Any, Optional, Dict, List, Union
from datetime import datetime, timedelta
import redis.asyncio as redis
from redis.asyncio import Redis
from apps.api.app.config import get_settings
from apps.api.app.logging_config import get_logger

logger = get_logger(__name__)


class CacheService:
    """Redis-based caching service"""
    
    def __init__(self):
        self.settings = get_settings()
        self.redis: Optional[Redis] = None
        self._connection_pool = None
    
    async def connect(self):
        """Connect to Redis"""
        try:
            self._connection_pool = redis.ConnectionPool.from_url(
                self.settings.redis.url,
                max_connections=self.settings.redis.max_connections,
                decode_responses=False  # We'll handle encoding ourselves
            )
            self.redis = redis.Redis(connection_pool=self._connection_pool)
            
            # Test connection
            await self.redis.ping()
            logger.info("Connected to Redis successfully")
            
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise
    
    async def disconnect(self):
        """Disconnect from Redis"""
        if self.redis:
            await self.redis.close()
        if self._connection_pool:
            await self._connection_pool.disconnect()
    
    def _serialize(self, data: Any) -> bytes:
        """Serialize data for storage"""
        try:
            # Try JSON first for simple data types
            return json.dumps(data, default=str).encode('utf-8')
        except (TypeError, ValueError):
            # Fall back to pickle for complex objects
            return pickle.dumps(data)
    
    def _deserialize(self, data: bytes) -> Any:
        """Deserialize data from storage"""
        try:
            # Try JSON first
            return json.loads(data.decode('utf-8'))
        except (json.JSONDecodeError, UnicodeDecodeError):
            # Fall back to pickle
            return pickle.loads(data)
    
    def _generate_key(self, prefix: str, *args) -> str:
        """Generate cache key"""
        key_parts = [prefix] + [str(arg) for arg in args]
        return ":".join(key_parts)
    
    def _generate_hash_key(self, data: str) -> str:
        """Generate hash key for data"""
        return hashlib.sha256(data.encode()).hexdigest()
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set cache value"""
        try:
            if not self.redis:
                await self.connect()
            
            serialized_value = self._serialize(value)
            ttl = ttl or self.settings.cache.default_ttl
            
            result = await self.redis.setex(key, ttl, serialized_value)
            logger.debug(f"Cache set: {key} (TTL: {ttl}s)")
            return bool(result)
            
        except Exception as e:
            logger.error(f"Failed to set cache key {key}: {e}")
            return False
    
    async def get(self, key: str) -> Optional[Any]:
        """Get cache value"""
        try:
            if not self.redis:
                await self.connect()
            
            data = await self.redis.get(key)
            if data is None:
                return None
            
            value = self._deserialize(data)
            logger.debug(f"Cache hit: {key}")
            return value
            
        except Exception as e:
            logger.error(f"Failed to get cache key {key}: {e}")
            return None
    
    async def delete(self, key: str) -> bool:
        """Delete cache key"""
        try:
            if not self.redis:
                await self.connect()
            
            result = await self.redis.delete(key)
            logger.debug(f"Cache delete: {key}")
            return bool(result)
            
        except Exception as e:
            logger.error(f"Failed to delete cache key {key}: {e}")
            return False
    
    async def exists(self, key: str) -> bool:
        """Check if key exists"""
        try:
            if not self.redis:
                await self.connect()
            
            result = await self.redis.exists(key)
            return bool(result)
            
        except Exception as e:
            logger.error(f"Failed to check cache key {key}: {e}")
            return False
    
    async def expire(self, key: str, ttl: int) -> bool:
        """Set TTL for existing key"""
        try:
            if not self.redis:
                await self.connect()
            
            result = await self.redis.expire(key, ttl)
            return bool(result)
            
        except Exception as e:
            logger.error(f"Failed to set TTL for key {key}: {e}")
            return False
    
    async def get_ttl(self, key: str) -> int:
        """Get TTL for key"""
        try:
            if not self.redis:
                await self.connect()
            
            ttl = await self.redis.ttl(key)
            return ttl
            
        except Exception as e:
            logger.error(f"Failed to get TTL for key {key}: {e}")
            return -1
    
    # Specialized cache methods
    
    async def cache_analysis_result(self, query: str, data_path: str, 
                                  result: Dict[str, Any]) -> bool:
        """Cache analysis result"""
        cache_key = self._generate_key("analysis", self._generate_hash_key(f"{query}:{data_path}"))
        return await self.set(cache_key, result, self.settings.cache.analysis_ttl)
    
    async def get_cached_analysis(self, query: str, data_path: str) -> Optional[Dict[str, Any]]:
        """Get cached analysis result"""
        cache_key = self._generate_key("analysis", self._generate_hash_key(f"{query}:{data_path}"))
        return await self.get(cache_key)
    
    async def cache_model_response(self, prompt: str, model_name: str, 
                                 response: str) -> bool:
        """Cache model response"""
        cache_key = self._generate_key("model", model_name, self._generate_hash_key(prompt))
        return await self.set(cache_key, response, self.settings.cache.model_ttl)
    
    async def get_cached_model_response(self, prompt: str, model_name: str) -> Optional[str]:
        """Get cached model response"""
        cache_key = self._generate_key("model", model_name, self._generate_hash_key(prompt))
        return await self.get(cache_key)
    
    async def cache_dataset_metadata(self, dataset_id: str, metadata: Dict[str, Any]) -> bool:
        """Cache dataset metadata"""
        cache_key = self._generate_key("dataset", dataset_id, "metadata")
        return await self.set(cache_key, metadata, self.settings.cache.default_ttl)
    
    async def get_cached_dataset_metadata(self, dataset_id: str) -> Optional[Dict[str, Any]]:
        """Get cached dataset metadata"""
        cache_key = self._generate_key("dataset", dataset_id, "metadata")
        return await self.get(cache_key)
    
    async def cache_user_session(self, user_id: str, session_data: Dict[str, Any]) -> bool:
        """Cache user session"""
        cache_key = self._generate_key("session", user_id)
        return await self.set(cache_key, session_data, 1800)  # 30 minutes
    
    async def get_cached_user_session(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get cached user session"""
        cache_key = self._generate_key("session", user_id)
        return await self.get(cache_key)
    
    async def invalidate_user_cache(self, user_id: str) -> bool:
        """Invalidate all cache entries for user"""
        try:
            if not self.redis:
                await self.connect()
            
            pattern = f"*:session:{user_id}*"
            keys = await self.redis.keys(pattern)
            
            if keys:
                await self.redis.delete(*keys)
                logger.info(f"Invalidated {len(keys)} cache entries for user {user_id}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to invalidate user cache for {user_id}: {e}")
            return False
    
    async def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        try:
            if not self.redis:
                await self.connect()
            
            info = await self.redis.info()
            return {
                "connected_clients": info.get("connected_clients", 0),
                "used_memory": info.get("used_memory_human", "0B"),
                "used_memory_peak": info.get("used_memory_peak_human", "0B"),
                "keyspace_hits": info.get("keyspace_hits", 0),
                "keyspace_misses": info.get("keyspace_misses", 0),
                "total_commands_processed": info.get("total_commands_processed", 0),
                "uptime_in_seconds": info.get("uptime_in_seconds", 0)
            }
            
        except Exception as e:
            logger.error(f"Failed to get cache stats: {e}")
            return {}


# Global cache service instance
cache_service = CacheService()


async def get_cache() -> CacheService:
    """Get cache service instance"""
    if not cache_service.redis:
        await cache_service.connect()
    return cache_service


# Cache decorator
def cache_result(ttl: int = 3600, key_prefix: str = "default"):
    """Decorator to cache function results"""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            # Generate cache key from function name and arguments
            cache_key_parts = [key_prefix, func.__name__] + [str(arg) for arg in args]
            cache_key = ":".join(cache_key_parts)
            
            # Try to get from cache
            cache = await get_cache()
            cached_result = await cache.get(cache_key)
            if cached_result is not None:
                logger.debug(f"Cache hit for {func.__name__}")
                return cached_result
            
            # Execute function and cache result
            result = await func(*args, **kwargs)
            await cache.set(cache_key, result, ttl)
            logger.debug(f"Cached result for {func.__name__}")
            return result
        
        return wrapper
    return decorator
