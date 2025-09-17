"""
Unit tests for security module
"""
import pytest
from unittest.mock import MagicMock, patch
from fastapi import Request, HTTPException
from starlette.responses import Response

from apps.api.app.security import (
    RateLimitMiddleware,
    SecurityHeadersMiddleware,
    InputValidationMiddleware,
    IPWhitelistMiddleware,
    SecurityAuditMiddleware,
    SecurityService
)


class TestRateLimitMiddleware:
    """Test rate limiting middleware"""
    
    @pytest.fixture
    def mock_app(self):
        """Mock ASGI app"""
        return MagicMock()
    
    @pytest.fixture
    def rate_limit_middleware(self, mock_app):
        """Rate limit middleware instance"""
        return RateLimitMiddleware(mock_app, calls_per_minute=10)
    
    def test_rate_limit_initialization(self, mock_app):
        """Test rate limit middleware initialization"""
        middleware = RateLimitMiddleware(mock_app, calls_per_minute=60)
        assert middleware.calls_per_minute == 60
        assert isinstance(middleware.requests, dict)
    
    def test_get_client_ip_direct(self, rate_limit_middleware):
        """Test getting client IP from direct connection"""
        request = MagicMock()
        request.client.host = "192.168.1.100"
        request.headers = {}
        
        ip = rate_limit_middleware._get_client_ip(request)
        assert ip == "192.168.1.100"
    
    def test_get_client_ip_forwarded(self, rate_limit_middleware):
        """Test getting client IP from forwarded header"""
        request = MagicMock()
        request.client.host = "10.0.0.1"
        request.headers = {"X-Forwarded-For": "203.0.113.195, 70.41.3.18, 150.172.238.178"}
        
        ip = rate_limit_middleware._get_client_ip(request)
        assert ip == "203.0.113.195"
    
    def test_get_client_ip_real_ip(self, rate_limit_middleware):
        """Test getting client IP from real IP header"""
        request = MagicMock()
        request.client.host = "10.0.0.1"
        request.headers = {"X-Real-IP": "203.0.113.195"}
        
        ip = rate_limit_middleware._get_client_ip(request)
        assert ip == "203.0.113.195"
    
    def test_is_rate_limited_false(self, rate_limit_middleware):
        """Test rate limit check when not limited"""
        client_ip = "192.168.1.100"
        assert not rate_limit_middleware._is_rate_limited(client_ip)
    
    def test_is_rate_limited_true(self, rate_limit_middleware):
        """Test rate limit check when limited"""
        client_ip = "192.168.1.100"
        
        # Add requests up to the limit
        for _ in range(10):
            rate_limit_middleware._record_request(client_ip)
        
        assert rate_limit_middleware._is_rate_limited(client_ip)
    
    def test_record_request(self, rate_limit_middleware):
        """Test recording a request"""
        client_ip = "192.168.1.100"
        initial_count = len(rate_limit_middleware.requests[client_ip])
        
        rate_limit_middleware._record_request(client_ip)
        
        assert len(rate_limit_middleware.requests[client_ip]) == initial_count + 1
    
    def test_get_remaining_requests(self, rate_limit_middleware):
        """Test getting remaining requests"""
        client_ip = "192.168.1.100"
        
        # No requests recorded
        remaining = rate_limit_middleware._get_remaining_requests(client_ip)
        assert remaining == 10  # calls_per_minute
        
        # Some requests recorded
        for _ in range(3):
            rate_limit_middleware._record_request(client_ip)
        
        remaining = rate_limit_middleware._get_remaining_requests(client_ip)
        assert remaining == 7


class TestSecurityHeadersMiddleware:
    """Test security headers middleware"""
    
    @pytest.fixture
    def mock_app(self):
        """Mock ASGI app"""
        return MagicMock()
    
    @pytest.fixture
    def security_headers_middleware(self, mock_app):
        """Security headers middleware instance"""
        return SecurityHeadersMiddleware(mock_app)
    
    @pytest.mark.asyncio
    async def test_security_headers_added(self, security_headers_middleware):
        """Test that security headers are added to response"""
        request = MagicMock()
        response = Response()
        
        # Mock the next middleware
        async def mock_call_next(req):
            return response
        
        security_headers_middleware.app = mock_call_next
        
        result = await security_headers_middleware.dispatch(request, mock_call_next)
        
        assert "X-Content-Type-Options" in result.headers
        assert "X-Frame-Options" in result.headers
        assert "X-XSS-Protection" in result.headers
        assert "Referrer-Policy" in result.headers
        assert "Permissions-Policy" in result.headers
        assert "Content-Security-Policy" in result.headers
    
    @pytest.mark.asyncio
    async def test_hsts_header_production(self, security_headers_middleware):
        """Test HSTS header in production"""
        request = MagicMock()
        response = Response()
        
        # Mock the next middleware
        async def mock_call_next(req):
            return response
        
        security_headers_middleware.app = mock_call_next
        
        with patch('apps.api.app.security.settings.is_production', return_value=True):
            result = await security_headers_middleware.dispatch(request, mock_call_next)
            assert "Strict-Transport-Security" in result.headers
    
    @pytest.mark.asyncio
    async def test_hsts_header_development(self, security_headers_middleware):
        """Test HSTS header not in development"""
        request = MagicMock()
        response = Response()
        
        # Mock the next middleware
        async def mock_call_next(req):
            return response
        
        security_headers_middleware.app = mock_call_next
        
        with patch('apps.api.app.security.settings.is_production', return_value=False):
            result = await security_headers_middleware.dispatch(request, mock_call_next)
            assert "Strict-Transport-Security" not in result.headers


class TestInputValidationMiddleware:
    """Test input validation middleware"""
    
    @pytest.fixture
    def mock_app(self):
        """Mock ASGI app"""
        return MagicMock()
    
    @pytest.fixture
    def input_validation_middleware(self, mock_app):
        """Input validation middleware instance"""
        return InputValidationMiddleware(mock_app)
    
    def test_contains_suspicious_patterns_true(self, input_validation_middleware):
        """Test detection of suspicious patterns"""
        suspicious_text = "<script>alert('xss')</script>"
        assert input_validation_middleware._contains_suspicious_patterns(suspicious_text)
    
    def test_contains_suspicious_patterns_false(self, input_validation_middleware):
        """Test no suspicious patterns detected"""
        clean_text = "This is a normal text without any suspicious content"
        assert not input_validation_middleware._contains_suspicious_patterns(clean_text)
    
    @pytest.mark.asyncio
    async def test_dispatch_clean_request(self, input_validation_middleware):
        """Test dispatch with clean request"""
        request = MagicMock()
        request.method = "POST"
        request.body = b'{"data": "clean data"}'
        
        response = Response()
        
        # Mock the next middleware
        async def mock_call_next(req):
            return response
        
        input_validation_middleware.app = mock_call_next
        
        result = await input_validation_middleware.dispatch(request, mock_call_next)
        assert result == response
    
    @pytest.mark.asyncio
    async def test_dispatch_suspicious_request(self, input_validation_middleware):
        """Test dispatch with suspicious request"""
        request = MagicMock()
        request.method = "POST"
        request.body = b'<script>alert("xss")</script>'
        
        # Mock the next middleware
        async def mock_call_next(req):
            return Response()
        
        input_validation_middleware.app = mock_call_next
        
        with pytest.raises(HTTPException) as exc_info:
            await input_validation_middleware.dispatch(request, mock_call_next)
        
        assert exc_info.value.status_code == 400
        assert "Invalid input detected" in str(exc_info.value.detail)


class TestIPWhitelistMiddleware:
    """Test IP whitelist middleware"""
    
    @pytest.fixture
    def mock_app(self):
        """Mock ASGI app"""
        return MagicMock()
    
    @pytest.fixture
    def ip_whitelist_middleware(self, mock_app):
        """IP whitelist middleware instance"""
        return IPWhitelistMiddleware(mock_app, allowed_ips=["192.168.1.0/24", "10.0.0.1"])
    
    def test_ip_whitelist_initialization(self, mock_app):
        """Test IP whitelist middleware initialization"""
        middleware = IPWhitelistMiddleware(mock_app, allowed_ips=["192.168.1.0/24"])
        assert len(middleware.allowed_networks) > 0
    
    def test_get_client_ip(self, ip_whitelist_middleware):
        """Test getting client IP"""
        request = MagicMock()
        request.client.host = "192.168.1.100"
        request.headers = {}
        
        ip = ip_whitelist_middleware._get_client_ip(request)
        assert ip == "192.168.1.100"
    
    def test_is_ip_allowed_network(self, ip_whitelist_middleware):
        """Test IP allowed check with network"""
        # IP in allowed network
        assert ip_whitelist_middleware._is_ip_allowed("192.168.1.100")
        
        # IP not in allowed network
        assert not ip_whitelist_middleware._is_ip_allowed("203.0.113.195")
    
    def test_is_ip_allowed_specific(self, ip_whitelist_middleware):
        """Test IP allowed check with specific IP"""
        # Specific allowed IP
        assert ip_whitelist_middleware._is_ip_allowed("10.0.0.1")
        
        # IP not in allowed list
        assert not ip_whitelist_middleware._is_ip_allowed("10.0.0.2")
    
    def test_is_ip_allowed_invalid(self, ip_whitelist_middleware):
        """Test IP allowed check with invalid IP"""
        assert not ip_whitelist_middleware._is_ip_allowed("invalid-ip")
    
    @pytest.mark.asyncio
    async def test_dispatch_allowed_ip(self, ip_whitelist_middleware):
        """Test dispatch with allowed IP"""
        request = MagicMock()
        request.client.host = "192.168.1.100"
        request.headers = {}
        
        response = Response()
        
        # Mock the next middleware
        async def mock_call_next(req):
            return response
        
        ip_whitelist_middleware.app = mock_call_next
        
        result = await ip_whitelist_middleware.dispatch(request, mock_call_next)
        assert result == response
    
    @pytest.mark.asyncio
    async def test_dispatch_blocked_ip(self, ip_whitelist_middleware):
        """Test dispatch with blocked IP"""
        request = MagicMock()
        request.client.host = "203.0.113.195"
        request.headers = {}
        
        # Mock the next middleware
        async def mock_call_next(req):
            return Response()
        
        ip_whitelist_middleware.app = mock_call_next
        
        with pytest.raises(HTTPException) as exc_info:
            await ip_whitelist_middleware.dispatch(request, mock_call_next)
        
        assert exc_info.value.status_code == 403
        assert "Access denied from this IP address" in str(exc_info.value.detail)


class TestSecurityAuditMiddleware:
    """Test security audit middleware"""
    
    @pytest.fixture
    def mock_app(self):
        """Mock ASGI app"""
        return MagicMock()
    
    @pytest.fixture
    def security_audit_middleware(self, mock_app):
        """Security audit middleware instance"""
        return SecurityAuditMiddleware(mock_app)
    
    def test_generate_request_id(self, security_audit_middleware):
        """Test request ID generation"""
        request = MagicMock()
        request.url = "http://example.com/test"
        request.client.host = "192.168.1.100"
        
        request_id = security_audit_middleware._generate_request_id(request)
        
        assert isinstance(request_id, str)
        assert len(request_id) == 16
    
    @pytest.mark.asyncio
    async def test_dispatch_logs_request(self, security_audit_middleware):
        """Test that dispatch logs the request"""
        request = MagicMock()
        request.method = "GET"
        request.url = "http://example.com/test"
        request.client.host = "192.168.1.100"
        
        response = Response()
        
        # Mock the next middleware
        async def mock_call_next(req):
            return response
        
        security_audit_middleware.app = mock_call_next
        
        with patch('apps.api.app.security.logger') as mock_logger:
            result = await security_audit_middleware.dispatch(request, mock_call_next)
            
            # Check that logging was called
            assert mock_logger.info.called
            assert result == response


class TestSecurityService:
    """Test security service utilities"""
    
    def test_validate_file_upload_valid(self):
        """Test valid file upload validation"""
        assert SecurityService.validate_file_upload("test.csv", "text/csv", 1024)
        assert SecurityService.validate_file_upload("data.json", "application/json", 2048)
        assert SecurityService.validate_file_upload("report.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", 5120)
    
    def test_validate_file_upload_invalid_extension(self):
        """Test invalid file extension validation"""
        assert not SecurityService.validate_file_upload("malicious.exe", "application/octet-stream", 1024)
        assert not SecurityService.validate_file_upload("script.js", "application/javascript", 1024)
    
    def test_validate_file_upload_invalid_type(self):
        """Test invalid MIME type validation"""
        assert not SecurityService.validate_file_upload("test.csv", "application/octet-stream", 1024)
        assert not SecurityService.validate_file_upload("data.json", "text/plain", 1024)
    
    def test_validate_file_upload_too_large(self):
        """Test file size validation"""
        assert not SecurityService.validate_file_upload("huge.csv", "text/csv", 20 * 1024 * 1024)  # 20MB
    
    def test_sanitize_filename(self):
        """Test filename sanitization"""
        # Test path components removal
        assert SecurityService.sanitize_filename("/path/to/file.csv") == "file.csv"
        assert SecurityService.sanitize_filename("C:\\path\\to\\file.csv") == "file.csv"
        
        # Test dangerous characters
        assert SecurityService.sanitize_filename("file<name>.csv") == "file_name_.csv"
        assert SecurityService.sanitize_filename("file:name.csv") == "file_name.csv"
        
        # Test length limit
        long_filename = "a" * 300 + ".csv"
        sanitized = SecurityService.sanitize_filename(long_filename)
        assert len(sanitized) <= 255
    
    def test_validate_sql_query_safe(self):
        """Test safe SQL query validation"""
        assert SecurityService.validate_sql_query("SELECT * FROM users WHERE id = 1")
        assert SecurityService.validate_sql_query("SELECT name, email FROM users")
        assert SecurityService.validate_sql_query("SELECT COUNT(*) FROM orders")
    
    def test_validate_sql_query_dangerous(self):
        """Test dangerous SQL query validation"""
        assert not SecurityService.validate_sql_query("DROP TABLE users")
        assert not SecurityService.validate_sql_query("DELETE FROM users WHERE 1=1")
        assert not SecurityService.validate_sql_query("INSERT INTO users VALUES ('hacker', 'admin')")
        assert not SecurityService.validate_sql_query("UPDATE users SET role = 'admin'")
        assert not SecurityService.validate_sql_query("SELECT * FROM users UNION SELECT * FROM passwords")
    
    def test_generate_secure_token(self):
        """Test secure token generation"""
        token = SecurityService.generate_secure_token(32)
        assert isinstance(token, str)
        assert len(token) == 32
    
    def test_hash_sensitive_data(self):
        """Test sensitive data hashing"""
        data = "sensitive information"
        hashed = SecurityService.hash_sensitive_data(data)
        
        assert isinstance(hashed, str)
        assert len(hashed) == 64  # SHA-256 hex length
        assert hashed != data
        assert SecurityService.hash_sensitive_data(data) == hashed  # Deterministic
