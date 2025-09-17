"""
Security middleware and utilities for Rocky AI
Rate limiting, input validation, and security headers
"""
import time
import hashlib
import re
from typing import Dict, List, Optional, Any
from collections import defaultdict, deque
from fastapi import Request, HTTPException, status
from fastapi.responses import Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp
import ipaddress
from apps.api.app.logging_config import get_logger
from apps.api.app.config import get_settings

logger = get_logger(__name__)
settings = get_settings()


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Rate limiting middleware"""
    
    def __init__(self, app: ASGIApp, calls_per_minute: int = 60):
        super().__init__(app)
        self.calls_per_minute = calls_per_minute
        self.requests = defaultdict(deque)
    
    async def dispatch(self, request: Request, call_next):
        # Get client IP
        client_ip = self._get_client_ip(request)
        
        # Check rate limit
        if self._is_rate_limited(client_ip):
            logger.warning(f"Rate limit exceeded for IP: {client_ip}")
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Rate limit exceeded. Please try again later."
            )
        
        # Record request
        self._record_request(client_ip)
        
        # Process request
        response = await call_next(request)
        
        # Add rate limit headers
        response.headers["X-RateLimit-Limit"] = str(self.calls_per_minute)
        response.headers["X-RateLimit-Remaining"] = str(self._get_remaining_requests(client_ip))
        
        return response
    
    def _get_client_ip(self, request: Request) -> str:
        """Get client IP address"""
        # Check for forwarded IP
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        
        # Check for real IP
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip
        
        # Use direct client IP
        return request.client.host if request.client else "unknown"
    
    def _is_rate_limited(self, client_ip: str) -> bool:
        """Check if client is rate limited"""
        now = time.time()
        minute_ago = now - 60
        
        # Clean old requests
        while self.requests[client_ip] and self.requests[client_ip][0] < minute_ago:
            self.requests[client_ip].popleft()
        
        # Check if limit exceeded
        return len(self.requests[client_ip]) >= self.calls_per_minute
    
    def _record_request(self, client_ip: str):
        """Record a request"""
        self.requests[client_ip].append(time.time())
    
    def _get_remaining_requests(self, client_ip: str) -> int:
        """Get remaining requests for client"""
        now = time.time()
        minute_ago = now - 60
        
        # Clean old requests
        while self.requests[client_ip] and self.requests[client_ip][0] < minute_ago:
            self.requests[client_ip].popleft()
        
        return max(0, self.calls_per_minute - len(self.requests[client_ip]))


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Security headers middleware"""
    
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        
        # Add security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Permissions-Policy"] = "geolocation=(), microphone=(), camera=()"
        
        # Content Security Policy
        csp = (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline' 'unsafe-eval'; "
            "style-src 'self' 'unsafe-inline'; "
            "img-src 'self' data: https:; "
            "font-src 'self' data:; "
            "connect-src 'self' ws: wss:; "
            "frame-ancestors 'none';"
        )
        response.headers["Content-Security-Policy"] = csp
        
        # HSTS (only in production)
        if settings.is_production():
            response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        
        return response


class InputValidationMiddleware(BaseHTTPMiddleware):
    """Input validation middleware"""
    
    def __init__(self, app: ASGIApp):
        super().__init__(app)
        self.suspicious_patterns = [
            r'<script[^>]*>.*?</script>',  # Script tags
            r'javascript:',  # JavaScript URLs
            r'on\w+\s*=',  # Event handlers
            r'<iframe[^>]*>.*?</iframe>',  # Iframe tags
            r'<object[^>]*>.*?</object>',  # Object tags
            r'<embed[^>]*>.*?</embed>',  # Embed tags
            r'<link[^>]*>.*?</link>',  # Link tags
            r'<meta[^>]*>.*?</meta>',  # Meta tags
            r'<style[^>]*>.*?</style>',  # Style tags
            r'<form[^>]*>.*?</form>',  # Form tags
            r'<input[^>]*>.*?</input>',  # Input tags
            r'<textarea[^>]*>.*?</textarea>',  # Textarea tags
            r'<select[^>]*>.*?</select>',  # Select tags
            r'<option[^>]*>.*?</option>',  # Option tags
            r'<button[^>]*>.*?</button>',  # Button tags
            r'<a[^>]*>.*?</a>',  # Anchor tags
            r'<img[^>]*>.*?</img>',  # Image tags
            r'<video[^>]*>.*?</video>',  # Video tags
            r'<audio[^>]*>.*?</audio>',  # Audio tags
            r'<source[^>]*>.*?</source>',  # Source tags
            r'<track[^>]*>.*?</track>',  # Track tags
            r'<canvas[^>]*>.*?</canvas>',  # Canvas tags
            r'<svg[^>]*>.*?</svg>',  # SVG tags
            r'<math[^>]*>.*?</math>',  # Math tags
            r'<table[^>]*>.*?</table>',  # Table tags
            r'<tr[^>]*>.*?</tr>',  # Table row tags
            r'<td[^>]*>.*?</td>',  # Table cell tags
            r'<th[^>]*>.*?</th>',  # Table header tags
            r'<thead[^>]*>.*?</thead>',  # Table head tags
            r'<tbody[^>]*>.*?</tbody>',  # Table body tags
            r'<tfoot[^>]*>.*?</tfoot>',  # Table foot tags
            r'<col[^>]*>.*?</col>',  # Column tags
            r'<colgroup[^>]*>.*?</colgroup>',  # Column group tags
            r'<caption[^>]*>.*?</caption>',  # Caption tags
            r'<div[^>]*>.*?</div>',  # Div tags
            r'<span[^>]*>.*?</span>',  # Span tags
            r'<p[^>]*>.*?</p>',  # Paragraph tags
            r'<h[1-6][^>]*>.*?</h[1-6]>',  # Heading tags
            r'<ul[^>]*>.*?</ul>',  # Unordered list tags
            r'<ol[^>]*>.*?</ol>',  # Ordered list tags
            r'<li[^>]*>.*?</li>',  # List item tags
            r'<dl[^>]*>.*?</dl>',  # Definition list tags
            r'<dt[^>]*>.*?</dt>',  # Definition term tags
            r'<dd[^>]*>.*?</dd>',  # Definition description tags
            r'<pre[^>]*>.*?</pre>',  # Preformatted text tags
            r'<code[^>]*>.*?</code>',  # Code tags
            r'<kbd[^>]*>.*?</kbd>',  # Keyboard input tags
            r'<samp[^>]*>.*?</samp>',  # Sample output tags
            r'<var[^>]*>.*?</var>',  # Variable tags
            r'<cite[^>]*>.*?</cite>',  # Citation tags
            r'<q[^>]*>.*?</q>',  # Quote tags
            r'<blockquote[^>]*>.*?</blockquote>',  # Block quote tags
            r'<address[^>]*>.*?</address>',  # Address tags
            r'<abbr[^>]*>.*?</abbr>',  # Abbreviation tags
            r'<acronym[^>]*>.*?</acronym>',  # Acronym tags
            r'<b[^>]*>.*?</b>',  # Bold tags
            r'<i[^>]*>.*?</i>',  # Italic tags
            r'<u[^>]*>.*?</u>',  # Underline tags
            r'<s[^>]*>.*?</s>',  # Strikethrough tags
            r'<strike[^>]*>.*?</strike>',  # Strikethrough tags
            r'<del[^>]*>.*?</del>',  # Deleted text tags
            r'<ins[^>]*>.*?</ins>',  # Inserted text tags
            r'<mark[^>]*>.*?</mark>',  # Marked text tags
            r'<small[^>]*>.*?</small>',  # Small text tags
            r'<big[^>]*>.*?</big>',  # Big text tags
            r'<sub[^>]*>.*?</sub>',  # Subscript tags
            r'<sup[^>]*>.*?</sup>',  # Superscript tags
            r'<tt[^>]*>.*?</tt>',  # Teletype tags
            r'<bdo[^>]*>.*?</bdo>',  # Bidirectional override tags
            r'<bdi[^>]*>.*?</bdi>',  # Bidirectional isolate tags
            r'<ruby[^>]*>.*?</ruby>',  # Ruby tags
            r'<rt[^>]*>.*?</rt>',  # Ruby text tags
            r'<rp[^>]*>.*?</rp>',  # Ruby parenthesis tags
            r'<wbr[^>]*>.*?</wbr>',  # Word break tags
            r'<br[^>]*>.*?</br>',  # Line break tags
            r'<hr[^>]*>.*?</hr>',  # Horizontal rule tags
            r'<area[^>]*>.*?</area>',  # Area tags
            r'<base[^>]*>.*?</base>',  # Base tags
            r'<basefont[^>]*>.*?</basefont>',  # Base font tags
            r'<bgsound[^>]*>.*?</bgsound>',  # Background sound tags
            r'<blink[^>]*>.*?</blink>',  # Blink tags
            r'<body[^>]*>.*?</body>',  # Body tags
            r'<center[^>]*>.*?</center>',  # Center tags
            r'<dir[^>]*>.*?</dir>',  # Directory list tags
            r'<font[^>]*>.*?</font>',  # Font tags
            r'<frame[^>]*>.*?</frame>',  # Frame tags
            r'<frameset[^>]*>.*?</frameset>',  # Frameset tags
            r'<head[^>]*>.*?</head>',  # Head tags
            r'<html[^>]*>.*?</html>',  # HTML tags
            r'<isindex[^>]*>.*?</isindex>',  # Isindex tags
            r'<keygen[^>]*>.*?</keygen>',  # Keygen tags
            r'<listing[^>]*>.*?</listing>',  # Listing tags
            r'<menu[^>]*>.*?</menu>',  # Menu tags
            r'<menuitem[^>]*>.*?</menuitem>',  # Menu item tags
            r'<nobr[^>]*>.*?</nobr>',  # No break tags
            r'<noframes[^>]*>.*?</noframes>',  # No frames tags
            r'<noscript[^>]*>.*?</noscript>',  # No script tags
            r'<param[^>]*>.*?</param>',  # Parameter tags
            r'<plaintext[^>]*>.*?</plaintext>',  # Plain text tags
            r'<samp[^>]*>.*?</samp>',  # Sample tags
            r'<xmp[^>]*>.*?</xmp>',  # XMP tags
        ]
    
    async def dispatch(self, request: Request, call_next):
        # Check for suspicious patterns in request body
        if request.method in ["POST", "PUT", "PATCH"]:
            body = await request.body()
            if body:
                body_str = body.decode('utf-8', errors='ignore')
                if self._contains_suspicious_patterns(body_str):
                    logger.warning(f"Suspicious input detected from {request.client.host}")
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail="Invalid input detected"
                    )
        
        response = await call_next(request)
        return response
    
    def _contains_suspicious_patterns(self, text: str) -> bool:
        """Check if text contains suspicious patterns"""
        for pattern in self.suspicious_patterns:
            if re.search(pattern, text, re.IGNORECASE | re.DOTALL):
                return True
        return False


class IPWhitelistMiddleware(BaseHTTPMiddleware):
    """IP whitelist middleware"""
    
    def __init__(self, app: ASGIApp, allowed_ips: List[str] = None):
        super().__init__(app)
        self.allowed_ips = allowed_ips or []
        self.allowed_networks = []
        
        # Parse IP addresses and networks
        for ip in self.allowed_ips:
            try:
                if '/' in ip:
                    self.allowed_networks.append(ipaddress.ip_network(ip, strict=False))
                else:
                    self.allowed_networks.append(ipaddress.ip_address(ip))
            except ValueError:
                logger.warning(f"Invalid IP address or network: {ip}")
    
    async def dispatch(self, request: Request, call_next):
        if self.allowed_ips:
            client_ip = self._get_client_ip(request)
            if not self._is_ip_allowed(client_ip):
                logger.warning(f"Blocked request from IP: {client_ip}")
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Access denied from this IP address"
                )
        
        response = await call_next(request)
        return response
    
    def _get_client_ip(self, request: Request) -> str:
        """Get client IP address"""
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip
        
        return request.client.host if request.client else "unknown"
    
    def _is_ip_allowed(self, client_ip: str) -> bool:
        """Check if client IP is allowed"""
        try:
            client_ip_obj = ipaddress.ip_address(client_ip)
            for allowed in self.allowed_networks:
                if isinstance(allowed, ipaddress.ip_network):
                    if client_ip_obj in allowed:
                        return True
                elif isinstance(allowed, ipaddress.ip_address):
                    if client_ip_obj == allowed:
                        return True
            return False
        except ValueError:
            return False


class SecurityAuditMiddleware(BaseHTTPMiddleware):
    """Security audit middleware"""
    
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        
        # Log request
        logger.info(f"Security audit: {request.method} {request.url} from {request.client.host}")
        
        # Process request
        response = await call_next(request)
        
        # Log response
        duration = time.time() - start_time
        logger.info(f"Security audit: {response.status_code} in {duration:.3f}s")
        
        # Add audit headers
        response.headers["X-Request-ID"] = self._generate_request_id(request)
        response.headers["X-Response-Time"] = f"{duration:.3f}s"
        
        return response
    
    def _generate_request_id(self, request: Request) -> str:
        """Generate unique request ID"""
        data = f"{request.url}{time.time()}{request.client.host}"
        return hashlib.md5(data.encode()).hexdigest()[:16]


class SecurityService:
    """Security service for additional security checks"""
    
    @staticmethod
    def validate_file_upload(filename: str, content_type: str, max_size: int = 10 * 1024 * 1024) -> bool:
        """Validate file upload"""
        # Check file extension
        allowed_extensions = ['.csv', '.json', '.xlsx', '.xls', '.txt', '.parquet']
        if not any(filename.lower().endswith(ext) for ext in allowed_extensions):
            return False
        
        # Check content type
        allowed_types = [
            'text/csv',
            'application/json',
            'application/vnd.ms-excel',
            'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            'text/plain',
            'application/octet-stream'
        ]
        if content_type not in allowed_types:
            return False
        
        return True
    
    @staticmethod
    def sanitize_filename(filename: str) -> str:
        """Sanitize filename for security"""
        # Remove path components
        filename = filename.split('/')[-1].split('\\')[-1]
        
        # Remove dangerous characters
        filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
        
        # Limit length
        if len(filename) > 255:
            name, ext = filename.rsplit('.', 1) if '.' in filename else (filename, '')
            filename = name[:255-len(ext)-1] + ('.' + ext if ext else '')
        
        return filename
    
    @staticmethod
    def validate_sql_query(query: str) -> bool:
        """Validate SQL query for security"""
        # Check for dangerous SQL keywords
        dangerous_keywords = [
            'DROP', 'DELETE', 'INSERT', 'UPDATE', 'ALTER', 'CREATE',
            'TRUNCATE', 'EXEC', 'EXECUTE', 'UNION', 'SELECT INTO'
        ]
        
        query_upper = query.upper()
        for keyword in dangerous_keywords:
            if keyword in query_upper:
                return False
        
        return True
    
    @staticmethod
    def generate_secure_token(length: int = 32) -> str:
        """Generate secure random token"""
        return secrets.token_urlsafe(length)
    
    @staticmethod
    def hash_sensitive_data(data: str) -> str:
        """Hash sensitive data"""
        return hashlib.sha256(data.encode()).hexdigest()


# Global security service
security_service = SecurityService()


def get_security_service() -> SecurityService:
    """Get the global security service"""
    return security_service
