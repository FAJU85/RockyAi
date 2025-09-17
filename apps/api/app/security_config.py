"""
Security configuration and policies for Rocky AI
Password requirements, session management, and security constants
"""
import secrets
import string
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from enum import Enum
from pydantic import BaseModel, Field
from apps.api.app.config import get_settings

settings = get_settings()


class SecurityLevel(str, Enum):
    """Security levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class PasswordPolicy(BaseModel):
    """Password policy configuration"""
    min_length: int = Field(default=12, ge=8, le=128)
    require_uppercase: bool = True
    require_lowercase: bool = True
    require_digits: bool = True
    require_special_chars: bool = True
    special_chars: str = "!@#$%^&*()_+-=[]{}|;:,.<>?"
    max_length: int = Field(default=128, ge=8, le=128)
    prevent_common_passwords: bool = True
    prevent_user_info: bool = True
    max_attempts: int = Field(default=5, ge=1, le=10)
    lockout_duration_minutes: int = Field(default=30, ge=5, le=1440)
    password_history_count: int = Field(default=5, ge=0, le=20)


class SessionPolicy(BaseModel):
    """Session policy configuration"""
    max_duration_hours: int = Field(default=8, ge=1, le=24)
    idle_timeout_minutes: int = Field(default=30, ge=5, le=480)
    max_concurrent_sessions: int = Field(default=3, ge=1, le=10)
    require_reauth_for_sensitive: bool = True
    session_rotation_hours: int = Field(default=4, ge=1, le=24)
    secure_cookies: bool = True
    http_only_cookies: bool = True
    same_site_policy: str = Field(default="strict", regex="^(strict|lax|none)$")


class AccessControlPolicy(BaseModel):
    """Access control policy configuration"""
    default_role: str = "user"
    admin_roles: List[str] = ["admin", "super_admin"]
    moderator_roles: List[str] = ["moderator", "admin", "super_admin"]
    user_roles: List[str] = ["user", "moderator", "admin", "super_admin"]
    guest_roles: List[str] = ["guest", "user", "moderator", "admin", "super_admin"]
    role_hierarchy: Dict[str, List[str]] = {
        "super_admin": ["admin", "moderator", "user", "guest"],
        "admin": ["moderator", "user", "guest"],
        "moderator": ["user", "guest"],
        "user": ["guest"],
        "guest": []
    }


class DataClassificationPolicy(BaseModel):
    """Data classification policy"""
    public: List[str] = ["public", "open"]
    internal: List[str] = ["internal", "confidential"]
    restricted: List[str] = ["restricted", "secret"]
    top_secret: List[str] = ["top_secret", "classified"]
    
    classification_levels: Dict[str, int] = {
        "public": 1,
        "internal": 2,
        "restricted": 3,
        "top_secret": 4
    }
    
    access_requirements: Dict[str, List[str]] = {
        "public": ["user"],
        "internal": ["moderator", "admin", "super_admin"],
        "restricted": ["admin", "super_admin"],
        "top_secret": ["super_admin"]
    }


class SecurityConstants:
    """Security constants and configurations"""
    
    # JWT Configuration
    JWT_ALGORITHM = "HS256"
    JWT_ACCESS_TOKEN_EXPIRE_MINUTES = 30
    JWT_REFRESH_TOKEN_EXPIRE_DAYS = 7
    JWT_SECRET_KEY = settings.security.jwt_secret_key
    
    # Password Requirements
    PASSWORD_MIN_LENGTH = 12
    PASSWORD_MAX_LENGTH = 128
    PASSWORD_REQUIRE_UPPERCASE = True
    PASSWORD_REQUIRE_LOWERCASE = True
    PASSWORD_REQUIRE_DIGITS = True
    PASSWORD_REQUIRE_SPECIAL_CHARS = True
    PASSWORD_SPECIAL_CHARS = "!@#$%^&*()_+-=[]{}|;:,.<>?"
    
    # Session Configuration
    SESSION_MAX_DURATION_HOURS = 8
    SESSION_IDLE_TIMEOUT_MINUTES = 30
    SESSION_MAX_CONCURRENT = 3
    SESSION_ROTATION_HOURS = 4
    
    # Rate Limiting
    RATE_LIMIT_CALLS_PER_MINUTE = 60
    RATE_LIMIT_BURST_SIZE = 10
    RATE_LIMIT_WINDOW_MINUTES = 1
    
    # File Upload
    MAX_FILE_SIZE_MB = 10
    ALLOWED_FILE_EXTENSIONS = ['.csv', '.json', '.xlsx', '.xls', '.txt', '.parquet']
    ALLOWED_MIME_TYPES = [
        'text/csv',
        'application/json',
        'application/vnd.ms-excel',
        'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        'text/plain',
        'application/octet-stream'
    ]
    
    # Input Validation
    MAX_INPUT_LENGTH = 10000
    MAX_QUERY_LENGTH = 1000
    MAX_DATASET_ROWS = 1000000
    MAX_DATASET_COLUMNS = 1000
    
    # Security Headers
    SECURITY_HEADERS = {
        "X-Content-Type-Options": "nosniff",
        "X-Frame-Options": "DENY",
        "X-XSS-Protection": "1; mode=block",
        "Referrer-Policy": "strict-origin-when-cross-origin",
        "Permissions-Policy": "geolocation=(), microphone=(), camera=()",
        "Content-Security-Policy": (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline' 'unsafe-eval'; "
            "style-src 'self' 'unsafe-inline'; "
            "img-src 'self' data: https:; "
            "font-src 'self' data:; "
            "connect-src 'self' ws: wss:; "
            "frame-ancestors 'none';"
        )
    }
    
    # CORS Configuration
    CORS_ORIGINS = settings.security.allowed_origins
    CORS_METHODS = ["GET", "POST", "PUT", "DELETE", "OPTIONS"]
    CORS_HEADERS = ["*"]
    CORS_CREDENTIALS = True
    
    # IP Whitelist
    ALLOWED_IPS = settings.security.allowed_ips
    BLOCKED_IPS = settings.security.blocked_ips
    
    # Audit Logging
    AUDIT_LOG_LEVEL = "INFO"
    AUDIT_LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    AUDIT_LOG_FILE = "logs/security_audit.log"
    
    # Encryption
    ENCRYPTION_ALGORITHM = "AES-256-GCM"
    ENCRYPTION_KEY_LENGTH = 32
    HASH_ALGORITHM = "sha256"
    HASH_SALT_LENGTH = 32
    
    # Database Security
    DB_CONNECTION_POOL_SIZE = 20
    DB_CONNECTION_POOL_OVERFLOW = 30
    DB_CONNECTION_POOL_TIMEOUT = 30
    DB_CONNECTION_POOL_RECYCLE = 3600
    
    # Cache Security
    CACHE_TTL_SECONDS = 3600
    CACHE_MAX_MEMORY_MB = 100
    CACHE_EVICTION_POLICY = "lru"
    
    # Monitoring
    SECURITY_METRICS_ENABLED = True
    SECURITY_ALERTS_ENABLED = True
    SECURITY_DASHBOARD_ENABLED = True


class SecurityPolicyManager:
    """Security policy manager"""
    
    def __init__(self):
        self.password_policy = PasswordPolicy()
        self.session_policy = SessionPolicy()
        self.access_control = AccessControlPolicy()
        self.data_classification = DataClassificationPolicy()
        self.constants = SecurityConstants()
    
    def validate_password(self, password: str, user_info: Optional[Dict] = None) -> Dict[str, Any]:
        """Validate password against policy"""
        errors = []
        warnings = []
        
        # Length check
        if len(password) < self.password_policy.min_length:
            errors.append(f"Password must be at least {self.password_policy.min_length} characters long")
        elif len(password) > self.password_policy.max_length:
            errors.append(f"Password must be no more than {self.password_policy.max_length} characters long")
        
        # Character requirements
        if self.password_policy.require_uppercase and not any(c.isupper() for c in password):
            errors.append("Password must contain at least one uppercase letter")
        
        if self.password_policy.require_lowercase and not any(c.islower() for c in password):
            errors.append("Password must contain at least one lowercase letter")
        
        if self.password_policy.require_digits and not any(c.isdigit() for c in password):
            errors.append("Password must contain at least one digit")
        
        if self.password_policy.require_special_chars:
            if not any(c in self.password_policy.special_chars for c in password):
                errors.append(f"Password must contain at least one special character: {self.password_policy.special_chars}")
        
        # Common password check
        if self.password_policy.prevent_common_passwords:
            common_passwords = [
                "password", "123456", "123456789", "qwerty", "abc123",
                "password123", "admin", "letmein", "welcome", "monkey"
            ]
            if password.lower() in common_passwords:
                errors.append("Password is too common and easily guessable")
        
        # User info check
        if self.password_policy.prevent_user_info and user_info:
            for field in ["username", "email", "first_name", "last_name"]:
                if field in user_info and user_info[field]:
                    if user_info[field].lower() in password.lower():
                        errors.append("Password cannot contain personal information")
                        break
        
        # Strength calculation
        strength = self._calculate_password_strength(password)
        
        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
            "strength": strength
        }
    
    def _calculate_password_strength(self, password: str) -> int:
        """Calculate password strength (0-100)"""
        score = 0
        
        # Length score
        if len(password) >= 12:
            score += 20
        elif len(password) >= 8:
            score += 10
        
        # Character variety score
        if any(c.isupper() for c in password):
            score += 10
        if any(c.islower() for c in password):
            score += 10
        if any(c.isdigit() for c in password):
            score += 10
        if any(c in self.password_policy.special_chars for c in password):
            score += 10
        
        # Complexity score
        if len(set(password)) >= len(password) * 0.7:
            score += 20
        
        # Entropy score
        entropy = self._calculate_entropy(password)
        if entropy >= 4:
            score += 20
        elif entropy >= 3:
            score += 10
        
        return min(score, 100)
    
    def _calculate_entropy(self, password: str) -> float:
        """Calculate password entropy"""
        char_set_size = 0
        if any(c.islower() for c in password):
            char_set_size += 26
        if any(c.isupper() for c in password):
            char_set_size += 26
        if any(c.isdigit() for c in password):
            char_set_size += 10
        if any(c in self.password_policy.special_chars for c in password):
            char_set_size += len(self.password_policy.special_chars)
        
        if char_set_size == 0:
            return 0
        
        return len(password) * (char_set_size ** 0.5) / 10
    
    def generate_secure_password(self, length: int = None) -> str:
        """Generate secure password"""
        if length is None:
            length = self.password_policy.min_length
        
        # Ensure minimum requirements
        password = []
        
        # Add required character types
        if self.password_policy.require_lowercase:
            password.append(secrets.choice(string.ascii_lowercase))
        if self.password_policy.require_uppercase:
            password.append(secrets.choice(string.ascii_uppercase))
        if self.password_policy.require_digits:
            password.append(secrets.choice(string.digits))
        if self.password_policy.require_special_chars:
            password.append(secrets.choice(self.password_policy.special_chars))
        
        # Fill remaining length
        all_chars = string.ascii_letters + string.digits + self.password_policy.special_chars
        for _ in range(length - len(password)):
            password.append(secrets.choice(all_chars))
        
        # Shuffle password
        secrets.SystemRandom().shuffle(password)
        
        return ''.join(password)
    
    def validate_session_duration(self, duration_hours: int) -> bool:
        """Validate session duration"""
        return 1 <= duration_hours <= self.session_policy.max_duration_hours
    
    def validate_idle_timeout(self, timeout_minutes: int) -> bool:
        """Validate idle timeout"""
        return 5 <= timeout_minutes <= self.session_policy.idle_timeout_minutes
    
    def get_role_permissions(self, role: str) -> List[str]:
        """Get permissions for a role"""
        return self.access_control.role_hierarchy.get(role, [])
    
    def can_access_data(self, user_role: str, data_classification: str) -> bool:
        """Check if user can access data with given classification"""
        required_roles = self.data_classification.access_requirements.get(data_classification, [])
        return user_role in required_roles
    
    def get_security_level(self, data_classification: str) -> int:
        """Get security level for data classification"""
        return self.data_classification.classification_levels.get(data_classification, 1)


# Global security policy manager
security_policy_manager = SecurityPolicyManager()


def get_security_policy_manager() -> SecurityPolicyManager:
    """Get the global security policy manager"""
    return security_policy_manager


def get_security_constants() -> SecurityConstants:
    """Get security constants"""
    return SecurityConstants()
