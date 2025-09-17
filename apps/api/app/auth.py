"""
Authentication and Authorization system for Rocky AI
JWT-based authentication with role-based access control
"""
import jwt
import hashlib
import secrets
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
from passlib.context import CryptContext
from fastapi import HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from apps.api.app.config import get_settings
from apps.api.app.logging_config import get_logger
from apps.api.app.database import get_db, DatabaseService, User

logger = get_logger(__name__)
settings = get_settings()

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# JWT security
security = HTTPBearer()


class TokenData(BaseModel):
    """Token data model"""
    username: Optional[str] = None
    user_id: Optional[str] = None
    roles: List[str] = []
    permissions: List[str] = []


class UserCreate(BaseModel):
    """User creation model"""
    username: str
    email: str
    password: str
    full_name: Optional[str] = None


class UserLogin(BaseModel):
    """User login model"""
    username: str
    password: str


class UserResponse(BaseModel):
    """User response model"""
    id: str
    username: str
    email: str
    full_name: Optional[str] = None
    is_active: bool
    is_admin: bool
    created_at: datetime
    last_login: Optional[datetime] = None


class Role(BaseModel):
    """Role model"""
    name: str
    description: str
    permissions: List[str]


class Permission(BaseModel):
    """Permission model"""
    name: str
    description: str
    resource: str
    action: str


# Predefined roles and permissions
ROLES = {
    "admin": Role(
        name="admin",
        description="System administrator with full access",
        permissions=["*"]
    ),
    "analyst": Role(
        name="analyst",
        description="Data analyst with analysis permissions",
        permissions=[
            "analysis:create",
            "analysis:read",
            "analysis:update",
            "dataset:upload",
            "dataset:read",
            "results:read",
            "results:export"
        ]
    ),
    "viewer": Role(
        name="viewer",
        description="Read-only access to results and datasets",
        permissions=[
            "analysis:read",
            "dataset:read",
            "results:read"
        ]
    ),
    "guest": Role(
        name="guest",
        description="Limited access for guest users",
        permissions=[
            "analysis:read",
            "results:read"
        ]
    )
}

PERMISSIONS = {
    "analysis:create": Permission(
        name="analysis:create",
        description="Create new analyses",
        resource="analysis",
        action="create"
    ),
    "analysis:read": Permission(
        name="analysis:read",
        description="Read analysis results",
        resource="analysis",
        action="read"
    ),
    "analysis:update": Permission(
        name="analysis:update",
        description="Update analyses",
        resource="analysis",
        action="update"
    ),
    "analysis:delete": Permission(
        name="analysis:delete",
        description="Delete analyses",
        resource="analysis",
        action="delete"
    ),
    "dataset:upload": Permission(
        name="dataset:upload",
        description="Upload datasets",
        resource="dataset",
        action="upload"
    ),
    "dataset:read": Permission(
        name="dataset:read",
        description="Read datasets",
        resource="dataset",
        action="read"
    ),
    "dataset:delete": Permission(
        name="dataset:delete",
        description="Delete datasets",
        resource="dataset",
        action="delete"
    ),
    "results:read": Permission(
        name="results:read",
        description="Read analysis results",
        resource="results",
        action="read"
    ),
    "results:export": Permission(
        name="results:export",
        description="Export results",
        resource="results",
        action="export"
    ),
    "admin:users": Permission(
        name="admin:users",
        description="Manage users",
        resource="admin",
        action="users"
    ),
    "admin:system": Permission(
        name="admin:system",
        description="System administration",
        resource="admin",
        action="system"
    )
}


class AuthService:
    """Authentication and authorization service"""
    
    def __init__(self):
        self.secret_key = settings.security.secret_key
        self.algorithm = settings.security.algorithm
        self.access_token_expire_minutes = settings.security.access_token_expire_minutes
    
    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify a password against its hash"""
        return pwd_context.verify(plain_password, hashed_password)
    
    def get_password_hash(self, password: str) -> str:
        """Hash a password"""
        return pwd_context.hash(password)
    
    def create_access_token(self, data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
        """Create a JWT access token"""
        to_encode = data.copy()
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=self.access_token_expire_minutes)
        
        to_encode.update({"exp": expire})
        encoded_jwt = jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)
        return encoded_jwt
    
    def verify_token(self, token: str) -> TokenData:
        """Verify and decode a JWT token"""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            username: str = payload.get("sub")
            user_id: str = payload.get("user_id")
            roles: List[str] = payload.get("roles", [])
            permissions: List[str] = payload.get("permissions", [])
            
            if username is None:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Could not validate credentials",
                    headers={"WWW-Authenticate": "Bearer"},
                )
            
            return TokenData(
                username=username,
                user_id=user_id,
                roles=roles,
                permissions=permissions
            )
        except jwt.PyJWTError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Could not validate credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )
    
    async def authenticate_user(self, username: str, password: str, db: DatabaseService) -> Optional[User]:
        """Authenticate a user with username and password"""
        user = db.get_user_by_username(username)
        if not user:
            return None
        if not self.verify_password(password, user.hashed_password):
            return None
        return user
    
    async def create_user(self, user_data: UserCreate, db: DatabaseService) -> User:
        """Create a new user"""
        # Check if user already exists
        if db.get_user_by_username(user_data.username):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Username already registered"
            )
        
        if db.get_user_by_email(user_data.email):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email already registered"
            )
        
        # Hash password
        hashed_password = self.get_password_hash(user_data.password)
        
        # Create user
        user = db.create_user(
            username=user_data.username,
            email=user_data.email,
            hashed_password=hashed_password
        )
        
        logger.info(f"Created new user: {user.username}")
        return user
    
    def get_user_permissions(self, user: User) -> List[str]:
        """Get user permissions based on roles"""
        if user.is_admin:
            return ["*"]  # Admin has all permissions
        
        # For now, return basic permissions
        # In a full implementation, this would query user roles from database
        return ROLES["analyst"].permissions
    
    def check_permission(self, user_permissions: List[str], required_permission: str) -> bool:
        """Check if user has required permission"""
        if "*" in user_permissions:
            return True
        return required_permission in user_permissions
    
    def create_user_token(self, user: User) -> str:
        """Create a token for a user"""
        permissions = self.get_user_permissions(user)
        roles = ["admin"] if user.is_admin else ["analyst"]
        
        token_data = {
            "sub": user.username,
            "user_id": str(user.id),
            "roles": roles,
            "permissions": permissions,
            "iat": datetime.utcnow(),
            "exp": datetime.utcnow() + timedelta(minutes=self.access_token_expire_minutes)
        }
        
        return self.create_access_token(token_data)


# Global auth service
auth_service = AuthService()


def get_auth_service() -> AuthService:
    """Get the global auth service"""
    return auth_service


# Dependency functions
async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: DatabaseService = Depends(lambda: DatabaseService(next(get_db())))
) -> User:
    """Get current authenticated user"""
    token = credentials.credentials
    token_data = auth_service.verify_token(token)
    
    user = db.get_user_by_username(token_data.username)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    return user


def require_permission(permission: str):
    """Decorator to require specific permission"""
    def permission_checker(current_user: User = Depends(get_current_user)):
        user_permissions = auth_service.get_user_permissions(current_user)
        if not auth_service.check_permission(user_permissions, permission):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Insufficient permissions"
            )
        return current_user
    return permission_checker


def require_admin(current_user: User = Depends(get_current_user)) -> User:
    """Require admin role"""
    if not current_user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required"
        )
    return current_user


def require_active_user(current_user: User = Depends(get_current_user)) -> User:
    """Require active user"""
    if not current_user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Account is deactivated"
        )
    return current_user


# Optional authentication (for public endpoints)
async def get_current_user_optional(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(HTTPBearer(auto_error=False))
) -> Optional[User]:
    """Get current user if authenticated, None otherwise"""
    if not credentials:
        return None
    
    try:
        token = credentials.credentials
        token_data = auth_service.verify_token(token)
        
        db = DatabaseService(next(get_db()))
        user = db.get_user_by_username(token_data.username)
        return user
    except HTTPException:
        return None
