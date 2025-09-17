"""
Security routes for Rocky AI
Authentication, authorization, and security management endpoints
"""
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from fastapi import APIRouter, Depends, HTTPException, status, Request, Response
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
from apps.api.app.auth import get_current_user, get_current_active_user, create_access_token, verify_password, get_password_hash
from apps.api.app.audit_logger import get_audit_logger, AuditEventType, AuditSeverity
from apps.api.app.security_config import get_security_policy_manager, get_security_constants
from apps.api.app.database import get_db, User, UserRole
from apps.api.app.logging_config import get_logger
from sqlalchemy.orm import Session

logger = get_logger(__name__)
security = HTTPBearer()
audit_logger = get_audit_logger()
security_policy = get_security_policy_manager()
security_constants = get_security_constants()

router = APIRouter(prefix="/security", tags=["security"])


# Request/Response Models
class LoginRequest(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    password: str = Field(..., min_length=1, max_length=128)


class LoginResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int
    user_id: str
    username: str
    role: str


class PasswordChangeRequest(BaseModel):
    current_password: str = Field(..., min_length=1, max_length=128)
    new_password: str = Field(..., min_length=8, max_length=128)


class PasswordChangeResponse(BaseModel):
    success: bool
    message: str


class PasswordResetRequest(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)


class PasswordResetResponse(BaseModel):
    success: bool
    message: str


class UserRoleRequest(BaseModel):
    user_id: str = Field(..., min_length=1)
    role: str = Field(..., min_length=1)


class UserRoleResponse(BaseModel):
    success: bool
    message: str
    user_id: str
    role: str


class SecurityStatusResponse(BaseModel):
    user_id: str
    username: str
    role: str
    is_active: bool
    last_login: Optional[datetime]
    failed_login_attempts: int
    is_locked: bool
    session_count: int
    security_level: str


class SecurityMetricsResponse(BaseModel):
    total_users: int
    active_sessions: int
    failed_login_attempts: int
    locked_accounts: int
    security_violations: int
    last_24h_events: int


# Authentication endpoints
@router.post("/login", response_model=LoginResponse)
async def login(
    request: LoginRequest,
    http_request: Request,
    db: Session = Depends(get_db)
):
    """Authenticate user and return access token"""
    try:
        # Get user from database
        user = db.query(User).filter(User.username == request.username).first()
        
        if not user:
            audit_logger.log_authentication_event(
                event_type=AuditEventType.LOGIN_FAILURE,
                user_id="unknown",
                ip_address=http_request.client.host,
                user_agent=http_request.headers.get("user-agent", ""),
                success=False,
                failure_reason="user_not_found"
            )
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid username or password"
            )
        
        # Check if account is locked
        if user.is_locked:
            audit_logger.log_authentication_event(
                event_type=AuditEventType.LOGIN_FAILURE,
                user_id=user.id,
                ip_address=http_request.client.host,
                user_agent=http_request.headers.get("user-agent", ""),
                success=False,
                failure_reason="account_locked"
            )
            raise HTTPException(
                status_code=status.HTTP_423_LOCKED,
                detail="Account is locked due to too many failed login attempts"
            )
        
        # Verify password
        if not verify_password(request.password, user.hashed_password):
            # Increment failed login attempts
            user.failed_login_attempts += 1
            if user.failed_login_attempts >= security_constants.PASSWORD_MAX_ATTEMPTS:
                user.is_locked = True
                user.locked_until = datetime.utcnow() + timedelta(minutes=security_constants.PASSWORD_LOCKOUT_DURATION)
                audit_logger.log_authentication_event(
                    event_type=AuditEventType.ACCOUNT_LOCKED,
                    user_id=user.id,
                    ip_address=http_request.client.host,
                    user_agent=http_request.headers.get("user-agent", ""),
                    success=False,
                    failure_reason="too_many_failed_attempts"
                )
            else:
                audit_logger.log_authentication_event(
                    event_type=AuditEventType.LOGIN_FAILURE,
                    user_id=user.id,
                    ip_address=http_request.client.host,
                    user_agent=http_request.headers.get("user-agent", ""),
                    success=False,
                    failure_reason="invalid_password"
                )
            
            db.commit()
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid username or password"
            )
        
        # Reset failed login attempts on successful login
        user.failed_login_attempts = 0
        user.is_locked = False
        user.locked_until = None
        user.last_login = datetime.utcnow()
        db.commit()
        
        # Create access token
        access_token = create_access_token(data={"sub": user.username})
        
        # Log successful login
        audit_logger.log_authentication_event(
            event_type=AuditEventType.LOGIN_SUCCESS,
            user_id=user.id,
            ip_address=http_request.client.host,
            user_agent=http_request.headers.get("user-agent", ""),
            success=True
        )
        
        return LoginResponse(
            access_token=access_token,
            expires_in=security_constants.JWT_ACCESS_TOKEN_EXPIRE_MINUTES * 60,
            user_id=user.id,
            username=user.username,
            role=user.role.value
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Login error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error during login"
        )


@router.post("/logout")
async def logout(
    current_user: User = Depends(get_current_active_user),
    http_request: Request = None
):
    """Logout user and invalidate session"""
    try:
        # Log logout event
        audit_logger.log_authentication_event(
            event_type=AuditEventType.LOGOUT,
            user_id=current_user.id,
            ip_address=http_request.client.host if http_request else None,
            user_agent=http_request.headers.get("user-agent", "") if http_request else None,
            success=True
        )
        
        return {"message": "Successfully logged out"}
        
    except Exception as e:
        logger.error(f"Logout error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error during logout"
        )


@router.post("/change-password", response_model=PasswordChangeResponse)
async def change_password(
    request: PasswordChangeRequest,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Change user password"""
    try:
        # Verify current password
        if not verify_password(request.current_password, current_user.hashed_password):
            audit_logger.log_authentication_event(
                event_type=AuditEventType.PASSWORD_CHANGE,
                user_id=current_user.id,
                ip_address=None,
                user_agent=None,
                success=False,
                failure_reason="invalid_current_password"
            )
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Current password is incorrect"
            )
        
        # Validate new password
        password_validation = security_policy.validate_password(
            request.new_password,
            {"username": current_user.username, "email": current_user.email}
        )
        
        if not password_validation["valid"]:
            audit_logger.log_authentication_event(
                event_type=AuditEventType.PASSWORD_CHANGE,
                user_id=current_user.id,
                ip_address=None,
                user_agent=None,
                success=False,
                failure_reason="password_validation_failed"
            )
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Password validation failed: {', '.join(password_validation['errors'])}"
            )
        
        # Update password
        current_user.hashed_password = get_password_hash(request.new_password)
        current_user.password_changed_at = datetime.utcnow()
        db.commit()
        
        # Log successful password change
        audit_logger.log_authentication_event(
            event_type=AuditEventType.PASSWORD_CHANGE,
            user_id=current_user.id,
            ip_address=None,
            user_agent=None,
            success=True
        )
        
        return PasswordChangeResponse(
            success=True,
            message="Password changed successfully"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Password change error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error during password change"
        )


@router.post("/reset-password", response_model=PasswordResetResponse)
async def reset_password(
    request: PasswordResetRequest,
    db: Session = Depends(get_db)
):
    """Reset user password (admin only)"""
    try:
        # Get user from database
        user = db.query(User).filter(User.username == request.username).first()
        
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        # Generate new password
        new_password = security_policy.generate_secure_password()
        user.hashed_password = get_password_hash(new_password)
        user.password_changed_at = datetime.utcnow()
        user.is_locked = False
        user.failed_login_attempts = 0
        db.commit()
        
        # Log password reset
        audit_logger.log_authentication_event(
            event_type=AuditEventType.PASSWORD_RESET,
            user_id=user.id,
            ip_address=None,
            user_agent=None,
            success=True
        )
        
        return PasswordResetResponse(
            success=True,
            message=f"Password reset successfully. New password: {new_password}"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Password reset error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error during password reset"
        )


# Authorization endpoints
@router.post("/assign-role", response_model=UserRoleResponse)
async def assign_role(
    request: UserRoleRequest,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Assign role to user (admin only)"""
    try:
        # Check if current user is admin
        if current_user.role not in [UserRole.ADMIN, UserRole.SUPER_ADMIN]:
            audit_logger.log_authorization_event(
                event_type=AuditEventType.PERMISSION_DENIED,
                user_id=current_user.id,
                resource="user_roles",
                action="assign_role",
                success=False
            )
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Insufficient permissions to assign roles"
            )
        
        # Get target user
        target_user = db.query(User).filter(User.id == request.user_id).first()
        if not target_user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        # Validate role
        try:
            new_role = UserRole(request.role)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid role"
            )
        
        # Update user role
        old_role = target_user.role
        target_user.role = new_role
        db.commit()
        
        # Log role assignment
        audit_logger.log_authorization_event(
            event_type=AuditEventType.ROLE_ASSIGNED,
            user_id=current_user.id,
            resource="user_roles",
            action="assign_role",
            success=True
        )
        
        return UserRoleResponse(
            success=True,
            message=f"Role assigned successfully",
            user_id=target_user.id,
            role=new_role.value
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Role assignment error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error during role assignment"
        )


# Security status endpoints
@router.get("/status", response_model=SecurityStatusResponse)
async def get_security_status(
    current_user: User = Depends(get_current_active_user)
):
    """Get current user's security status"""
    try:
        return SecurityStatusResponse(
            user_id=current_user.id,
            username=current_user.username,
            role=current_user.role.value,
            is_active=current_user.is_active,
            last_login=current_user.last_login,
            failed_login_attempts=current_user.failed_login_attempts,
            is_locked=current_user.is_locked,
            session_count=0,  # This would be calculated from active sessions
            security_level="medium"  # This would be calculated based on user's role and permissions
        )
        
    except Exception as e:
        logger.error(f"Security status error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error while getting security status"
        )


@router.get("/metrics", response_model=SecurityMetricsResponse)
async def get_security_metrics(
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get security metrics (admin only)"""
    try:
        # Check if current user is admin
        if current_user.role not in [UserRole.ADMIN, UserRole.SUPER_ADMIN]:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Insufficient permissions to view security metrics"
            )
        
        # Calculate metrics
        total_users = db.query(User).count()
        active_sessions = 0  # This would be calculated from active sessions
        failed_login_attempts = db.query(User).filter(User.failed_login_attempts > 0).count()
        locked_accounts = db.query(User).filter(User.is_locked == True).count()
        security_violations = 0  # This would be calculated from audit logs
        last_24h_events = 0  # This would be calculated from audit logs
        
        return SecurityMetricsResponse(
            total_users=total_users,
            active_sessions=active_sessions,
            failed_login_attempts=failed_login_attempts,
            locked_accounts=locked_accounts,
            security_violations=security_violations,
            last_24h_events=last_24h_events
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Security metrics error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error while getting security metrics"
        )


# Security policy endpoints
@router.get("/policy/password")
async def get_password_policy():
    """Get password policy"""
    try:
        return {
            "min_length": security_policy.password_policy.min_length,
            "max_length": security_policy.password_policy.max_length,
            "require_uppercase": security_policy.password_policy.require_uppercase,
            "require_lowercase": security_policy.password_policy.require_lowercase,
            "require_digits": security_policy.password_policy.require_digits,
            "require_special_chars": security_policy.password_policy.require_special_chars,
            "special_chars": security_policy.password_policy.special_chars,
            "prevent_common_passwords": security_policy.password_policy.prevent_common_passwords,
            "prevent_user_info": security_policy.password_policy.prevent_user_info,
            "max_attempts": security_policy.password_policy.max_attempts,
            "lockout_duration_minutes": security_policy.password_policy.lockout_duration_minutes
        }
        
    except Exception as e:
        logger.error(f"Password policy error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error while getting password policy"
        )


@router.get("/policy/session")
async def get_session_policy():
    """Get session policy"""
    try:
        return {
            "max_duration_hours": security_policy.session_policy.max_duration_hours,
            "idle_timeout_minutes": security_policy.session_policy.idle_timeout_minutes,
            "max_concurrent_sessions": security_policy.session_policy.max_concurrent_sessions,
            "require_reauth_for_sensitive": security_policy.session_policy.require_reauth_for_sensitive,
            "session_rotation_hours": security_policy.session_policy.session_rotation_hours,
            "secure_cookies": security_policy.session_policy.secure_cookies,
            "http_only_cookies": security_policy.session_policy.http_only_cookies,
            "same_site_policy": security_policy.session_policy.same_site_policy
        }
        
    except Exception as e:
        logger.error(f"Session policy error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error while getting session policy"
        )


# Audit log endpoints
@router.get("/audit/events")
async def get_audit_events(
    event_type: Optional[str] = None,
    severity: Optional[str] = None,
    user_id: Optional[str] = None,
    limit: int = 100,
    current_user: User = Depends(get_current_active_user)
):
    """Get audit events (admin only)"""
    try:
        # Check if current user is admin
        if current_user.role not in [UserRole.ADMIN, UserRole.SUPER_ADMIN]:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Insufficient permissions to view audit events"
            )
        
        # This would typically query the audit log database
        # For now, return empty list
        return {
            "events": [],
            "total": 0,
            "limit": limit
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Audit events error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error while getting audit events"
        )


@router.get("/audit/report")
async def get_audit_report(
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
    report_type: str = "general",
    current_user: User = Depends(get_current_active_user)
):
    """Get audit report (admin only)"""
    try:
        # Check if current user is admin
        if current_user.role not in [UserRole.ADMIN, UserRole.SUPER_ADMIN]:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Insufficient permissions to view audit reports"
            )
        
        # Set default time range if not provided
        if not start_time:
            start_time = datetime.utcnow() - timedelta(days=7)
        if not end_time:
            end_time = datetime.utcnow()
        
        # Generate audit report
        report = audit_logger.generate_compliance_report(start_time, end_time, report_type)
        
        return report
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Audit report error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error while generating audit report"
        )
