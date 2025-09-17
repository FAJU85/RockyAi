"""
Audit logging system for Rocky AI
Comprehensive security and compliance logging
"""
import json
import hashlib
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Union
from enum import Enum
from pydantic import BaseModel, Field
from apps.api.app.logging_config import get_logger
from apps.api.app.config import get_settings

logger = get_logger(__name__)
settings = get_settings()


class AuditEventType(str, Enum):
    """Audit event types"""
    # Authentication events
    LOGIN_SUCCESS = "login_success"
    LOGIN_FAILURE = "login_failure"
    LOGOUT = "logout"
    PASSWORD_CHANGE = "password_change"
    PASSWORD_RESET = "password_reset"
    ACCOUNT_LOCKED = "account_locked"
    ACCOUNT_UNLOCKED = "account_unlocked"
    
    # Authorization events
    PERMISSION_GRANTED = "permission_granted"
    PERMISSION_DENIED = "permission_denied"
    ROLE_ASSIGNED = "role_assigned"
    ROLE_REVOKED = "role_revoked"
    ACCESS_GRANTED = "access_granted"
    ACCESS_DENIED = "access_denied"
    
    # Data access events
    DATA_VIEWED = "data_viewed"
    DATA_CREATED = "data_created"
    DATA_UPDATED = "data_updated"
    DATA_DELETED = "data_deleted"
    DATA_EXPORTED = "data_exported"
    DATA_IMPORTED = "data_imported"
    
    # Analysis events
    ANALYSIS_STARTED = "analysis_started"
    ANALYSIS_COMPLETED = "analysis_completed"
    ANALYSIS_FAILED = "analysis_failed"
    ANALYSIS_CANCELLED = "analysis_cancelled"
    
    # System events
    SYSTEM_STARTUP = "system_startup"
    SYSTEM_SHUTDOWN = "system_shutdown"
    CONFIGURATION_CHANGED = "configuration_changed"
    SECURITY_POLICY_UPDATED = "security_policy_updated"
    
    # Security events
    SECURITY_VIOLATION = "security_violation"
    SUSPICIOUS_ACTIVITY = "suspicious_activity"
    RATE_LIMIT_EXCEEDED = "rate_limit_exceeded"
    INVALID_INPUT = "invalid_input"
    FILE_UPLOAD_BLOCKED = "file_upload_blocked"
    
    # Compliance events
    DATA_RETENTION_POLICY_APPLIED = "data_retention_policy_applied"
    DATA_ANONYMIZATION = "data_anonymization"
    BACKUP_CREATED = "backup_created"
    BACKUP_RESTORED = "backup_restored"


class AuditSeverity(str, Enum):
    """Audit severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AuditEvent(BaseModel):
    """Audit event model"""
    event_id: str = Field(..., description="Unique event identifier")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    event_type: AuditEventType = Field(..., description="Type of audit event")
    severity: AuditSeverity = Field(..., description="Severity level")
    user_id: Optional[str] = Field(None, description="User ID if applicable")
    session_id: Optional[str] = Field(None, description="Session ID if applicable")
    ip_address: Optional[str] = Field(None, description="Client IP address")
    user_agent: Optional[str] = Field(None, description="Client user agent")
    resource: Optional[str] = Field(None, description="Resource being accessed")
    action: Optional[str] = Field(None, description="Action performed")
    result: str = Field(..., description="Result of the action")
    details: Dict[str, Any] = Field(default_factory=dict, description="Additional event details")
    risk_score: Optional[int] = Field(None, ge=0, le=100, description="Risk score (0-100)")
    correlation_id: Optional[str] = Field(None, description="Correlation ID for related events")
    tags: List[str] = Field(default_factory=list, description="Event tags for categorization")


class AuditLogger:
    """Audit logger for security and compliance"""
    
    def __init__(self):
        self.logger = logger
        self.settings = settings
        self._event_buffer = []
        self._buffer_size = 100
        self._flush_interval = 30  # seconds
    
    def log_event(
        self,
        event_type: AuditEventType,
        severity: AuditSeverity,
        result: str,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        resource: Optional[str] = None,
        action: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        risk_score: Optional[int] = None,
        correlation_id: Optional[str] = None,
        tags: Optional[List[str]] = None
    ) -> str:
        """Log an audit event"""
        event_id = self._generate_event_id()
        
        event = AuditEvent(
            event_id=event_id,
            event_type=event_type,
            severity=severity,
            user_id=user_id,
            session_id=session_id,
            ip_address=ip_address,
            user_agent=user_agent,
            resource=resource,
            action=action,
            result=result,
            details=details or {},
            risk_score=risk_score,
            correlation_id=correlation_id,
            tags=tags or []
        )
        
        # Add to buffer
        self._event_buffer.append(event)
        
        # Log to standard logger
        self._log_to_standard_logger(event)
        
        # Flush buffer if needed
        if len(self._event_buffer) >= self._buffer_size:
            self._flush_buffer()
        
        return event_id
    
    def _generate_event_id(self) -> str:
        """Generate unique event ID"""
        timestamp = datetime.now(timezone.utc).isoformat()
        random_data = f"{timestamp}{id(self)}"
        return hashlib.sha256(random_data.encode()).hexdigest()[:16]
    
    def _log_to_standard_logger(self, event: AuditEvent):
        """Log event to standard logger"""
        log_data = {
            "event_id": event.event_id,
            "event_type": event.event_type.value,
            "severity": event.severity.value,
            "timestamp": event.timestamp.isoformat(),
            "user_id": event.user_id,
            "session_id": event.session_id,
            "ip_address": event.ip_address,
            "resource": event.resource,
            "action": event.action,
            "result": event.result,
            "risk_score": event.risk_score,
            "correlation_id": event.correlation_id,
            "tags": event.tags,
            "details": event.details
        }
        
        # Log based on severity
        if event.severity == AuditSeverity.CRITICAL:
            self.logger.critical(f"AUDIT: {json.dumps(log_data)}")
        elif event.severity == AuditSeverity.HIGH:
            self.logger.error(f"AUDIT: {json.dumps(log_data)}")
        elif event.severity == AuditSeverity.MEDIUM:
            self.logger.warning(f"AUDIT: {json.dumps(log_data)}")
        else:
            self.logger.info(f"AUDIT: {json.dumps(log_data)}")
    
    def _flush_buffer(self):
        """Flush event buffer"""
        if not self._event_buffer:
            return
        
        # Here you would typically send to external audit system
        # For now, we'll just clear the buffer
        self._event_buffer.clear()
    
    def log_authentication_event(
        self,
        event_type: AuditEventType,
        user_id: str,
        ip_address: str,
        user_agent: str,
        success: bool,
        failure_reason: Optional[str] = None
    ) -> str:
        """Log authentication event"""
        severity = AuditSeverity.HIGH if not success else AuditSeverity.MEDIUM
        result = "success" if success else f"failure: {failure_reason}"
        
        return self.log_event(
            event_type=event_type,
            severity=severity,
            result=result,
            user_id=user_id,
            ip_address=ip_address,
            user_agent=user_agent,
            resource="authentication",
            action="authenticate",
            details={"failure_reason": failure_reason} if failure_reason else {},
            risk_score=80 if not success else 20,
            tags=["authentication", "security"]
        )
    
    def log_authorization_event(
        self,
        event_type: AuditEventType,
        user_id: str,
        resource: str,
        action: str,
        success: bool,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None
    ) -> str:
        """Log authorization event"""
        severity = AuditSeverity.HIGH if not success else AuditSeverity.MEDIUM
        result = "success" if success else "denied"
        
        return self.log_event(
            event_type=event_type,
            severity=severity,
            result=result,
            user_id=user_id,
            ip_address=ip_address,
            user_agent=user_agent,
            resource=resource,
            action=action,
            details={"authorized": success},
            risk_score=90 if not success else 10,
            tags=["authorization", "access_control"]
        )
    
    def log_data_access_event(
        self,
        event_type: AuditEventType,
        user_id: str,
        resource: str,
        action: str,
        data_classification: str,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ) -> str:
        """Log data access event"""
        severity = AuditSeverity.MEDIUM
        result = "success"
        
        return self.log_event(
            event_type=event_type,
            severity=severity,
            result=result,
            user_id=user_id,
            ip_address=ip_address,
            user_agent=user_agent,
            resource=resource,
            action=action,
            details={
                "data_classification": data_classification,
                **(details or {})
            },
            risk_score=30,
            tags=["data_access", "privacy"]
        )
    
    def log_analysis_event(
        self,
        event_type: AuditEventType,
        user_id: str,
        analysis_id: str,
        success: bool,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ) -> str:
        """Log analysis event"""
        severity = AuditSeverity.HIGH if not success else AuditSeverity.MEDIUM
        result = "success" if success else "failed"
        
        return self.log_event(
            event_type=event_type,
            severity=severity,
            result=result,
            user_id=user_id,
            ip_address=ip_address,
            user_agent=user_agent,
            resource="analysis",
            action="execute",
            details={
                "analysis_id": analysis_id,
                **(details or {})
            },
            risk_score=40 if not success else 20,
            tags=["analysis", "execution"]
        )
    
    def log_security_event(
        self,
        event_type: AuditEventType,
        severity: AuditSeverity,
        description: str,
        user_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        resource: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ) -> str:
        """Log security event"""
        return self.log_event(
            event_type=event_type,
            severity=severity,
            result=description,
            user_id=user_id,
            ip_address=ip_address,
            user_agent=user_agent,
            resource=resource,
            action="security_check",
            details=details or {},
            risk_score=80 if severity == AuditSeverity.CRITICAL else 60,
            tags=["security", "violation"]
        )
    
    def log_system_event(
        self,
        event_type: AuditEventType,
        description: str,
        details: Optional[Dict[str, Any]] = None
    ) -> str:
        """Log system event"""
        return self.log_event(
            event_type=event_type,
            severity=AuditSeverity.MEDIUM,
            result=description,
            resource="system",
            action="system_operation",
            details=details or {},
            risk_score=20,
            tags=["system", "operation"]
        )
    
    def get_events_by_user(self, user_id: str, limit: int = 100) -> List[AuditEvent]:
        """Get audit events for a specific user"""
        # This would typically query a database
        # For now, return empty list
        return []
    
    def get_events_by_type(self, event_type: AuditEventType, limit: int = 100) -> List[AuditEvent]:
        """Get audit events by type"""
        # This would typically query a database
        # For now, return empty list
        return []
    
    def get_events_by_severity(self, severity: AuditSeverity, limit: int = 100) -> List[AuditEvent]:
        """Get audit events by severity"""
        # This would typically query a database
        # For now, return empty list
        return []
    
    def get_events_by_time_range(
        self,
        start_time: datetime,
        end_time: datetime,
        limit: int = 100
    ) -> List[AuditEvent]:
        """Get audit events by time range"""
        # This would typically query a database
        # For now, return empty list
        return []
    
    def search_events(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        limit: int = 100
    ) -> List[AuditEvent]:
        """Search audit events"""
        # This would typically query a database
        # For now, return empty list
        return []
    
    def generate_compliance_report(
        self,
        start_time: datetime,
        end_time: datetime,
        report_type: str = "general"
    ) -> Dict[str, Any]:
        """Generate compliance report"""
        # This would typically generate a comprehensive report
        # For now, return basic structure
        return {
            "report_type": report_type,
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "total_events": 0,
            "events_by_type": {},
            "events_by_severity": {},
            "security_violations": 0,
            "data_access_events": 0,
            "authentication_events": 0,
            "authorization_events": 0,
            "analysis_events": 0,
            "system_events": 0
        }


# Global audit logger
audit_logger = AuditLogger()


def get_audit_logger() -> AuditLogger:
    """Get the global audit logger"""
    return audit_logger
