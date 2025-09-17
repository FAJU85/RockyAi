"""
Test configuration and fixtures for Rocky AI
"""
import asyncio
import os
import tempfile
from typing import AsyncGenerator, Generator
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from apps.api.app.main import app
from apps.api.app.database import get_db, Base
from apps.api.app.config import get_settings
from apps.api.app.cache import get_cache
from apps.api.app.orchestrator import get_orchestrator
from apps.api.app.websocket import get_connection_manager
from apps.api.app.metrics import get_metrics_collector
from apps.api.app.audit_logger import get_audit_logger
from apps.api.app.security_config import get_security_policy_manager
from apps.api.app.database import User, Analysis, Dataset, UserRole


# Test database URL
TEST_DATABASE_URL = "sqlite:///./test.db"

# Test settings
@pytest.fixture(scope="session")
def test_settings():
    """Test settings fixture"""
    return {
        "database_url": TEST_DATABASE_URL,
        "redis_url": "redis://localhost:6379/1",
        "dmr_url": "http://localhost:8001",
        "jwt_secret_key": "test-secret-key",
        "jwt_algorithm": "HS256",
        "jwt_access_token_expire_minutes": 30,
        "security_allowed_origins": ["http://localhost:3000"],
        "security_allowed_ips": [],
        "security_blocked_ips": [],
        "is_production": False,
        "log_level": "DEBUG"
    }


# Database fixtures
@pytest.fixture(scope="session")
def test_engine():
    """Test database engine"""
    engine = create_engine(
        TEST_DATABASE_URL,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    return engine


@pytest.fixture(scope="session")
def test_db_session(test_engine):
    """Test database session"""
    Base.metadata.create_all(bind=test_engine)
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)
    session = TestingSessionLocal()
    yield session
    session.close()
    Base.metadata.drop_all(bind=test_engine)


@pytest.fixture
def db_session(test_db_session):
    """Database session for individual tests"""
    yield test_db_session
    test_db_session.rollback()


@pytest.fixture
def override_get_db(db_session):
    """Override database dependency"""
    def _override_get_db():
        try:
            yield db_session
        finally:
            pass
    return _override_get_db


# Cache fixtures
@pytest.fixture
def mock_cache():
    """Mock cache for testing"""
    cache = MagicMock()
    cache.get.return_value = None
    cache.set.return_value = True
    cache.delete.return_value = True
    cache.exists.return_value = False
    cache.keys.return_value = []
    cache.flushdb.return_value = True
    return cache


@pytest.fixture
def override_get_cache(mock_cache):
    """Override cache dependency"""
    def _override_get_cache():
        return mock_cache
    return _override_get_cache


# Orchestrator fixtures
@pytest.fixture
def mock_orchestrator():
    """Mock orchestrator for testing"""
    orchestrator = AsyncMock()
    orchestrator.execute_analysis.return_value = {
        "analysis_id": "test-analysis-123",
        "status": "completed",
        "result": "Test analysis result",
        "plan": "Test analysis plan",
        "code": "print('Test analysis code')",
        "output": "Test analysis output",
        "error": None,
        "execution_time": 1.5
    }
    orchestrator.initialize.return_value = None
    orchestrator.close.return_value = None
    return orchestrator


@pytest.fixture
def override_get_orchestrator(mock_orchestrator):
    """Override orchestrator dependency"""
    def _override_get_orchestrator():
        return mock_orchestrator
    return _override_get_orchestrator


# WebSocket fixtures
@pytest.fixture
def mock_connection_manager():
    """Mock WebSocket connection manager"""
    manager = MagicMock()
    manager.connect.return_value = None
    manager.disconnect.return_value = None
    manager.send_personal_message.return_value = None
    manager.broadcast.return_value = None
    manager.get_connection_count.return_value = 0
    return manager


@pytest.fixture
def override_get_connection_manager(mock_connection_manager):
    """Override connection manager dependency"""
    def _override_get_connection_manager():
        return mock_connection_manager
    return _override_get_connection_manager


# Metrics fixtures
@pytest.fixture
def mock_metrics_collector():
    """Mock metrics collector for testing"""
    collector = MagicMock()
    collector.record_analysis_request.return_value = None
    collector.record_analysis_error.return_value = None
    collector.record_execution_time.return_value = None
    collector.get_metrics.return_value = {}
    return collector


@pytest.fixture
def override_get_metrics_collector(mock_metrics_collector):
    """Override metrics collector dependency"""
    def _override_get_metrics_collector():
        return mock_metrics_collector
    return _override_get_metrics_collector


# Audit logger fixtures
@pytest.fixture
def mock_audit_logger():
    """Mock audit logger for testing"""
    logger = MagicMock()
    logger.log_event.return_value = "test-event-id"
    logger.log_authentication_event.return_value = "test-auth-event-id"
    logger.log_authorization_event.return_value = "test-authz-event-id"
    logger.log_data_access_event.return_value = "test-data-event-id"
    logger.log_analysis_event.return_value = "test-analysis-event-id"
    logger.log_security_event.return_value = "test-security-event-id"
    logger.log_system_event.return_value = "test-system-event-id"
    return logger


@pytest.fixture
def override_get_audit_logger(mock_audit_logger):
    """Override audit logger dependency"""
    def _override_get_audit_logger():
        return mock_audit_logger
    return _override_get_audit_logger


# Security fixtures
@pytest.fixture
def mock_security_policy():
    """Mock security policy manager for testing"""
    policy = MagicMock()
    policy.validate_password.return_value = {
        "valid": True,
        "errors": [],
        "warnings": [],
        "strength": 85
    }
    policy.generate_secure_password.return_value = "TestPassword123!"
    policy.validate_session_duration.return_value = True
    policy.validate_idle_timeout.return_value = True
    policy.get_role_permissions.return_value = ["user", "guest"]
    policy.can_access_data.return_value = True
    policy.get_security_level.return_value = 2
    return policy


@pytest.fixture
def override_get_security_policy(mock_security_policy):
    """Override security policy dependency"""
    def _override_get_security_policy():
        return mock_security_policy
    return _override_get_security_policy


# Test client fixtures
@pytest.fixture
def client(
    override_get_db,
    override_get_cache,
    override_get_orchestrator,
    override_get_connection_manager,
    override_get_metrics_collector,
    override_get_audit_logger,
    override_get_security_policy
):
    """Test client with mocked dependencies"""
    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[get_cache] = override_get_cache
    app.dependency_overrides[get_orchestrator] = override_get_orchestrator
    app.dependency_overrides[get_connection_manager] = override_get_connection_manager
    app.dependency_overrides[get_metrics_collector] = override_get_metrics_collector
    app.dependency_overrides[get_audit_logger] = override_get_audit_logger
    app.dependency_overrides[get_security_policy_manager] = override_get_security_policy
    
    with TestClient(app) as test_client:
        yield test_client
    
    # Clean up overrides
    app.dependency_overrides.clear()


# User fixtures
@pytest.fixture
def test_user(db_session):
    """Create test user"""
    user = User(
        username="testuser",
        email="test@example.com",
        hashed_password="$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewdBPj4J/4X5Q5K2",  # "testpassword"
        role=UserRole.USER,
        is_active=True,
        failed_login_attempts=0,
        is_locked=False
    )
    db_session.add(user)
    db_session.commit()
    db_session.refresh(user)
    return user


@pytest.fixture
def test_admin_user(db_session):
    """Create test admin user"""
    user = User(
        username="admin",
        email="admin@example.com",
        hashed_password="$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewdBPj4J/4X5Q5K2",  # "testpassword"
        role=UserRole.ADMIN,
        is_active=True,
        failed_login_attempts=0,
        is_locked=False
    )
    db_session.add(user)
    db_session.commit()
    db_session.refresh(user)
    return user


@pytest.fixture
def test_analysis(db_session, test_user):
    """Create test analysis"""
    analysis = Analysis(
        id="test-analysis-123",
        user_id=test_user.id,
        query="Test analysis query",
        status="completed",
        result="Test analysis result",
        plan="Test analysis plan",
        code="print('Test analysis code')",
        output="Test analysis output",
        error=None,
        execution_time=1.5
    )
    db_session.add(analysis)
    db_session.commit()
    db_session.refresh(analysis)
    return analysis


@pytest.fixture
def test_dataset(db_session, test_user):
    """Create test dataset"""
    dataset = Dataset(
        id="test-dataset-123",
        user_id=test_user.id,
        name="Test Dataset",
        filename="test.csv",
        file_path="/tmp/test.csv",
        file_size=1024,
        mime_type="text/csv",
        row_count=100,
        column_count=5,
        created_at=datetime.utcnow()
    )
    db_session.add(dataset)
    db_session.commit()
    db_session.refresh(dataset)
    return dataset


# Authentication fixtures
@pytest.fixture
def auth_headers(test_user):
    """Get authentication headers for test user"""
    from apps.api.app.auth import create_access_token
    token = create_access_token(data={"sub": test_user.username})
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def admin_auth_headers(test_admin_user):
    """Get authentication headers for admin user"""
    from apps.api.app.auth import create_access_token
    token = create_access_token(data={"sub": test_admin_user.username})
    return {"Authorization": f"Bearer {token}"}


# Async fixtures
@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests"""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


# Test data fixtures
@pytest.fixture
def sample_csv_data():
    """Sample CSV data for testing"""
    return """name,age,city,salary
John,25,New York,50000
Jane,30,Los Angeles,60000
Bob,35,Chicago,55000
Alice,28,San Francisco,70000
Charlie,32,Boston,65000"""


@pytest.fixture
def sample_json_data():
    """Sample JSON data for testing"""
    return {
        "users": [
            {"name": "John", "age": 25, "city": "New York", "salary": 50000},
            {"name": "Jane", "age": 30, "city": "Los Angeles", "salary": 60000},
            {"name": "Bob", "age": 35, "city": "Chicago", "salary": 55000},
            {"name": "Alice", "age": 28, "city": "San Francisco", "salary": 70000},
            {"name": "Charlie", "age": 32, "city": "Boston", "salary": 65000}
        ]
    }


@pytest.fixture
def sample_analysis_query():
    """Sample analysis query for testing"""
    return "Analyze the salary distribution by city and create a visualization"


# Performance testing fixtures
@pytest.fixture
def performance_test_data():
    """Large dataset for performance testing"""
    import pandas as pd
    import numpy as np
    
    # Generate large dataset
    np.random.seed(42)
    n_rows = 10000
    data = {
        'id': range(n_rows),
        'value1': np.random.normal(100, 15, n_rows),
        'value2': np.random.normal(50, 10, n_rows),
        'category': np.random.choice(['A', 'B', 'C', 'D'], n_rows),
        'date': pd.date_range('2023-01-01', periods=n_rows, freq='H')
    }
    return pd.DataFrame(data)


# Cleanup fixtures
@pytest.fixture(autouse=True)
def cleanup_test_files():
    """Clean up test files after each test"""
    yield
    # Clean up any temporary files created during tests
    import tempfile
    import shutil
    temp_dir = tempfile.gettempdir()
    for item in os.listdir(temp_dir):
        if item.startswith('rockyai_test_'):
            item_path = os.path.join(temp_dir, item)
            if os.path.isfile(item_path):
                os.remove(item_path)
            elif os.path.isdir(item_path):
                shutil.rmtree(item_path)
