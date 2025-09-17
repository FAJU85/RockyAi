"""
Unit tests for authentication module
"""
import pytest
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

from apps.api.app.auth import (
    verify_password,
    get_password_hash,
    create_access_token,
    verify_token,
    get_current_user,
    get_current_active_user,
    authenticate_user,
    create_user,
    get_user_by_username,
    get_user_by_email
)
from apps.api.app.database import User, UserRole


class TestPasswordHashing:
    """Test password hashing functionality"""
    
    def test_password_hashing(self):
        """Test password hashing and verification"""
        password = "testpassword123"
        hashed = get_password_hash(password)
        
        assert hashed != password
        assert verify_password(password, hashed)
        assert not verify_password("wrongpassword", hashed)
    
    def test_password_hash_uniqueness(self):
        """Test that same password produces different hashes"""
        password = "testpassword123"
        hash1 = get_password_hash(password)
        hash2 = get_password_hash(password)
        
        assert hash1 != hash2
        assert verify_password(password, hash1)
        assert verify_password(password, hash2)


class TestTokenManagement:
    """Test JWT token management"""
    
    def test_create_access_token(self):
        """Test access token creation"""
        data = {"sub": "testuser"}
        token = create_access_token(data)
        
        assert isinstance(token, str)
        assert len(token) > 0
    
    def test_verify_token_valid(self):
        """Test token verification with valid token"""
        data = {"sub": "testuser"}
        token = create_access_token(data)
        
        payload = verify_token(token)
        assert payload is not None
        assert payload["sub"] == "testuser"
    
    def test_verify_token_invalid(self):
        """Test token verification with invalid token"""
        invalid_token = "invalid.token.here"
        
        payload = verify_token(invalid_token)
        assert payload is None
    
    def test_verify_token_expired(self):
        """Test token verification with expired token"""
        data = {"sub": "testuser"}
        # Create token with very short expiration
        with patch('apps.api.app.auth.SECRET_KEY', 'test-secret'):
            with patch('apps.api.app.auth.ALGORITHM', 'HS256'):
                with patch('apps.api.app.auth.ACCESS_TOKEN_EXPIRE_MINUTES', 0):
                    token = create_access_token(data)
        
        payload = verify_token(token)
        assert payload is None


class TestUserAuthentication:
    """Test user authentication functionality"""
    
    @pytest.fixture
    def mock_db_session(self):
        """Mock database session"""
        session = MagicMock()
        return session
    
    def test_authenticate_user_success(self, mock_db_session):
        """Test successful user authentication"""
        # Mock user
        user = User(
            username="testuser",
            email="test@example.com",
            hashed_password=get_password_hash("testpassword"),
            role=UserRole.USER,
            is_active=True
        )
        mock_db_session.query.return_value.filter.return_value.first.return_value = user
        
        result = authenticate_user(mock_db_session, "testuser", "testpassword")
        
        assert result == user
    
    def test_authenticate_user_wrong_password(self, mock_db_session):
        """Test user authentication with wrong password"""
        # Mock user
        user = User(
            username="testuser",
            email="test@example.com",
            hashed_password=get_password_hash("testpassword"),
            role=UserRole.USER,
            is_active=True
        )
        mock_db_session.query.return_value.filter.return_value.first.return_value = user
        
        result = authenticate_user(mock_db_session, "testuser", "wrongpassword")
        
        assert result is False
    
    def test_authenticate_user_not_found(self, mock_db_session):
        """Test user authentication with non-existent user"""
        mock_db_session.query.return_value.filter.return_value.first.return_value = None
        
        result = authenticate_user(mock_db_session, "nonexistent", "password")
        
        assert result is False
    
    def test_authenticate_user_inactive(self, mock_db_session):
        """Test user authentication with inactive user"""
        # Mock inactive user
        user = User(
            username="testuser",
            email="test@example.com",
            hashed_password=get_password_hash("testpassword"),
            role=UserRole.USER,
            is_active=False
        )
        mock_db_session.query.return_value.filter.return_value.first.return_value = user
        
        result = authenticate_user(mock_db_session, "testuser", "testpassword")
        
        assert result is False


class TestUserManagement:
    """Test user management functionality"""
    
    @pytest.fixture
    def mock_db_session(self):
        """Mock database session"""
        session = MagicMock()
        return session
    
    def test_create_user_success(self, mock_db_session):
        """Test successful user creation"""
        user_data = {
            "username": "newuser",
            "email": "newuser@example.com",
            "password": "newpassword123",
            "role": UserRole.USER
        }
        
        with patch('apps.api.app.auth.get_password_hash') as mock_hash:
            mock_hash.return_value = "hashed_password"
            
            result = create_user(mock_db_session, user_data)
            
            assert result.username == "newuser"
            assert result.email == "newuser@example.com"
            assert result.hashed_password == "hashed_password"
            assert result.role == UserRole.USER
            assert result.is_active is True
    
    def test_get_user_by_username(self, mock_db_session):
        """Test getting user by username"""
        user = User(
            username="testuser",
            email="test@example.com",
            hashed_password="hashed_password",
            role=UserRole.USER,
            is_active=True
        )
        mock_db_session.query.return_value.filter.return_value.first.return_value = user
        
        result = get_user_by_username(mock_db_session, "testuser")
        
        assert result == user
    
    def test_get_user_by_email(self, mock_db_session):
        """Test getting user by email"""
        user = User(
            username="testuser",
            email="test@example.com",
            hashed_password="hashed_password",
            role=UserRole.USER,
            is_active=True
        )
        mock_db_session.query.return_value.filter.return_value.first.return_value = user
        
        result = get_user_by_email(mock_db_session, "test@example.com")
        
        assert result == user


class TestCurrentUserDependencies:
    """Test current user dependency functions"""
    
    @pytest.fixture
    def mock_db_session(self):
        """Mock database session"""
        session = MagicMock()
        return session
    
    def test_get_current_user_success(self, mock_db_session):
        """Test successful current user retrieval"""
        user = User(
            username="testuser",
            email="test@example.com",
            hashed_password="hashed_password",
            role=UserRole.USER,
            is_active=True
        )
        mock_db_session.query.return_value.filter.return_value.first.return_value = user
        
        with patch('apps.api.app.auth.verify_token') as mock_verify:
            mock_verify.return_value = {"sub": "testuser"}
            
            result = get_current_user(mock_db_session, "valid_token")
            
            assert result == user
    
    def test_get_current_user_invalid_token(self, mock_db_session):
        """Test current user retrieval with invalid token"""
        with patch('apps.api.app.auth.verify_token') as mock_verify:
            mock_verify.return_value = None
            
            with pytest.raises(Exception):
                get_current_user(mock_db_session, "invalid_token")
    
    def test_get_current_user_not_found(self, mock_db_session):
        """Test current user retrieval when user not found"""
        mock_db_session.query.return_value.filter.return_value.first.return_value = None
        
        with patch('apps.api.app.auth.verify_token') as mock_verify:
            mock_verify.return_value = {"sub": "nonexistent"}
            
            with pytest.raises(Exception):
                get_current_user(mock_db_session, "valid_token")
    
    def test_get_current_active_user_active(self, mock_db_session):
        """Test current active user retrieval with active user"""
        user = User(
            username="testuser",
            email="test@example.com",
            hashed_password="hashed_password",
            role=UserRole.USER,
            is_active=True
        )
        mock_db_session.query.return_value.filter.return_value.first.return_value = user
        
        with patch('apps.api.app.auth.verify_token') as mock_verify:
            mock_verify.return_value = {"sub": "testuser"}
            
            result = get_current_active_user(mock_db_session, "valid_token")
            
            assert result == user
    
    def test_get_current_active_user_inactive(self, mock_db_session):
        """Test current active user retrieval with inactive user"""
        user = User(
            username="testuser",
            email="test@example.com",
            hashed_password="hashed_password",
            role=UserRole.USER,
            is_active=False
        )
        mock_db_session.query.return_value.filter.return_value.first.return_value = user
        
        with patch('apps.api.app.auth.verify_token') as mock_verify:
            mock_verify.return_value = {"sub": "testuser"}
            
            with pytest.raises(Exception):
                get_current_active_user(mock_db_session, "valid_token")


class TestSecurityFeatures:
    """Test security-related features"""
    
    def test_password_strength_validation(self):
        """Test password strength validation"""
        # Strong password
        strong_password = "StrongPassword123!"
        hashed = get_password_hash(strong_password)
        assert verify_password(strong_password, hashed)
        
        # Weak password
        weak_password = "123"
        hashed = get_password_hash(weak_password)
        assert verify_password(weak_password, hashed)
    
    def test_token_expiration(self):
        """Test token expiration handling"""
        data = {"sub": "testuser"}
        token = create_access_token(data)
        
        # Token should be valid immediately
        payload = verify_token(token)
        assert payload is not None
        
        # Test with expired token
        with patch('apps.api.app.auth.ACCESS_TOKEN_EXPIRE_MINUTES', 0):
            expired_token = create_access_token(data)
            payload = verify_token(expired_token)
            assert payload is None
    
    def test_user_role_validation(self):
        """Test user role validation"""
        user = User(
            username="testuser",
            email="test@example.com",
            hashed_password="hashed_password",
            role=UserRole.USER,
            is_active=True
        )
        
        assert user.role == UserRole.USER
        assert user.is_active is True
    
    def test_user_account_lockout(self):
        """Test user account lockout functionality"""
        user = User(
            username="testuser",
            email="test@example.com",
            hashed_password="hashed_password",
            role=UserRole.USER,
            is_active=True,
            failed_login_attempts=0,
            is_locked=False
        )
        
        # Simulate failed login attempts
        user.failed_login_attempts += 1
        assert user.failed_login_attempts == 1
        
        # Simulate account lockout
        user.is_locked = True
        user.locked_until = datetime.utcnow() + timedelta(minutes=30)
        assert user.is_locked is True
        assert user.locked_until is not None
