"""
Integration tests for API endpoints
"""
import pytest
import json
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

from apps.api.app.main import app
from apps.api.app.database import User, Analysis, Dataset, UserRole


class TestHealthEndpoints:
    """Test health check endpoints"""
    
    def test_health_check(self, client):
        """Test basic health check"""
        response = client.get("/health")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert "version" in data
    
    def test_health_check_detailed(self, client):
        """Test detailed health check"""
        response = client.get("/health/detailed")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "healthy"
        assert "services" in data
        assert "database" in data["services"]
        assert "cache" in data["services"]
        assert "dmr" in data["services"]


class TestAuthenticationEndpoints:
    """Test authentication endpoints"""
    
    def test_login_success(self, client, test_user):
        """Test successful login"""
        login_data = {
            "username": "testuser",
            "password": "testpassword"
        }
        
        response = client.post("/security/login", json=login_data)
        assert response.status_code == 200
        
        data = response.json()
        assert "access_token" in data
        assert data["token_type"] == "bearer"
        assert data["user_id"] == test_user.id
        assert data["username"] == "testuser"
        assert data["role"] == "user"
    
    def test_login_invalid_credentials(self, client):
        """Test login with invalid credentials"""
        login_data = {
            "username": "nonexistent",
            "password": "wrongpassword"
        }
        
        response = client.post("/security/login", json=login_data)
        assert response.status_code == 401
        assert "Invalid username or password" in response.json()["detail"]
    
    def test_login_locked_account(self, client, db_session):
        """Test login with locked account"""
        # Create locked user
        locked_user = User(
            username="lockeduser",
            email="locked@example.com",
            hashed_password="$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewdBPj4J/4X5Q5K2",
            role=UserRole.USER,
            is_active=True,
            failed_login_attempts=5,
            is_locked=True
        )
        db_session.add(locked_user)
        db_session.commit()
        
        login_data = {
            "username": "lockeduser",
            "password": "testpassword"
        }
        
        response = client.post("/security/login", json=login_data)
        assert response.status_code == 423
        assert "Account is locked" in response.json()["detail"]
    
    def test_logout(self, client, auth_headers):
        """Test logout"""
        response = client.post("/security/logout", headers=auth_headers)
        assert response.status_code == 200
        assert "Successfully logged out" in response.json()["message"]
    
    def test_change_password_success(self, client, auth_headers, test_user):
        """Test successful password change"""
        password_data = {
            "current_password": "testpassword",
            "new_password": "NewPassword123!"
        }
        
        response = client.post("/security/change-password", json=password_data, headers=auth_headers)
        assert response.status_code == 200
        assert response.json()["success"] is True
        assert "Password changed successfully" in response.json()["message"]
    
    def test_change_password_wrong_current(self, client, auth_headers):
        """Test password change with wrong current password"""
        password_data = {
            "current_password": "wrongpassword",
            "new_password": "NewPassword123!"
        }
        
        response = client.post("/security/change-password", json=password_data, headers=auth_headers)
        assert response.status_code == 400
        assert "Current password is incorrect" in response.json()["detail"]
    
    def test_change_password_weak_new(self, client, auth_headers):
        """Test password change with weak new password"""
        password_data = {
            "current_password": "testpassword",
            "new_password": "123"
        }
        
        response = client.post("/security/change-password", json=password_data, headers=auth_headers)
        assert response.status_code == 400
        assert "Password validation failed" in response.json()["detail"]


class TestAnalysisEndpoints:
    """Test analysis endpoints"""
    
    def test_analyze_data_success(self, client, auth_headers, sample_analysis_query):
        """Test successful data analysis"""
        analysis_data = {
            "query": sample_analysis_query,
            "data_path": None
        }
        
        response = client.post("/analyze", json=analysis_data, headers=auth_headers)
        assert response.status_code == 200
        
        data = response.json()
        assert "analysis_id" in data
        assert data["status"] == "completed"
        assert "result" in data
        assert "plan" in data
        assert "code" in data
        assert "output" in data
    
    def test_analyze_data_with_file(self, client, auth_headers, sample_analysis_query, sample_csv_data):
        """Test data analysis with file upload"""
        # First upload a file
        files = {"file": ("test.csv", sample_csv_data, "text/csv")}
        upload_response = client.post("/upload", files=files, headers=auth_headers)
        assert upload_response.status_code == 200
        
        file_id = upload_response.json()["file_id"]
        
        # Then analyze the data
        analysis_data = {
            "query": sample_analysis_query,
            "data_path": file_id
        }
        
        response = client.post("/analyze", json=analysis_data, headers=auth_headers)
        assert response.status_code == 200
        
        data = response.json()
        assert "analysis_id" in data
        assert data["status"] == "completed"
    
    def test_analyze_data_unauthorized(self, client, sample_analysis_query):
        """Test data analysis without authentication"""
        analysis_data = {
            "query": sample_analysis_query,
            "data_path": None
        }
        
        response = client.post("/analyze", json=analysis_data)
        assert response.status_code == 401
    
    def test_analyze_data_invalid_query(self, client, auth_headers):
        """Test data analysis with invalid query"""
        analysis_data = {
            "query": "",  # Empty query
            "data_path": None
        }
        
        response = client.post("/analyze", json=analysis_data, headers=auth_headers)
        assert response.status_code == 400
    
    def test_get_analysis_history(self, client, auth_headers, test_analysis):
        """Test getting analysis history"""
        response = client.get("/analyses", headers=auth_headers)
        assert response.status_code == 200
        
        data = response.json()
        assert isinstance(data, list)
        assert len(data) >= 1
        
        # Check that our test analysis is in the results
        analysis_ids = [analysis["id"] for analysis in data]
        assert test_analysis.id in analysis_ids
    
    def test_get_analysis_by_id(self, client, auth_headers, test_analysis):
        """Test getting specific analysis by ID"""
        response = client.get(f"/analyses/{test_analysis.id}", headers=auth_headers)
        assert response.status_code == 200
        
        data = response.json()
        assert data["id"] == test_analysis.id
        assert data["query"] == test_analysis.query
        assert data["status"] == test_analysis.status
        assert data["result"] == test_analysis.result
    
    def test_get_analysis_not_found(self, client, auth_headers):
        """Test getting non-existent analysis"""
        response = client.get("/analyses/nonexistent-id", headers=auth_headers)
        assert response.status_code == 404


class TestDatasetEndpoints:
    """Test dataset endpoints"""
    
    def test_upload_dataset_success(self, client, auth_headers, sample_csv_data):
        """Test successful dataset upload"""
        files = {"file": ("test.csv", sample_csv_data, "text/csv")}
        
        response = client.post("/upload", files=files, headers=auth_headers)
        assert response.status_code == 200
        
        data = response.json()
        assert "file_id" in data
        assert "filename" in data
        assert "file_size" in data
        assert "mime_type" in data
        assert data["filename"] == "test.csv"
        assert data["mime_type"] == "text/csv"
    
    def test_upload_dataset_invalid_type(self, client, auth_headers):
        """Test dataset upload with invalid file type"""
        files = {"file": ("malicious.exe", b"malicious content", "application/octet-stream")}
        
        response = client.post("/upload", files=files, headers=auth_headers)
        assert response.status_code == 400
        assert "Invalid file type" in response.json()["detail"]
    
    def test_upload_dataset_too_large(self, client, auth_headers):
        """Test dataset upload with file too large"""
        # Create a large file (20MB)
        large_content = b"x" * (20 * 1024 * 1024)
        files = {"file": ("large.csv", large_content, "text/csv")}
        
        response = client.post("/upload", files=files, headers=auth_headers)
        assert response.status_code == 400
        assert "File too large" in response.json()["detail"]
    
    def test_get_datasets(self, client, auth_headers, test_dataset):
        """Test getting user datasets"""
        response = client.get("/datasets", headers=auth_headers)
        assert response.status_code == 200
        
        data = response.json()
        assert isinstance(data, list)
        assert len(data) >= 1
        
        # Check that our test dataset is in the results
        dataset_ids = [dataset["id"] for dataset in data]
        assert test_dataset.id in dataset_ids
    
    def test_get_dataset_by_id(self, client, auth_headers, test_dataset):
        """Test getting specific dataset by ID"""
        response = client.get(f"/datasets/{test_dataset.id}", headers=auth_headers)
        assert response.status_code == 200
        
        data = response.json()
        assert data["id"] == test_dataset.id
        assert data["name"] == test_dataset.name
        assert data["filename"] == test_dataset.filename
        assert data["mime_type"] == test_dataset.mime_type
    
    def test_delete_dataset(self, client, auth_headers, test_dataset):
        """Test deleting a dataset"""
        response = client.delete(f"/datasets/{test_dataset.id}", headers=auth_headers)
        assert response.status_code == 200
        assert "Dataset deleted successfully" in response.json()["message"]
    
    def test_delete_dataset_not_found(self, client, auth_headers):
        """Test deleting non-existent dataset"""
        response = client.delete("/datasets/nonexistent-id", headers=auth_headers)
        assert response.status_code == 404


class TestWebSocketEndpoints:
    """Test WebSocket endpoints"""
    
    def test_websocket_connection(self, client, auth_headers):
        """Test WebSocket connection"""
        with client.websocket_connect("/ws", headers=auth_headers) as websocket:
            # Send a test message
            websocket.send_text("test message")
            
            # Receive response
            data = websocket.receive_text()
            assert data is not None
    
    def test_websocket_unauthorized(self, client):
        """Test WebSocket connection without authentication"""
        with pytest.raises(Exception):  # Should raise connection error
            with client.websocket_connect("/ws") as websocket:
                pass


class TestMetricsEndpoints:
    """Test metrics endpoints"""
    
    def test_metrics_endpoint(self, client):
        """Test metrics endpoint"""
        response = client.get("/metrics")
        assert response.status_code == 200
        assert "text/plain" in response.headers["content-type"]
    
    def test_metrics_json(self, client):
        """Test metrics JSON endpoint"""
        response = client.get("/metrics/json")
        assert response.status_code == 200
        
        data = response.json()
        assert "metrics" in data
        assert isinstance(data["metrics"], dict)


class TestSecurityEndpoints:
    """Test security endpoints"""
    
    def test_security_status(self, client, auth_headers, test_user):
        """Test security status endpoint"""
        response = client.get("/security/status", headers=auth_headers)
        assert response.status_code == 200
        
        data = response.json()
        assert data["user_id"] == test_user.id
        assert data["username"] == test_user.username
        assert data["role"] == test_user.role.value
        assert data["is_active"] == test_user.is_active
    
    def test_security_metrics_admin(self, client, admin_auth_headers):
        """Test security metrics endpoint (admin only)"""
        response = client.get("/security/metrics", headers=admin_auth_headers)
        assert response.status_code == 200
        
        data = response.json()
        assert "total_users" in data
        assert "active_sessions" in data
        assert "failed_login_attempts" in data
        assert "locked_accounts" in data
    
    def test_security_metrics_unauthorized(self, client, auth_headers):
        """Test security metrics endpoint without admin privileges"""
        response = client.get("/security/metrics", headers=auth_headers)
        assert response.status_code == 403
    
    def test_password_policy(self, client):
        """Test password policy endpoint"""
        response = client.get("/security/policy/password")
        assert response.status_code == 200
        
        data = response.json()
        assert "min_length" in data
        assert "require_uppercase" in data
        assert "require_lowercase" in data
        assert "require_digits" in data
        assert "require_special_chars" in data
    
    def test_session_policy(self, client):
        """Test session policy endpoint"""
        response = client.get("/security/policy/session")
        assert response.status_code == 200
        
        data = response.json()
        assert "max_duration_hours" in data
        assert "idle_timeout_minutes" in data
        assert "max_concurrent_sessions" in data


class TestErrorHandling:
    """Test error handling"""
    
    def test_404_endpoint(self, client):
        """Test 404 for non-existent endpoint"""
        response = client.get("/nonexistent")
        assert response.status_code == 404
    
    def test_405_method_not_allowed(self, client):
        """Test 405 for unsupported HTTP method"""
        response = client.delete("/health")
        assert response.status_code == 405
    
    def test_422_validation_error(self, client, auth_headers):
        """Test 422 for validation error"""
        invalid_data = {
            "query": 123,  # Should be string
            "data_path": None
        }
        
        response = client.post("/analyze", json=invalid_data, headers=auth_headers)
        assert response.status_code == 422
    
    def test_500_internal_server_error(self, client, auth_headers):
        """Test 500 for internal server error"""
        # Mock an internal error
        with patch('apps.api.app.orchestrator.get_orchestrator') as mock_orchestrator:
            mock_orchestrator.side_effect = Exception("Internal error")
            
            analysis_data = {
                "query": "Test query",
                "data_path": None
            }
            
            response = client.post("/analyze", json=analysis_data, headers=auth_headers)
            assert response.status_code == 500
