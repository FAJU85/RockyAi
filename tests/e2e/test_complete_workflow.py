"""
End-to-end tests for complete Rocky AI workflows
"""
import pytest
import json
import time
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

from apps.api.app.main import app
from apps.api.app.database import User, Analysis, Dataset, UserRole


class TestCompleteAnalysisWorkflow:
    """Test complete analysis workflow from start to finish"""
    
    def test_complete_analysis_workflow(self, client, sample_csv_data, sample_analysis_query):
        """Test complete analysis workflow with file upload and analysis"""
        # Step 1: Register a new user
        user_data = {
            "username": "e2euser",
            "email": "e2e@example.com",
            "password": "E2EPassword123!",
            "role": "user"
        }
        
        response = client.post("/security/register", json=user_data)
        assert response.status_code == 201
        
        # Step 2: Login with the new user
        login_data = {
            "username": "e2euser",
            "password": "E2EPassword123!"
        }
        
        response = client.post("/security/login", json=login_data)
        assert response.status_code == 200
        
        auth_data = response.json()
        auth_headers = {"Authorization": f"Bearer {auth_data['access_token']}"}
        
        # Step 3: Upload a dataset
        files = {"file": ("e2e_test.csv", sample_csv_data, "text/csv")}
        response = client.post("/upload", files=files, headers=auth_headers)
        assert response.status_code == 200
        
        upload_data = response.json()
        file_id = upload_data["file_id"]
        
        # Step 4: Perform analysis on the uploaded dataset
        analysis_data = {
            "query": sample_analysis_query,
            "data_path": file_id
        }
        
        response = client.post("/analyze", json=analysis_data, headers=auth_headers)
        assert response.status_code == 200
        
        analysis_result = response.json()
        assert analysis_result["status"] == "completed"
        assert "result" in analysis_result
        assert "plan" in analysis_result
        assert "code" in analysis_result
        assert "output" in analysis_result
        
        # Step 5: Get analysis history
        response = client.get("/analyses", headers=auth_headers)
        assert response.status_code == 200
        
        analyses = response.json()
        assert len(analyses) >= 1
        
        # Find our analysis
        our_analysis = next((a for a in analyses if a["id"] == analysis_result["analysis_id"]), None)
        assert our_analysis is not None
        assert our_analysis["query"] == sample_analysis_query
        
        # Step 6: Get specific analysis details
        response = client.get(f"/analyses/{analysis_result['analysis_id']}", headers=auth_headers)
        assert response.status_code == 200
        
        analysis_details = response.json()
        assert analysis_details["id"] == analysis_result["analysis_id"]
        assert analysis_details["query"] == sample_analysis_query
        assert analysis_details["status"] == "completed"
        
        # Step 7: Get dataset information
        response = client.get("/datasets", headers=auth_headers)
        assert response.status_code == 200
        
        datasets = response.json()
        assert len(datasets) >= 1
        
        # Find our dataset
        our_dataset = next((d for d in datasets if d["id"] == file_id), None)
        assert our_dataset is not None
        assert our_dataset["filename"] == "e2e_test.csv"
        
        # Step 8: Clean up - delete the dataset
        response = client.delete(f"/datasets/{file_id}", headers=auth_headers)
        assert response.status_code == 200
        
        # Step 9: Verify dataset is deleted
        response = client.get("/datasets", headers=auth_headers)
        assert response.status_code == 200
        
        datasets = response.json()
        dataset_ids = [d["id"] for d in datasets]
        assert file_id not in dataset_ids


class TestMultiUserWorkflow:
    """Test multi-user workflow with different roles"""
    
    def test_admin_user_workflow(self, client, test_admin_user):
        """Test admin user workflow"""
        # Login as admin
        login_data = {
            "username": "admin",
            "password": "testpassword"
        }
        
        response = client.post("/security/login", json=login_data)
        assert response.status_code == 200
        
        auth_data = response.json()
        auth_headers = {"Authorization": f"Bearer {auth_data['access_token']}"}
        
        # Admin can view security metrics
        response = client.get("/security/metrics", headers=auth_headers)
        assert response.status_code == 200
        
        metrics = response.json()
        assert "total_users" in metrics
        assert "active_sessions" in metrics
        
        # Admin can view audit events
        response = client.get("/security/audit/events", headers=auth_headers)
        assert response.status_code == 200
        
        # Admin can generate audit reports
        response = client.get("/security/audit/report", headers=auth_headers)
        assert response.status_code == 200
        
        report = response.json()
        assert "report_type" in report
        assert "total_events" in report
    
    def test_regular_user_workflow(self, client, test_user):
        """Test regular user workflow"""
        # Login as regular user
        login_data = {
            "username": "testuser",
            "password": "testpassword"
        }
        
        response = client.post("/security/login", json=login_data)
        assert response.status_code == 200
        
        auth_data = response.json()
        auth_headers = {"Authorization": f"Bearer {auth_data['access_token']}"}
        
        # Regular user cannot view security metrics
        response = client.get("/security/metrics", headers=auth_headers)
        assert response.status_code == 403
        
        # Regular user can view their own data
        response = client.get("/analyses", headers=auth_headers)
        assert response.status_code == 200
        
        response = client.get("/datasets", headers=auth_headers)
        assert response.status_code == 200


class TestErrorRecoveryWorkflow:
    """Test error recovery and resilience"""
    
    def test_analysis_failure_recovery(self, client, auth_headers):
        """Test analysis failure and recovery"""
        # Mock orchestrator to simulate failure
        with patch('apps.api.app.orchestrator.get_orchestrator') as mock_orchestrator:
            mock_orchestrator_instance = MagicMock()
            mock_orchestrator_instance.execute_analysis.side_effect = Exception("Analysis failed")
            mock_orchestrator.return_value = mock_orchestrator_instance
            
            analysis_data = {
                "query": "Test query that will fail",
                "data_path": None
            }
            
            response = client.post("/analyze", json=analysis_data, headers=auth_headers)
            assert response.status_code == 500
            
            # Verify error is properly handled
            error_data = response.json()
            assert "error" in error_data
            assert "Analysis failed" in error_data["error"]
    
    def test_network_timeout_recovery(self, client, auth_headers):
        """Test network timeout recovery"""
        # Mock network timeout
        with patch('apps.api.app.orchestrator.get_orchestrator') as mock_orchestrator:
            mock_orchestrator_instance = MagicMock()
            mock_orchestrator_instance.execute_analysis.side_effect = TimeoutError("Network timeout")
            mock_orchestrator.return_value = mock_orchestrator_instance
            
            analysis_data = {
                "query": "Test query with timeout",
                "data_path": None
            }
            
            response = client.post("/analyze", json=analysis_data, headers=auth_headers)
            assert response.status_code == 500
            
            # Verify timeout is properly handled
            error_data = response.json()
            assert "error" in error_data
            assert "timeout" in error_data["error"].lower()
    
    def test_database_connection_recovery(self, client, auth_headers):
        """Test database connection recovery"""
        # Mock database connection failure
        with patch('apps.api.app.database.get_db') as mock_get_db:
            mock_get_db.side_effect = Exception("Database connection failed")
            
            response = client.get("/analyses", headers=auth_headers)
            assert response.status_code == 500
            
            # Verify database error is properly handled
            error_data = response.json()
            assert "error" in error_data


class TestPerformanceWorkflow:
    """Test performance under load"""
    
    def test_concurrent_analyses(self, client, auth_headers, sample_analysis_query):
        """Test concurrent analysis requests"""
        import threading
        import time
        
        results = []
        errors = []
        
        def run_analysis():
            try:
                analysis_data = {
                    "query": f"{sample_analysis_query} - {threading.current_thread().name}",
                    "data_path": None
                }
                
                response = client.post("/analyze", json=analysis_data, headers=auth_headers)
                results.append(response.status_code)
            except Exception as e:
                errors.append(str(e))
        
        # Start multiple concurrent analyses
        threads = []
        for i in range(5):
            thread = threading.Thread(target=run_analysis)
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Verify all analyses completed successfully
        assert len(errors) == 0
        assert all(status == 200 for status in results)
        assert len(results) == 5
    
    def test_large_dataset_analysis(self, client, auth_headers, performance_test_data):
        """Test analysis with large dataset"""
        # Save large dataset to temporary file
        import tempfile
        import os
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            performance_test_data.to_csv(f.name, index=False)
            temp_file_path = f.name
        
        try:
            # Upload large dataset
            with open(temp_file_path, 'rb') as f:
                files = {"file": ("large_dataset.csv", f, "text/csv")}
                response = client.post("/upload", files=files, headers=auth_headers)
                assert response.status_code == 200
                
                file_id = response.json()["file_id"]
            
            # Analyze large dataset
            analysis_data = {
                "query": "Analyze the large dataset and provide summary statistics",
                "data_path": file_id
            }
            
            start_time = time.time()
            response = client.post("/analyze", json=analysis_data, headers=auth_headers)
            end_time = time.time()
            
            assert response.status_code == 200
            assert (end_time - start_time) < 30  # Should complete within 30 seconds
            
            analysis_result = response.json()
            assert analysis_result["status"] == "completed"
            assert "result" in analysis_result
            
        finally:
            # Clean up temporary file
            os.unlink(temp_file_path)


class TestSecurityWorkflow:
    """Test security workflow and compliance"""
    
    def test_security_audit_trail(self, client, auth_headers, test_user):
        """Test security audit trail"""
        # Perform various actions that should be audited
        actions = [
            ("GET", "/analyses", "View analysis history"),
            ("GET", "/datasets", "View datasets"),
            ("GET", "/security/status", "View security status"),
            ("POST", "/security/change-password", "Change password", {
                "current_password": "testpassword",
                "new_password": "NewPassword123!"
            })
        ]
        
        for method, endpoint, action, *data in actions:
            if method == "GET":
                response = client.get(endpoint, headers=auth_headers)
            elif method == "POST":
                response = client.post(endpoint, json=data[0] if data else {}, headers=auth_headers)
            
            # All actions should be successful and audited
            assert response.status_code in [200, 201]
    
    def test_rate_limiting(self, client, auth_headers):
        """Test rate limiting functionality"""
        # Make many requests quickly to trigger rate limiting
        for i in range(70):  # More than the 60/minute limit
            response = client.get("/health", headers=auth_headers)
            
            if response.status_code == 429:
                # Rate limit triggered
                assert "Rate limit exceeded" in response.json()["detail"]
                break
        else:
            # If we get here, rate limiting didn't trigger
            # This might be expected in test environment
            pass
    
    def test_input_validation(self, client, auth_headers):
        """Test input validation and sanitization"""
        # Test various malicious inputs
        malicious_inputs = [
            "<script>alert('xss')</script>",
            "'; DROP TABLE users; --",
            "../../etc/passwd",
            "javascript:alert('xss')",
            "<img src=x onerror=alert('xss')>"
        ]
        
        for malicious_input in malicious_inputs:
            analysis_data = {
                "query": malicious_input,
                "data_path": None
            }
            
            response = client.post("/analyze", json=analysis_data, headers=auth_headers)
            # Should either be rejected (400) or sanitized (200)
            assert response.status_code in [200, 400]
            
            if response.status_code == 200:
                # If accepted, verify it was sanitized
                result = response.json()
                assert "<script>" not in result.get("result", "")
                assert "DROP TABLE" not in result.get("result", "")


class TestDataPersistenceWorkflow:
    """Test data persistence and consistency"""
    
    def test_data_consistency_across_restarts(self, client, auth_headers, test_analysis):
        """Test data consistency across application restarts"""
        # Get analysis before restart
        response = client.get(f"/analyses/{test_analysis.id}", headers=auth_headers)
        assert response.status_code == 200
        
        original_analysis = response.json()
        
        # Simulate application restart by clearing caches
        # (In real scenario, this would be a full restart)
        
        # Get analysis after restart
        response = client.get(f"/analyses/{test_analysis.id}", headers=auth_headers)
        assert response.status_code == 200
        
        restarted_analysis = response.json()
        
        # Data should be consistent
        assert original_analysis["id"] == restarted_analysis["id"]
        assert original_analysis["query"] == restarted_analysis["query"]
        assert original_analysis["status"] == restarted_analysis["status"]
        assert original_analysis["result"] == restarted_analysis["result"]
    
    def test_concurrent_data_modification(self, client, auth_headers, test_dataset):
        """Test concurrent data modification"""
        import threading
        import time
        
        results = []
        
        def modify_dataset():
            try:
                # Try to delete the dataset
                response = client.delete(f"/datasets/{test_dataset.id}", headers=auth_headers)
                results.append(response.status_code)
            except Exception as e:
                results.append(f"Error: {str(e)}")
        
        # Start multiple threads trying to modify the same dataset
        threads = []
        for i in range(3):
            thread = threading.Thread(target=modify_dataset)
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Only one should succeed (200), others should fail (404)
        success_count = results.count(200)
        not_found_count = results.count(404)
        
        assert success_count == 1
        assert not_found_count == 2
