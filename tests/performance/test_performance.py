"""
Performance tests for Rocky AI
"""
import pytest
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch, MagicMock

from apps.api.app.main import app
from apps.api.app.orchestrator import Orchestrator
from apps.api.app.database import User, Analysis, Dataset, UserRole


class TestAPIPerformance:
    """Test API performance characteristics"""
    
    def test_health_check_performance(self, client):
        """Test health check endpoint performance"""
        start_time = time.time()
        
        for _ in range(100):
            response = client.get("/health")
            assert response.status_code == 200
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Health check should be very fast
        assert total_time < 5.0  # 100 requests in under 5 seconds
        assert total_time / 100 < 0.05  # Each request under 50ms
    
    def test_analysis_performance(self, client, auth_headers, sample_analysis_query):
        """Test analysis endpoint performance"""
        analysis_data = {
            "query": sample_analysis_query,
            "data_path": None
        }
        
        start_time = time.time()
        
        response = client.post("/analyze", json=analysis_data, headers=auth_headers)
        
        end_time = time.time()
        response_time = end_time - start_time
        
        assert response.status_code == 200
        assert response_time < 10.0  # Analysis should complete within 10 seconds
    
    def test_concurrent_analysis_performance(self, client, auth_headers, sample_analysis_query):
        """Test concurrent analysis performance"""
        def run_analysis():
            analysis_data = {
                "query": sample_analysis_query,
                "data_path": None
            }
            
            start_time = time.time()
            response = client.post("/analyze", json=analysis_data, headers=auth_headers)
            end_time = time.time()
            
            return {
                "status_code": response.status_code,
                "response_time": end_time - start_time
            }
        
        # Run 10 concurrent analyses
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(run_analysis) for _ in range(10)]
            results = [future.result() for future in futures]
        
        # All should succeed
        assert all(result["status_code"] == 200 for result in results)
        
        # Average response time should be reasonable
        avg_response_time = sum(result["response_time"] for result in results) / len(results)
        assert avg_response_time < 15.0  # Average under 15 seconds
        
        # No single request should take too long
        max_response_time = max(result["response_time"] for result in results)
        assert max_response_time < 30.0  # No single request over 30 seconds
    
    def test_database_query_performance(self, client, auth_headers, test_analysis):
        """Test database query performance"""
        start_time = time.time()
        
        # Perform multiple database queries
        for _ in range(50):
            response = client.get("/analyses", headers=auth_headers)
            assert response.status_code == 200
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Database queries should be fast
        assert total_time < 5.0  # 50 queries in under 5 seconds
        assert total_time / 50 < 0.1  # Each query under 100ms
    
    def test_file_upload_performance(self, client, auth_headers, sample_csv_data):
        """Test file upload performance"""
        files = {"file": ("performance_test.csv", sample_csv_data, "text/csv")}
        
        start_time = time.time()
        
        response = client.post("/upload", files=files, headers=auth_headers)
        
        end_time = time.time()
        upload_time = end_time - start_time
        
        assert response.status_code == 200
        assert upload_time < 2.0  # File upload should complete within 2 seconds
    
    def test_large_file_upload_performance(self, client, auth_headers):
        """Test large file upload performance"""
        # Create a larger CSV file (1MB)
        large_csv_data = "name,age,city,salary\n" + "\n".join([
            f"User{i},{20 + (i % 50)},{'City' + str(i % 10)},{30000 + (i * 1000)}"
            for i in range(10000)
        ])
        
        files = {"file": ("large_performance_test.csv", large_csv_data, "text/csv")}
        
        start_time = time.time()
        
        response = client.post("/upload", files=files, headers=auth_headers)
        
        end_time = time.time()
        upload_time = end_time - start_time
        
        assert response.status_code == 200
        assert upload_time < 10.0  # Large file upload should complete within 10 seconds


class TestOrchestratorPerformance:
    """Test orchestrator performance"""
    
    @pytest.fixture
    def orchestrator(self):
        """Create orchestrator instance for testing"""
        return Orchestrator()
    
    def test_orchestrator_initialization_performance(self, orchestrator):
        """Test orchestrator initialization performance"""
        start_time = time.time()
        
        # Initialize orchestrator
        asyncio.run(orchestrator.initialize())
        
        end_time = time.time()
        init_time = end_time - start_time
        
        assert init_time < 5.0  # Initialization should complete within 5 seconds
    
    def test_orchestrator_analysis_performance(self, orchestrator, sample_analysis_query):
        """Test orchestrator analysis performance"""
        # Mock dependencies
        with patch('apps.api.app.orchestrator.get_cache') as mock_cache, \
             patch('apps.api.app.orchestrator.get_db') as mock_db, \
             patch('apps.api.app.orchestrator.get_executor_service') as mock_executor, \
             patch('apps.api.app.orchestrator.get_connection_manager') as mock_websocket:
            
            # Setup mocks
            mock_cache.return_value = MagicMock()
            mock_db.return_value = MagicMock()
            mock_executor.return_value = MagicMock()
            mock_websocket.return_value = MagicMock()
            
            start_time = time.time()
            
            # Run analysis
            result = asyncio.run(orchestrator.execute_analysis(
                query=sample_analysis_query,
                user_id="test-user",
                data_path=None
            ))
            
            end_time = time.time()
            analysis_time = end_time - start_time
            
            assert result is not None
            assert analysis_time < 15.0  # Analysis should complete within 15 seconds
    
    def test_orchestrator_concurrent_analyses(self, orchestrator, sample_analysis_query):
        """Test orchestrator concurrent analysis performance"""
        async def run_analysis():
            with patch('apps.api.app.orchestrator.get_cache') as mock_cache, \
                 patch('apps.api.app.orchestrator.get_db') as mock_db, \
                 patch('apps.api.app.orchestrator.get_executor_service') as mock_executor, \
                 patch('apps.api.app.orchestrator.get_connection_manager') as mock_websocket:
                
                # Setup mocks
                mock_cache.return_value = MagicMock()
                mock_db.return_value = MagicMock()
                mock_executor.return_value = MagicMock()
                mock_websocket.return_value = MagicMock()
                
                start_time = time.time()
                
                result = await orchestrator.execute_analysis(
                    query=sample_analysis_query,
                    user_id="test-user",
                    data_path=None
                )
                
                end_time = time.time()
                
                return {
                    "result": result,
                    "response_time": end_time - start_time
                }
        
        # Run 5 concurrent analyses
        start_time = time.time()
        
        tasks = [run_analysis() for _ in range(5)]
        results = asyncio.run(asyncio.gather(*tasks))
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # All analyses should succeed
        assert all(result["result"] is not None for result in results)
        
        # Total time should be reasonable
        assert total_time < 20.0  # 5 concurrent analyses in under 20 seconds
        
        # Average response time should be reasonable
        avg_response_time = sum(result["response_time"] for result in results) / len(results)
        assert avg_response_time < 10.0  # Average under 10 seconds


class TestDatabasePerformance:
    """Test database performance"""
    
    def test_database_connection_performance(self, db_session):
        """Test database connection performance"""
        start_time = time.time()
        
        # Perform multiple database operations
        for _ in range(100):
            # Simple query
            result = db_session.query(User).first()
        
        end_time = time.time()
        total_time = end_time - start_time
        
        assert total_time < 5.0  # 100 queries in under 5 seconds
        assert total_time / 100 < 0.05  # Each query under 50ms
    
    def test_database_insert_performance(self, db_session):
        """Test database insert performance"""
        start_time = time.time()
        
        # Insert multiple records
        for i in range(100):
            user = User(
                username=f"perfuser{i}",
                email=f"perfuser{i}@example.com",
                hashed_password="hashed_password",
                role=UserRole.USER,
                is_active=True
            )
            db_session.add(user)
        
        db_session.commit()
        
        end_time = time.time()
        insert_time = end_time - start_time
        
        assert insert_time < 10.0  # 100 inserts in under 10 seconds
        assert insert_time / 100 < 0.1  # Each insert under 100ms
    
    def test_database_query_performance(self, db_session, test_user):
        """Test database query performance"""
        start_time = time.time()
        
        # Perform various queries
        for _ in range(100):
            # Query by username
            user = db_session.query(User).filter(User.username == test_user.username).first()
            assert user is not None
            
            # Query by email
            user = db_session.query(User).filter(User.email == test_user.email).first()
            assert user is not None
            
            # Query by role
            users = db_session.query(User).filter(User.role == UserRole.USER).all()
            assert len(users) >= 1
        
        end_time = time.time()
        query_time = end_time - start_time
        
        assert query_time < 10.0  # 300 queries in under 10 seconds
        assert query_time / 300 < 0.033  # Each query under 33ms
    
    def test_database_transaction_performance(self, db_session):
        """Test database transaction performance"""
        start_time = time.time()
        
        # Perform multiple transactions
        for i in range(50):
            # Start transaction
            user = User(
                username=f"txuser{i}",
                email=f"txuser{i}@example.com",
                hashed_password="hashed_password",
                role=UserRole.USER,
                is_active=True
            )
            db_session.add(user)
            
            # Commit transaction
            db_session.commit()
        
        end_time = time.time()
        transaction_time = end_time - start_time
        
        assert transaction_time < 15.0  # 50 transactions in under 15 seconds
        assert transaction_time / 50 < 0.3  # Each transaction under 300ms


class TestCachePerformance:
    """Test cache performance"""
    
    def test_cache_set_performance(self, mock_cache):
        """Test cache set performance"""
        start_time = time.time()
        
        # Set multiple cache entries
        for i in range(1000):
            mock_cache.set(f"key{i}", f"value{i}", ttl=3600)
        
        end_time = time.time()
        set_time = end_time - start_time
        
        assert set_time < 1.0  # 1000 sets in under 1 second
        assert set_time / 1000 < 0.001  # Each set under 1ms
    
    def test_cache_get_performance(self, mock_cache):
        """Test cache get performance"""
        # Set up cache entries
        for i in range(1000):
            mock_cache.set(f"key{i}", f"value{i}", ttl=3600)
        
        start_time = time.time()
        
        # Get multiple cache entries
        for i in range(1000):
            value = mock_cache.get(f"key{i}")
            assert value == f"value{i}"
        
        end_time = time.time()
        get_time = end_time - start_time
        
        assert get_time < 1.0  # 1000 gets in under 1 second
        assert get_time / 1000 < 0.001  # Each get under 1ms
    
    def test_cache_delete_performance(self, mock_cache):
        """Test cache delete performance"""
        # Set up cache entries
        for i in range(1000):
            mock_cache.set(f"key{i}", f"value{i}", ttl=3600)
        
        start_time = time.time()
        
        # Delete multiple cache entries
        for i in range(1000):
            mock_cache.delete(f"key{i}")
        
        end_time = time.time()
        delete_time = end_time - start_time
        
        assert delete_time < 1.0  # 1000 deletes in under 1 second
        assert delete_time / 1000 < 0.001  # Each delete under 1ms


class TestMemoryPerformance:
    """Test memory performance"""
    
    def test_memory_usage_analysis(self, client, auth_headers, performance_test_data):
        """Test memory usage during analysis"""
        import psutil
        import os
        
        # Get initial memory usage
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Perform analysis with large dataset
        analysis_data = {
            "query": "Analyze the large dataset and provide summary statistics",
            "data_path": None
        }
        
        response = client.post("/analyze", json=analysis_data, headers=auth_headers)
        
        # Get memory usage after analysis
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        assert response.status_code == 200
        assert memory_increase < 100  # Memory increase should be less than 100MB
    
    def test_memory_usage_concurrent_analyses(self, client, auth_headers, sample_analysis_query):
        """Test memory usage during concurrent analyses"""
        import psutil
        import os
        
        # Get initial memory usage
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        def run_analysis():
            analysis_data = {
                "query": sample_analysis_query,
                "data_path": None
            }
            
            response = client.post("/analyze", json=analysis_data, headers=auth_headers)
            return response.status_code
        
        # Run 10 concurrent analyses
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(run_analysis) for _ in range(10)]
            results = [future.result() for future in futures]
        
        # Get memory usage after analyses
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        assert all(status == 200 for status in results)
        assert memory_increase < 200  # Memory increase should be less than 200MB


class TestScalabilityPerformance:
    """Test scalability performance"""
    
    def test_user_scalability(self, client, auth_headers):
        """Test performance with many users"""
        # Simulate many users by creating many analyses
        start_time = time.time()
        
        for i in range(100):
            analysis_data = {
                "query": f"Analysis query {i}",
                "data_path": None
            }
            
            response = client.post("/analyze", json=analysis_data, headers=auth_headers)
            assert response.status_code == 200
        
        end_time = time.time()
        total_time = end_time - start_time
        
        assert total_time < 60.0  # 100 analyses in under 60 seconds
        assert total_time / 100 < 0.6  # Each analysis under 600ms
    
    def test_data_scalability(self, client, auth_headers, performance_test_data):
        """Test performance with large datasets"""
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
            assert (end_time - start_time) < 30.0  # Should complete within 30 seconds
            
        finally:
            # Clean up temporary file
            os.unlink(temp_file_path)
    
    def test_concurrent_user_scalability(self, client, auth_headers, sample_analysis_query):
        """Test performance with many concurrent users"""
        def run_user_session():
            # Simulate a user session with multiple operations
            operations = [
                ("GET", "/analyses"),
                ("GET", "/datasets"),
                ("POST", "/analyze", {"query": sample_analysis_query, "data_path": None}),
                ("GET", "/security/status")
            ]
            
            results = []
            for method, endpoint, *data in operations:
                if method == "GET":
                    response = client.get(endpoint, headers=auth_headers)
                elif method == "POST":
                    response = client.post(endpoint, json=data[0] if data else {}, headers=auth_headers)
                
                results.append(response.status_code)
            
            return results
        
        # Run 20 concurrent user sessions
        with ThreadPoolExecutor(max_workers=20) as executor:
            futures = [executor.submit(run_user_session) for _ in range(20)]
            results = [future.result() for future in futures]
        
        # All operations should succeed
        assert all(all(status in [200, 201] for status in session) for session in results)
        assert len(results) == 20
