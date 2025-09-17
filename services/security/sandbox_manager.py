"""
Sandbox Manager for Rocky AI
Provides secure, isolated execution environments with resource limits
"""
import os
import tempfile
import shutil
import subprocess
import psutil
import signal
import time
import logging
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import json
import uuid
from contextlib import contextmanager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SandboxManager:
    """Manage secure sandboxed execution environments"""
    
    def __init__(self, 
                 base_dir: str = "sandboxes",
                 max_memory_mb: int = 512,
                 max_cpu_percent: int = 50,
                 max_execution_time: int = 30,
                 max_file_size_mb: int = 10):
        
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)
        
        # Resource limits
        self.max_memory_mb = max_memory_mb
        self.max_cpu_percent = max_cpu_percent
        self.max_execution_time = max_execution_time
        self.max_file_size_mb = max_file_size_mb
        
        # Security settings
        self.allowed_extensions = {'.py', '.r', '.csv', '.json', '.txt', '.md'}
        self.blocked_paths = {
            '/etc', '/usr', '/bin', '/sbin', '/var', '/sys', '/proc',
            'C:\\Windows', 'C:\\System32', 'C:\\Program Files'
        }
        
        # Active sandboxes
        self.active_sandboxes = {}
        
        # Cleanup old sandboxes on startup
        self.cleanup_old_sandboxes()
    
    def create_sandbox(self, 
                      sandbox_id: Optional[str] = None,
                      language: str = 'python') -> str:
        """Create a new sandbox environment"""
        
        if sandbox_id is None:
            sandbox_id = str(uuid.uuid4())
        
        sandbox_path = self.base_dir / sandbox_id
        sandbox_path.mkdir(exist_ok=True)
        
        # Create subdirectories
        (sandbox_path / 'code').mkdir(exist_ok=True)
        (sandbox_path / 'data').mkdir(exist_ok=True)
        (sandbox_path / 'output').mkdir(exist_ok=True)
        (sandbox_path / 'logs').mkdir(exist_ok=True)
        
        # Create security configuration
        security_config = {
            'sandbox_id': sandbox_id,
            'created_at': time.time(),
            'language': language,
            'max_memory_mb': self.max_memory_mb,
            'max_cpu_percent': self.max_cpu_percent,
            'max_execution_time': self.max_execution_time,
            'max_file_size_mb': self.max_file_size_mb,
            'allowed_extensions': list(self.allowed_extensions),
            'blocked_paths': list(self.blocked_paths)
        }
        
        config_file = sandbox_path / 'security.json'
        with open(config_file, 'w') as f:
            json.dump(security_config, f, indent=2)
        
        # Track active sandbox
        self.active_sandboxes[sandbox_id] = {
            'path': sandbox_path,
            'created_at': time.time(),
            'language': language,
            'processes': []
        }
        
        logger.info(f"Created sandbox: {sandbox_id}")
        return sandbox_id
    
    def execute_code(self, 
                    sandbox_id: str,
                    code: str,
                    input_data: Optional[Dict[str, Any]] = None,
                    timeout: Optional[int] = None) -> Dict[str, Any]:
        """Execute code in sandbox with security constraints"""
        
        if sandbox_id not in self.active_sandboxes:
            raise ValueError(f"Sandbox {sandbox_id} not found")
        
        sandbox_info = self.active_sandboxes[sandbox_id]
        sandbox_path = sandbox_info['path']
        language = sandbox_info['language']
        
        # Validate code
        self._validate_code(code, language)
        
        # Prepare execution environment
        execution_id = str(uuid.uuid4())
        code_file = sandbox_path / 'code' / f'{execution_id}.{language}'
        output_file = sandbox_path / 'output' / f'{execution_id}.txt'
        error_file = sandbox_path / 'logs' / f'{execution_id}_error.txt'
        
        # Write code to file
        with open(code_file, 'w', encoding='utf-8') as f:
            f.write(code)
        
        # Prepare input data if provided
        if input_data:
            input_file = sandbox_path / 'data' / f'{execution_id}_input.json'
            with open(input_file, 'w') as f:
                json.dump(input_data, f)
        
        # Execute code
        start_time = time.time()
        result = self._execute_in_sandbox(
            sandbox_id=sandbox_id,
            code_file=code_file,
            output_file=output_file,
            error_file=error_file,
            language=language,
            timeout=timeout or self.max_execution_time
        )
        
        execution_time = time.time() - start_time
        
        # Read output
        output = ""
        if output_file.exists():
            with open(output_file, 'r', encoding='utf-8') as f:
                output = f.read()
        
        # Read errors
        error = ""
        if error_file.exists():
            with open(error_file, 'r', encoding='utf-8') as f:
                error = f.read()
        
        # Clean up execution files
        self._cleanup_execution_files(sandbox_path, execution_id)
        
        return {
            'sandbox_id': sandbox_id,
            'execution_id': execution_id,
            'success': result['success'],
            'output': output,
            'error': error,
            'execution_time': execution_time,
            'memory_used_mb': result.get('memory_used_mb', 0),
            'cpu_used_percent': result.get('cpu_used_percent', 0),
            'exit_code': result.get('exit_code', 0)
        }
    
    def _execute_in_sandbox(self, 
                           sandbox_id: str,
                           code_file: Path,
                           output_file: Path,
                           error_file: Path,
                           language: str,
                           timeout: int) -> Dict[str, Any]:
        """Execute code in isolated environment"""
        
        sandbox_path = self.active_sandboxes[sandbox_id]['path']
        
        # Prepare command based on language
        if language == 'python':
            cmd = ['python', str(code_file)]
        elif language == 'r':
            cmd = ['Rscript', str(code_file)]
        else:
            raise ValueError(f"Unsupported language: {language}")
        
        # Set up process with resource limits
        process = subprocess.Popen(
            cmd,
            stdout=open(output_file, 'w'),
            stderr=open(error_file, 'w'),
            cwd=str(sandbox_path),
            preexec_fn=self._set_resource_limits if os.name != 'nt' else None
        )
        
        # Track process
        self.active_sandboxes[sandbox_id]['processes'].append(process.pid)
        
        try:
            # Monitor execution with timeout
            start_time = time.time()
            memory_usage = 0
            cpu_usage = 0
            
            while process.poll() is None:
                # Check timeout
                if time.time() - start_time > timeout:
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                    
                    return {
                        'success': False,
                        'exit_code': -1,
                        'memory_used_mb': memory_usage,
                        'cpu_used_percent': cpu_usage,
                        'error': 'Execution timeout'
                    }
                
                # Monitor resource usage
                try:
                    proc = psutil.Process(process.pid)
                    memory_usage = max(memory_usage, proc.memory_info().rss / 1024 / 1024)
                    cpu_usage = max(cpu_usage, proc.cpu_percent())
                    
                    # Check memory limit
                    if memory_usage > self.max_memory_mb:
                        process.terminate()
                        return {
                            'success': False,
                            'exit_code': -1,
                            'memory_used_mb': memory_usage,
                            'cpu_used_percent': cpu_usage,
                            'error': 'Memory limit exceeded'
                        }
                    
                    # Check CPU limit
                    if cpu_usage > self.max_cpu_percent:
                        process.terminate()
                        return {
                            'success': False,
                            'exit_code': -1,
                            'memory_used_mb': memory_usage,
                            'cpu_used_percent': cpu_usage,
                            'error': 'CPU limit exceeded'
                        }
                
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
                
                time.sleep(0.1)
            
            # Get final resource usage
            try:
                proc = psutil.Process(process.pid)
                memory_usage = proc.memory_info().rss / 1024 / 1024
                cpu_usage = proc.cpu_percent()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
            
            return {
                'success': process.returncode == 0,
                'exit_code': process.returncode,
                'memory_used_mb': memory_usage,
                'cpu_used_percent': cpu_usage
            }
            
        except Exception as e:
            process.terminate()
            return {
                'success': False,
                'exit_code': -1,
                'memory_used_mb': memory_usage,
                'cpu_used_percent': cpu_usage,
                'error': str(e)
            }
        finally:
            # Remove process from tracking
            if process.pid in self.active_sandboxes[sandbox_id]['processes']:
                self.active_sandboxes[sandbox_id]['processes'].remove(process.pid)
    
    def _set_resource_limits(self):
        """Set resource limits for Unix systems"""
        try:
            import resource
            
            # Set memory limit
            memory_limit = self.max_memory_mb * 1024 * 1024  # Convert to bytes
            resource.setrlimit(resource.RLIMIT_AS, (memory_limit, memory_limit))
            
            # Set CPU time limit
            cpu_limit = self.max_execution_time
            resource.setrlimit(resource.RLIMIT_CPU, (cpu_limit, cpu_limit))
            
            # Set file size limit
            file_size_limit = self.max_file_size_mb * 1024 * 1024
            resource.setrlimit(resource.RLIMIT_FSIZE, (file_size_limit, file_size_limit))
            
        except Exception as e:
            logger.warning(f"Failed to set resource limits: {e}")
    
    def _validate_code(self, code: str, language: str):
        """Validate code for security issues"""
        
        # Check for dangerous operations
        dangerous_patterns = [
            'import os', 'import subprocess', 'import sys',
            'open(', 'file(', 'exec(', 'eval(',
            '__import__', 'getattr', 'setattr',
            'input(', 'raw_input('
        ]
        
        for pattern in dangerous_patterns:
            if pattern in code:
                raise SecurityError(f"Dangerous operation detected: {pattern}")
        
        # Check for file system access
        if 'open(' in code or 'file(' in code:
            raise SecurityError("File system access not allowed")
        
        # Check for network access
        network_patterns = ['urllib', 'requests', 'http', 'socket', 'ftplib']
        for pattern in network_patterns:
            if pattern in code:
                raise SecurityError(f"Network access not allowed: {pattern}")
    
    def _cleanup_execution_files(self, sandbox_path: Path, execution_id: str):
        """Clean up temporary execution files"""
        try:
            # Remove code file
            code_file = sandbox_path / 'code' / f'{execution_id}.py'
            if code_file.exists():
                code_file.unlink()
            
            code_file = sandbox_path / 'code' / f'{execution_id}.r'
            if code_file.exists():
                code_file.unlink()
            
            # Remove output files older than 1 hour
            output_dir = sandbox_path / 'output'
            for file in output_dir.glob('*'):
                if file.stat().st_mtime < time.time() - 3600:
                    file.unlink()
            
            # Remove log files older than 1 hour
            log_dir = sandbox_path / 'logs'
            for file in log_dir.glob('*'):
                if file.stat().st_mtime < time.time() - 3600:
                    file.unlink()
        
        except Exception as e:
            logger.warning(f"Failed to cleanup execution files: {e}")
    
    def cleanup_old_sandboxes(self, max_age_hours: int = 24):
        """Clean up old sandbox directories"""
        current_time = time.time()
        max_age_seconds = max_age_hours * 3600
        
        for sandbox_dir in self.base_dir.iterdir():
            if sandbox_dir.is_dir():
                try:
                    # Check if sandbox is too old
                    created_time = sandbox_dir.stat().st_ctime
                    if current_time - created_time > max_age_seconds:
                        # Check if sandbox is not active
                        sandbox_id = sandbox_dir.name
                        if sandbox_id not in self.active_sandboxes:
                            shutil.rmtree(sandbox_dir)
                            logger.info(f"Cleaned up old sandbox: {sandbox_id}")
                except Exception as e:
                    logger.warning(f"Failed to cleanup sandbox {sandbox_dir}: {e}")
    
    def destroy_sandbox(self, sandbox_id: str):
        """Destroy a sandbox and clean up resources"""
        
        if sandbox_id not in self.active_sandboxes:
            logger.warning(f"Sandbox {sandbox_id} not found")
            return
        
        sandbox_info = self.active_sandboxes[sandbox_id]
        sandbox_path = sandbox_info['path']
        
        # Terminate any running processes
        for pid in sandbox_info['processes']:
            try:
                process = psutil.Process(pid)
                process.terminate()
                process.wait(timeout=5)
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.TimeoutExpired):
                try:
                    process.kill()
                except:
                    pass
        
        # Remove sandbox directory
        try:
            shutil.rmtree(sandbox_path)
            logger.info(f"Destroyed sandbox: {sandbox_id}")
        except Exception as e:
            logger.error(f"Failed to destroy sandbox {sandbox_id}: {e}")
        
        # Remove from active sandboxes
        del self.active_sandboxes[sandbox_id]
    
    def get_sandbox_status(self, sandbox_id: str) -> Dict[str, Any]:
        """Get status of a sandbox"""
        
        if sandbox_id not in self.active_sandboxes:
            return {'status': 'not_found'}
        
        sandbox_info = self.active_sandboxes[sandbox_id]
        sandbox_path = sandbox_info['path']
        
        # Check if sandbox directory exists
        if not sandbox_path.exists():
            return {'status': 'destroyed'}
        
        # Get resource usage
        total_memory = 0
        total_cpu = 0
        
        for pid in sandbox_info['processes']:
            try:
                process = psutil.Process(pid)
                total_memory += process.memory_info().rss / 1024 / 1024
                total_cpu += process.cpu_percent()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        
        return {
            'status': 'active',
            'sandbox_id': sandbox_id,
            'created_at': sandbox_info['created_at'],
            'language': sandbox_info['language'],
            'active_processes': len(sandbox_info['processes']),
            'memory_used_mb': total_memory,
            'cpu_used_percent': total_cpu,
            'path': str(sandbox_path)
        }
    
    def list_sandboxes(self) -> List[Dict[str, Any]]:
        """List all active sandboxes"""
        return [
            self.get_sandbox_status(sandbox_id)
            for sandbox_id in self.active_sandboxes.keys()
        ]


class SecurityError(Exception):
    """Security violation in sandbox"""
    pass


# Example usage
if __name__ == "__main__":
    # Initialize sandbox manager
    sandbox_mgr = SandboxManager()
    
    # Create a sandbox
    sandbox_id = sandbox_mgr.create_sandbox(language='python')
    print(f"Created sandbox: {sandbox_id}")
    
    # Test code execution
    test_code = """
import pandas as pd
import numpy as np

# Create sample data
data = {'x': [1, 2, 3, 4, 5], 'y': [2, 4, 6, 8, 10]}
df = pd.DataFrame(data)

# Calculate correlation
correlation = df['x'].corr(df['y'])
print(f"Correlation: {correlation}")

# Basic statistics
print(f"Mean of x: {df['x'].mean()}")
print(f"Mean of y: {df['y'].mean()}")
"""
    
    try:
        result = sandbox_mgr.execute_code(sandbox_id, test_code)
        print("Execution result:")
        print(f"Success: {result['success']}")
        print(f"Output: {result['output']}")
        print(f"Error: {result['error']}")
        print(f"Execution time: {result['execution_time']:.2f}s")
        print(f"Memory used: {result['memory_used_mb']:.2f}MB")
    except Exception as e:
        print(f"Execution failed: {e}")
    finally:
        # Clean up
        sandbox_mgr.destroy_sandbox(sandbox_id)
        print("Sandbox destroyed")
