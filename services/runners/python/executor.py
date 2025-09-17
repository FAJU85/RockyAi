"""
Python Code Executor for Rocky AI
Provides secure, sandboxed execution of Python code with resource limits
"""
import asyncio
import subprocess
import tempfile
import os
import sys
import time
import psutil
import signal
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from pathlib import Path
import json
import shutil

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ExecutionResult:
    """Result of code execution"""
    success: bool
    output: str
    error: str
    execution_time: float
    memory_used_mb: float
    return_code: int
    stdout: str
    stderr: str


class PythonExecutor:
    """Secure Python code executor with resource limits"""
    
    def __init__(self, 
                 max_execution_time: int = 30,
                 max_memory_mb: int = 512,
                 max_output_size: int = 1024 * 1024,  # 1MB
                 allowed_libraries: List[str] = None,
                 working_dir: str = None):
        
        self.max_execution_time = max_execution_time
        self.max_memory_mb = max_memory_mb
        self.max_output_size = max_output_size
        self.allowed_libraries = allowed_libraries or self._get_default_allowed_libraries()
        self.working_dir = working_dir or tempfile.mkdtemp(prefix="rocky_python_")
        
        # Ensure working directory exists
        os.makedirs(self.working_dir, exist_ok=True)
        
        logger.info(f"Python executor initialized with working dir: {self.working_dir}")
    
    def _get_default_allowed_libraries(self) -> List[str]:
        """Get list of allowed Python libraries for security"""
        return [
            # Data manipulation
            "pandas", "numpy", "scipy", "scikit-learn", "statsmodels",
            # Visualization
            "matplotlib", "seaborn", "plotly",
            # Statistical analysis
            "lifelines",  # for survival analysis
            # Standard library
            "json", "csv", "math", "statistics", "random", "datetime",
            "collections", "itertools", "functools", "operator"
        ]
    
    def _validate_code(self, code: str) -> Tuple[bool, str]:
        """Validate code for security and allowed libraries"""
        try:
            # Parse the code to check for imports
            import ast
            tree = ast.parse(code)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        module_name = alias.name.split('.')[0]
                        if module_name not in self.allowed_libraries:
                            return False, f"Import of '{module_name}' not allowed"
                
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        module_name = node.module.split('.')[0]
                        if module_name not in self.allowed_libraries:
                            return False, f"Import from '{module_name}' not allowed"
            
            # Check for dangerous operations
            dangerous_patterns = [
                'os.system', 'subprocess', 'eval', 'exec', 'compile',
                '__import__', 'open(', 'file(', 'input(', 'raw_input('
            ]
            
            for pattern in dangerous_patterns:
                if pattern in code:
                    return False, f"Dangerous operation '{pattern}' not allowed"
            
            return True, ""
            
        except SyntaxError as e:
            return False, f"Syntax error: {e}"
        except Exception as e:
            return False, f"Validation error: {e}"
    
    def _create_sandbox_script(self, code: str) -> str:
        """Create a sandboxed execution script"""
        sandbox_code = f'''
import sys
import os
import signal
import resource
import time
import io
import contextlib
from pathlib import Path

# Set up resource limits
def set_limits():
    # Memory limit (in bytes)
    memory_limit = {self.max_memory_mb * 1024 * 1024}
    resource.setrlimit(resource.RLIMIT_AS, (memory_limit, memory_limit))
    
    # CPU time limit
    cpu_limit = {self.max_execution_time}
    resource.setrlimit(resource.RLIMIT_CPU, (cpu_limit, cpu_limit))

# Set up signal handlers for timeout
def timeout_handler(signum, frame):
    raise TimeoutError("Execution timed out")

signal.signal(signal.SIGALRM, timeout_handler)
signal.alarm({self.max_execution_time})

# Set working directory
os.chdir("{self.working_dir}")

# Capture output
old_stdout = sys.stdout
old_stderr = sys.stderr
sys.stdout = io.StringIO()
sys.stderr = io.StringIO()

try:
    # Set resource limits
    set_limits()
    
    # Execute user code
    exec(compile("""{code.replace('"', '\\"')}""", "<user_code>", "exec"))
    
    # Get output
    stdout_output = sys.stdout.getvalue()
    stderr_output = sys.stderr.getvalue()
    
    # Check output size
    total_output = len(stdout_output) + len(stderr_output)
    if total_output > {self.max_output_size}:
        stdout_output = stdout_output[:{self.max_output_size // 2}]
        stderr_output = stderr_output[:{self.max_output_size // 2}]
        stderr_output += "\\n[Output truncated due to size limit]"
    
    print(f"STDOUT_START{{stdout_output}}STDOUT_END")
    print(f"STDERR_START{{stderr_output}}STDERR_END")
    
except Exception as e:
    stderr_output = sys.stderr.getvalue() + str(e)
    print(f"STDOUT_START{{sys.stdout.getvalue()}}STDOUT_END")
    print(f"STDERR_START{{stderr_output}}STDERR_END")
    sys.exit(1)

finally:
    # Restore stdout/stderr
    sys.stdout = old_stdout
    sys.stderr = old_stderr
    signal.alarm(0)
'''
        return sandbox_code
    
    async def execute(self, code: str, timeout: int = None) -> ExecutionResult:
        """Execute Python code in a sandboxed environment"""
        start_time = time.time()
        
        # Validate code
        is_valid, error_msg = self._validate_code(code)
        if not is_valid:
            return ExecutionResult(
                success=False,
                output="",
                error=f"Code validation failed: {error_msg}",
                execution_time=0,
                memory_used_mb=0,
                return_code=1,
                stdout="",
                stderr=error_msg
            )
        
        # Create sandbox script
        sandbox_script = self._create_sandbox_script(code)
        
        # Write script to temporary file
        script_path = os.path.join(self.working_dir, "execution_script.py")
        with open(script_path, 'w', encoding='utf-8') as f:
            f.write(sandbox_script)
        
        try:
            # Execute with subprocess
            process = await asyncio.create_subprocess_exec(
                sys.executable, script_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self.working_dir
            )
            
            # Wait for completion with timeout
            timeout = timeout or self.max_execution_time
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(), 
                    timeout=timeout + 5  # Add buffer for cleanup
                )
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
                return ExecutionResult(
                    success=False,
                    output="",
                    error="Execution timed out",
                    execution_time=time.time() - start_time,
                    memory_used_mb=0,
                    return_code=-1,
                    stdout="",
                    stderr="Execution timed out"
                )
            
            # Parse output
            stdout_str = stdout.decode('utf-8', errors='replace')
            stderr_str = stderr.decode('utf-8', errors='replace')
            
            # Extract captured output
            stdout_output = ""
            stderr_output = ""
            
            if "STDOUT_START" in stdout_str and "STDOUT_END" in stdout_str:
                start_idx = stdout_str.find("STDOUT_START") + len("STDOUT_START")
                end_idx = stdout_str.find("STDOUT_END")
                stdout_output = stdout_str[start_idx:end_idx]
            
            if "STDERR_START" in stdout_str and "STDERR_END" in stdout_str:
                start_idx = stdout_str.find("STDERR_START") + len("STDERR_START")
                end_idx = stdout_str.find("STDERR_END")
                stderr_output = stdout_str[start_idx:end_idx]
            
            # Calculate memory usage (approximate)
            memory_used_mb = 0
            try:
                process_info = psutil.Process(process.pid)
                memory_used_mb = process_info.memory_info().rss / (1024 * 1024)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
            
            execution_time = time.time() - start_time
            success = process.returncode == 0 and not stderr_output
            
            return ExecutionResult(
                success=success,
                output=stdout_output,
                error=stderr_output,
                execution_time=execution_time,
                memory_used_mb=memory_used_mb,
                return_code=process.returncode,
                stdout=stdout_output,
                stderr=stderr_output
            )
            
        except Exception as e:
            logger.error(f"Execution failed: {e}")
            return ExecutionResult(
                success=False,
                output="",
                error=str(e),
                execution_time=time.time() - start_time,
                memory_used_mb=0,
                return_code=-1,
                stdout="",
                stderr=str(e)
            )
    
    def install_package(self, package_name: str) -> bool:
        """Install a Python package in the sandbox environment"""
        try:
            # Check if package is allowed
            if package_name not in self.allowed_libraries:
                logger.warning(f"Package {package_name} not in allowed list")
                return False
            
            # Install package
            result = subprocess.run([
                sys.executable, "-m", "pip", "install", package_name,
                "--target", self.working_dir
            ], capture_output=True, text=True, timeout=60)
            
            return result.returncode == 0
            
        except Exception as e:
            logger.error(f"Failed to install package {package_name}: {e}")
            return False
    
    def cleanup(self):
        """Clean up the working directory"""
        try:
            if os.path.exists(self.working_dir):
                shutil.rmtree(self.working_dir)
                logger.info(f"Cleaned up working directory: {self.working_dir}")
        except Exception as e:
            logger.error(f"Failed to cleanup working directory: {e}")
    
    def __del__(self):
        """Cleanup on destruction"""
        self.cleanup()


# Global executor instance
executor = None

def get_executor() -> PythonExecutor:
    """Get or create Python executor instance"""
    global executor
    if executor is None:
        executor = PythonExecutor()
    return executor


# Example usage and testing
async def test_executor():
    """Test the Python executor"""
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
    
    executor = PythonExecutor()
    result = await executor.execute(test_code)
    
    print(f"Success: {result.success}")
    print(f"Output: {result.output}")
    print(f"Error: {result.error}")
    print(f"Execution time: {result.execution_time:.2f}s")
    print(f"Memory used: {result.memory_used_mb:.2f}MB")
    
    executor.cleanup()


if __name__ == "__main__":
    asyncio.run(test_executor())
