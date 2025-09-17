"""
Model detection and routing for Docker Model Runner
Automatically selects the best available model based on system resources
"""
import subprocess
import psutil
import os
import yaml
from typing import Dict, Optional, Tuple


class ModelDetector:
    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = config_path
        self.config = self._load_config()
    
    def _load_config(self) -> Dict:
        """Load DMR configuration"""
        try:
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict:
        """Fallback configuration if config file not found"""
        return {
            'models': [
                {'name': 'ai/llama3', 'when': {'gpu': True, 'min_vram_gb': 10}},
                {'name': 'ai/llama2', 'when': {'gpu': False}}
            ],
            'routing': {'default': 'ai/llama2', 'gpu_preferred': 'ai/llama3'}
        }
    
    def detect_gpu(self) -> Tuple[bool, float]:
        """Detect GPU availability and VRAM"""
        try:
            # Check for NVIDIA GPU
            result = subprocess.run(['nvidia-smi', '--query-gpu=memory.total', '--format=csv,noheader,nounits'], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                vram_mb = float(result.stdout.strip())
                vram_gb = vram_mb / 1024
                return True, vram_gb
        except (subprocess.TimeoutExpired, FileNotFoundError, ValueError):
            pass
        
        # Check for other GPU types (AMD, Intel)
        try:
            result = subprocess.run(['lspci'], capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                gpu_detected = any(keyword in result.stdout.lower() 
                                 for keyword in ['nvidia', 'amd', 'radeon', 'intel'])
                return gpu_detected, 0.0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
        
        return False, 0.0
    
    def get_system_resources(self) -> Dict:
        """Get current system resource information"""
        memory_gb = psutil.virtual_memory().total / (1024**3)
        cpu_count = psutil.cpu_count()
        gpu_available, vram_gb = self.detect_gpu()
        
        return {
            'memory_gb': memory_gb,
            'cpu_count': cpu_count,
            'gpu_available': gpu_available,
            'vram_gb': vram_gb
        }
    
    def select_model(self) -> str:
        """Select the best model based on system resources"""
        resources = self.get_system_resources()
        
        # Check each model's requirements
        for model in self.config.get('models', []):
            name = model['name']
            requirements = model.get('when', {})
            
            # Check GPU requirement
            if requirements.get('gpu', False):
                if not resources['gpu_available']:
                    continue
                
                # Check VRAM requirement
                min_vram = requirements.get('min_vram_gb', 0)
                if resources['vram_gb'] < min_vram:
                    continue
            
            # Check memory requirement
            min_memory = requirements.get('min_memory_gb', 0)
            if resources['memory_gb'] < min_memory:
                continue
            
            return name
        
        # Fallback to default
        return self.config.get('routing', {}).get('default', 'ai/llama2')
    
    def get_model_config(self, model_name: str) -> Dict:
        """Get configuration for a specific model"""
        model_configs = self.config.get('model_configs', {})
        return model_configs.get(model_name, {
            'context_length': 4096,
            'temperature': 0.7,
            'top_p': 0.9,
            'max_tokens': 2048
        })
    
    def log_detection_results(self):
        """Log the detection results for debugging"""
        resources = self.get_system_resources()
        selected_model = self.select_model()
        
        print(f"System Resources:")
        print(f"  Memory: {resources['memory_gb']:.1f} GB")
        print(f"  CPU Cores: {resources['cpu_count']}")
        print(f"  GPU Available: {resources['gpu_available']}")
        if resources['gpu_available']:
            print(f"  VRAM: {resources['vram_gb']:.1f} GB")
        print(f"Selected Model: {selected_model}")


if __name__ == "__main__":
    detector = ModelDetector()
    detector.log_detection_results()
