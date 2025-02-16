import os
import torch
import psutil
import uuid
import warnings
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

class SandboxManager:
    def __init__(self):
        self.sandbox_id = None
        self.checkpoint_dir = "checkpoints"
        self.resource_limits = {
            'max_memory_mb': 2048,  # 2GB
            'max_cpu_percent': 80,
            'max_gpu_memory_mb': 1024 if torch.cuda.is_available() else 0
        }
        self._ensure_checkpoint_dir()
        self.container = None
        self.sandbox_path = Path("./sandbox")
        self.sandbox_path.mkdir(exist_ok=True)
        self.using_docker = False
        
        # Try to import docker, fallback to process-based isolation if not available
        try:
            import docker
            self.client = docker.from_env()
            self.using_docker = True
        except (ImportError, Exception) as e:
            warnings.warn(f"Docker not available: {e}. Using process-based isolation instead.")
            self.process = psutil.Process(os.getpid())

    def _ensure_checkpoint_dir(self):
        """Ensure checkpoint directory exists"""
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
    
    def create_sandbox(self) -> str:
        """Create a new sandbox environment"""
        self.sandbox_id = str(uuid.uuid4())
        
        # Create sandbox-specific checkpoint directory
        sandbox_dir = os.path.join(self.checkpoint_dir, self.sandbox_id)
        os.makedirs(sandbox_dir, exist_ok=True)
        
        return self.sandbox_id
    
    def monitor_resources(self) -> Dict[str, float]:
        """Monitor resource usage"""
        process = psutil.Process(os.getpid())
        
        # Get memory usage
        memory_info = process.memory_info()
        memory_mb = memory_info.rss / 1024 / 1024
        
        # Get CPU usage
        cpu_percent = process.cpu_percent()
        
        # Get GPU memory if available
        gpu_memory_mb = 0
        if torch.cuda.is_available():
            gpu_memory_mb = torch.cuda.memory_allocated() / 1024 / 1024
        
        resources = {
            'memory_usage': memory_mb,
            'cpu_usage': cpu_percent,
            'gpu_memory_usage': gpu_memory_mb
        }
        
        # Check resource limits
        if memory_mb > self.resource_limits['max_memory_mb']:
            print(f"Warning: Memory usage ({memory_mb:.1f}MB) exceeds limit")
            
        if cpu_percent > self.resource_limits['max_cpu_percent']:
            print(f"Warning: CPU usage ({cpu_percent:.1f}%) exceeds limit")
            
        if gpu_memory_mb > self.resource_limits['max_gpu_memory_mb']:
            print(f"Warning: GPU memory usage ({gpu_memory_mb:.1f}MB) exceeds limit")
        
        return resources
    
    def save_state(self, filename: str) -> Optional[str]:
        """Save current state to checkpoint"""
        try:
            if not self.sandbox_id:
                print("No active sandbox")
                return None
            
            # Create timestamp-based filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            full_filename = f"{filename}_{timestamp}.pth"
            
            # Save to sandbox-specific directory
            save_path = os.path.join(self.checkpoint_dir, 
                                   self.sandbox_id, 
                                   full_filename)
            
            # Actual saving should be done by the caller
            return save_path
            
        except Exception as e:
            print(f"Error saving state: {str(e)}")
            return None
    
    def load_state(self, checkpoint_path: str) -> bool:
        """Load state from checkpoint"""
        try:
            if not os.path.exists(checkpoint_path):
                print(f"Checkpoint not found: {checkpoint_path}")
                return False
            
            # Actual loading should be done by the caller
            return True
            
        except Exception as e:
            print(f"Error loading state: {str(e)}")
            return False
    
    def cleanup(self):
        """Clean up sandbox resources"""
        if self.sandbox_id:
            try:
                # Clean up any temporary files
                sandbox_dir = os.path.join(self.checkpoint_dir, self.sandbox_id)
                if os.path.exists(sandbox_dir):
                    # Keep the directory but clean old checkpoints
                    checkpoints = os.listdir(sandbox_dir)
                    if len(checkpoints) > 10:  # Keep only last 10 checkpoints
                        checkpoints.sort()
                        for old_checkpoint in checkpoints[:-10]:
                            os.remove(os.path.join(sandbox_dir, old_checkpoint))
                
                # Reset sandbox ID
                self.sandbox_id = None
                
            except Exception as e:
                print(f"Error during cleanup: {str(e)}")
    
    def is_safe_operation(self, operation_type: str) -> bool:
        """Check if an operation is safe to perform"""
        if not self.sandbox_id:
            return False
            
        # Check resource usage before allowing operation
        resources = self.monitor_resources()
        
        # Define safety checks for different operation types
        safety_checks = {
            'memory_intensive': resources['memory_usage'] < self.resource_limits['max_memory_mb'],
            'cpu_intensive': resources['cpu_usage'] < self.resource_limits['max_cpu_percent'],
            'gpu_intensive': resources['gpu_memory_usage'] < self.resource_limits['max_gpu_memory_mb']
        }
        
        return safety_checks.get(operation_type, False)

    def _parse_memory_limit(self, limit: str) -> int:
        """Convert memory limit string (e.g., '4g') to bytes"""
        unit = limit[-1].lower()
        value = float(limit[:-1])
        multipliers = {'k': 1024, 'm': 1024**2, 'g': 1024**3}
        return int(value * multipliers.get(unit, 1))

    def create_sandbox(self) -> str:
        """Create an isolated environment for the digital child"""
        if self.using_docker:
            try:
                return self._create_docker_sandbox()
            except Exception as e:
                warnings.warn(f"Docker sandbox creation failed: {e}. Falling back to process isolation.")
                self.using_docker = False
                
        return self._create_process_sandbox()

    def _create_docker_sandbox(self) -> str:
        """Create Docker-based sandbox"""
        container_config = {
            'image': 'python:3.9-slim',
            'command': 'python /app/main.py',
            'volumes': {
                str(self.sandbox_path.absolute()): {
                    'bind': '/app',
                    'mode': 'rw'
                }
            },
            'mem_limit': f"{self._parse_memory_limit('4g')}b",
            'cpu_period': 100000,
            'cpu_quota': int(100000 * 0.5),
            'environment': {
                'SANDBOX_MODE': 'true'
            }
        }
        
        self.container = self.client.containers.run(**container_config, detach=True)
        return self.container.id

    def _create_process_sandbox(self) -> str:
        """Create process-based sandbox"""
        try:
            # Set process nice value for CPU priority
            os.nice(10)  # Lower priority
            
            # Set process memory limit
            import resource
            resource.setrlimit(resource.RLIMIT_AS, (self._parse_memory_limit('4g'), self._parse_memory_limit('4g')))
            
        except Exception as e:
            warnings.warn(f"Failed to set process limits: {e}")
            
        return f"process-{os.getpid()}"

    def monitor_resources(self) -> Dict[str, Any]:
        """Monitor resource usage"""
        if self.using_docker and self.container:
            try:
                stats = self.container.stats(stream=False)
                return {
                    'memory_usage': stats['memory_stats']['usage'],
                    'cpu_usage': stats['cpu_stats']['cpu_usage']['total_usage'],
                    'network_io': stats['networks']
                }
            except Exception:
                self.using_docker = False
        
        # Process-based monitoring
        try:
            return {
                'memory_usage': self.process.memory_info().rss,
                'cpu_usage': self.process.cpu_percent(),
                'network_io': {'rx_bytes': 0, 'tx_bytes': 0}
            }
        except Exception as e:
            return {
                'memory_usage': 0,
                'cpu_usage': 0,
                'network_io': {'rx_bytes': 0, 'tx_bytes': 0}
            }

    def save_state(self, checkpoint_path: str):
        """Save the current state of the digital child"""
        if self.using_docker and self.container:
            try:
                self.container.exec_run(
                    f"python -c 'import torch; torch.save(child.state_dict(), \"{checkpoint_path}\")'"
                )
            except Exception as e:
                print(f"Failed to save state in container: {e}")
                self._fallback_save_state(checkpoint_path)
        else:
            self._fallback_save_state(checkpoint_path)

    def _fallback_save_state(self, checkpoint_path: str):
        """Save state without container isolation"""
        try:
            if hasattr(self, 'child'):
                torch.save(self.child.state_dict(), checkpoint_path)
        except Exception as e:
            print(f"Failed to save state: {e}")

    def cleanup(self):
        """Clean up sandbox resources"""
        if self.using_docker and self.container:
            try:
                self.container.stop()
                self.container.remove()
            except Exception as e:
                warnings.warn(f"Failed to cleanup container: {e}")
