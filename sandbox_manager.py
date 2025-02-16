import os
import psutil
import warnings
from pathlib import Path
from typing import Dict, Any

class SandboxManager:
    def __init__(self, memory_limit: str = "4g", cpu_limit: float = 0.5):
        self.memory_limit = self._parse_memory_limit(memory_limit)
        self.cpu_limit = cpu_limit
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
            'mem_limit': f"{self.memory_limit}b",
            'cpu_period': 100000,
            'cpu_quota': int(100000 * self.cpu_limit),
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
            resource.setrlimit(resource.RLIMIT_AS, (self.memory_limit, self.memory_limit))
            
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
