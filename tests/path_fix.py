"""
path_fix.py - Neural Child Path Management
Created: 2024-03-21
Description: Handles Python path setup and environment verification for test execution.
Ensures reliable imports regardless of execution context.

Author: Dr. Celaya
Project: Neural Child + Meta-Learning
"""

import os
import sys
from pathlib import Path
from typing import Tuple, List

def find_project_root() -> Path:
    """
    Find the project root by looking for key marker files.
    Walks up the directory tree until it finds the project root.
    """
    current = Path.cwd()
    marker_files = ['.cursorrules', 'requirements.txt', 'main.py']
    
    # Walk up until we find the markers or hit the root
    while current != current.parent:
        if any(marker.exists() for marker in (current / mf for mf in marker_files)):
            return current
        current = current.parent
        
    # If we didn't find it, try one level up from the tests directory
    if 'tests' in str(Path(__file__)):
        return Path(__file__).parent.parent
        
    return Path.cwd()

def verify_project_structure(root: Path) -> Tuple[bool, List[str]]:
    """
    Verify the project has all required files and structure.
    """
    required_files = [
        'main.py',
        'memory_module.py',
        'child_model.py',
        'llm_module.py',
        'requirements.txt',
        '.cursorrules'
    ]
    
    missing = []
    for file in required_files:
        if not (root / file).exists():
            missing.append(file)
            
    return len(missing) == 0, missing

def setup_test_paths() -> Tuple[Path, bool]:
    """
    Set up Python path for test execution.
    Returns the project root and success status.
    """
    # Find project root
    project_root = find_project_root()
    
    # Verify project structure
    is_valid, missing = verify_project_structure(project_root)
    
    # Add to Python path if not already there
    project_root_str = str(project_root.absolute())
    if project_root_str not in sys.path:
        sys.path.insert(0, project_root_str)
        
    # Also add the tests directory
    tests_dir = project_root / 'tests'
    tests_dir_str = str(tests_dir.absolute())
    if tests_dir_str not in sys.path:
        sys.path.insert(0, tests_dir_str)
        
    return project_root, is_valid

# Execute path setup when module is imported
PROJECT_ROOT, IS_VALID = setup_test_paths()

# Print status
if not IS_VALID:
    print(f"⚠️  Warning: Project structure may be incomplete at {PROJECT_ROOT}")
    print("Please ensure you're running tests from the project root directory")
else:
    print(f"✓ Project root verified: {PROJECT_ROOT}")

# Export for use in other modules
__all__ = ['PROJECT_ROOT', 'IS_VALID', 'setup_test_paths', 'verify_project_structure'] 