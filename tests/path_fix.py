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
import logging
from pathlib import Path
from typing import Tuple, List, Optional

logger = logging.getLogger(__name__)

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

def setup_test_paths() -> Optional[str]:
    """
    Set up Python path for test modules.
    
    Adds the project root directory to Python path to allow importing from the main project.
    
    Returns:
        str: Path to project root if successful, None otherwise
    
    Raises:
        ImportError: If unable to set up paths correctly
    """
    try:
        # Get the directory containing this file
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Get project root (parent of tests directory)
        project_root = os.path.dirname(current_dir)
        
        # Add to Python path if not already there
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
            logger.debug(f"Added project root to Python path: {project_root}")
        
        return project_root
        
    except Exception as e:
        logger.error(f"Failed to set up test paths: {str(e)}")
        logger.error(f"Current Python path: {sys.path}")
        raise ImportError(f"Unable to set up test paths: {str(e)}")

# Set up paths when module is imported
project_root = setup_test_paths()

# Print status
if project_root:
    print(f"✓ Project root verified: {project_root}")
else:
    print("⚠️  Warning: Project structure may be incomplete")
    print("Please ensure you're running tests from the project root directory")

# Export for use in other modules
__all__ = ['project_root', 'verify_project_structure'] 