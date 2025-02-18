"""
tests/__init__.py - Neural Child Test Package Initialization
Created: 2024-03-21
Description: Initializes the test package and sets up proper path handling.

Author: Dr. Celaya
Project: Neural Child + Meta-Learning
"""

import os
import sys
import logging
from pathlib import Path
from typing import Tuple

def get_project_root() -> Tuple[str, bool]:
    """
    Get the absolute path to the project root and verify its validity.
    
    Returns:
        Tuple[str, bool]: Project root path and validity status
    """
    try:
        project_root = str(Path(__file__).parent.parent.absolute())
        
        # Verify this is actually the project root by checking for key files
        required_files = ['main.py', 'requirements.txt', '.cursorrules']
        missing_files = [f for f in required_files if not Path(project_root, f).exists()]
        
        if missing_files:
            print(f"Warning: Missing required files in project root: {', '.join(missing_files)}")
            return project_root, False
            
        return project_root, True
        
    except Exception as e:
        print(f"Error determining project root: {str(e)}")
        return str(Path.cwd()), False

# Get project root and verify
PROJECT_ROOT, is_valid = get_project_root()

# Add project root to Python path if not already there
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
    print(f"Added {PROJECT_ROOT} to Python path")
    
# Add tests directory to Python path
TESTS_DIR = str(Path(__file__).parent.absolute())
if TESTS_DIR not in sys.path:
    sys.path.insert(0, TESTS_DIR)

# Configure basic logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Log initialization status
logger = logging.getLogger(__name__)
if is_valid:
    logger.info(f"Neural Child test environment initialized at: {PROJECT_ROOT}")
else:
    logger.warning("Neural Child test environment initialization may be incomplete")
    logger.warning("Please ensure you're running tests from the project root directory") 