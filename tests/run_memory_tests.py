"""
tests/run_memory_tests.py
Created: 2024-03-21
Description: Script to run memory system tests for the Neural Child project.

This script provides a command-line interface to run memory tests individually
or as a complete suite.

Author: Dr. Celaya
Project: Neural Child + Meta-Learning
"""

import os
import sys
import logging
import argparse
from typing import Optional, List

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('memory_tests.log')
    ]
)

logger = logging.getLogger(__name__)

def setup_environment() -> bool:
    """
    Set up the test environment.
    
    Returns:
        bool: True if setup was successful, False otherwise
    """
    try:
        # Import path fix to set up Python path
        from path_fix import setup_test_paths
        project_root = setup_test_paths()
        logger.info(f"Project root set to: {project_root}")
        return True
    except ImportError as e:
        logger.error(f"Failed to set up test environment: {str(e)}")
        return False

def run_tests(test_names: Optional[List[str]] = None) -> int:
    """
    Run the specified memory tests or all tests if none specified.
    
    Args:
        test_names: Optional list of test names to run
        
    Returns:
        int: Exit code (0 for success, 1 for failure)
    """
    try:
        from memory_tests import MemoryTestSuite
        
        test_suite = MemoryTestSuite()
        
        if test_names:
            logger.info(f"Running specified tests: {test_names}")
            for test_name in test_names:
                test_method = getattr(test_suite, f"_test_{test_name}", None)
                if test_method:
                    test_method()
                else:
                    logger.error(f"Test not found: {test_name}")
        else:
            logger.info("Running all memory tests")
            test_suite.run_all_tests()
            
        return 0
        
    except Exception as e:
        logger.error(f"Error running tests: {str(e)}", exc_info=True)
        return 1

def main():
    """Main entry point for the test runner."""
    parser = argparse.ArgumentParser(description="Neural Child Memory Tests Runner")
    parser.add_argument(
        '--tests',
        nargs='*',
        help='Specific tests to run (e.g., short_term_capacity emotional_memory_integration)'
    )
    
    args = parser.parse_args()
    
    if not setup_environment():
        logger.error("Failed to set up test environment")
        return 1
        
    return run_tests(args.tests)

if __name__ == "__main__":
    sys.exit(main()) 