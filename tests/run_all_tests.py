"""
run_all_tests.py - Neural Child Test Runner
Created: 2024-03-21
Description: Script to run all test suites in the Neural Child project.
Provides comprehensive test execution and result reporting.

Author: Dr. Celaya
Project: Neural Child + Meta-Learning
"""

import os
import sys
import logging
import datetime
from pathlib import Path
import unittest
import time

# Import path management (this will set up the Python path)
from path_fix import PROJECT_ROOT, IS_VALID, setup_test_paths
from test_utils import (
    STATUS_PASS, STATUS_FAIL, STATUS_TEST, STATUS_WARN, STATUS_STATS,
    UnicodeStreamHandler
)

# Configure logging
log_dir = Path(PROJECT_ROOT) / "logs"
log_dir.mkdir(exist_ok=True)

# Create timestamped log file
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = log_dir / f"full_test_run_{timestamp}.log"

# Configure logging with Unicode support
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file, encoding='utf-8'),
        UnicodeStreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def run_all_tests():
    """Run all test suites"""
    logger.info(f"\n{STATUS_TEST} Starting Full Test Suite")
    start_time = time.time()
    
    # Verify environment setup
    if not IS_VALID:
        logger.error(f"{STATUS_FAIL} Project structure verification failed")
        return False
        
    try:
        # Import test modules
        from experimental.advanced_consciousness_tests import run_experimental_tests
        from experimental.cognitive_enhancement_tests import run_enhancement_tests
        from emotional.test_mother_child_bonding import test_positive_bonding
        from emotional.test_emotional_distress import test_emotional_distress
        from experimental.quantum_evolution_tests import run_quantum_evolution_tests
        
        # Run each test suite
        test_suites = [
            ("Experimental Consciousness Tests", run_experimental_tests),
            ("Cognitive Enhancement Tests", run_enhancement_tests),
            ("Mother-Child Bonding Tests", test_positive_bonding),
            ("Emotional Distress Tests", test_emotional_distress),
            ("Quantum Evolution Tests", run_quantum_evolution_tests)
        ]
        
        results = []
        for suite_name, suite_func in test_suites:
            logger.info(f"\n=== Running {suite_name} ===")
            try:
                suite_result = suite_func()
                results.append({
                    'suite': suite_name,
                    'success': True,
                    'result': suite_result
                })
                logger.info(f"{STATUS_PASS} {suite_name} completed successfully")
            except Exception as e:
                logger.error(f"{STATUS_FAIL} Error in {suite_name}: {str(e)}")
                results.append({
                    'suite': suite_name,
                    'success': False,
                    'error': str(e)
                })
                
        # Print summary
        duration = time.time() - start_time
        logger.info(f"\n{STATUS_STATS} Test Run Summary")
        logger.info("=" * 50)
        logger.info(f"Duration: {duration:.2f} seconds")
        logger.info(f"Total suites: {len(test_suites)}")
        successful_suites = sum(1 for r in results if r['success'])
        logger.info(f"Successful: {successful_suites}")
        logger.info(f"Failed: {len(test_suites) - successful_suites}")
        
        # Print failed suites
        if len(test_suites) - successful_suites > 0:
            logger.info("\nFailed test suites:")
            for result in results:
                if not result['success']:
                    logger.error(f"{STATUS_FAIL} {result['suite']}")
                    logger.error(f"  Error: {result['error']}")
                    
        # Save results to file
        results_file = log_dir / f"test_summary_{timestamp}.log"
        with open(results_file, 'w', encoding='utf-8') as f:
            f.write("=== Neural Child Test Run Summary ===\n")
            f.write(f"Run Date: {datetime.datetime.now()}\n")
            f.write(f"Duration: {duration:.2f} seconds\n\n")
            f.write(f"Total Suites: {len(test_suites)}\n")
            f.write(f"Successful: {successful_suites}\n")
            f.write(f"Failed: {len(test_suites) - successful_suites}\n\n")
            
            if len(test_suites) - successful_suites > 0:
                f.write("Failed Suites:\n")
                for result in results:
                    if not result['success']:
                        f.write(f"- {result['suite']}\n")
                        f.write(f"  Error: {result['error']}\n")
                        
        logger.info(f"\nTest summary saved to: {results_file}")
        return all(r['success'] for r in results)
        
    except Exception as e:
        logger.error(f"{STATUS_FAIL} Critical error in test execution: {str(e)}")
        return False

if __name__ == "__main__":
    # Ensure we're in the tests directory
    os.chdir(Path(__file__).parent)
    
    # Set console to UTF-8 mode on Windows
    if sys.platform == 'win32':
        os.system('chcp 65001')
    
    success = run_all_tests()
    sys.exit(0 if success else 1) 