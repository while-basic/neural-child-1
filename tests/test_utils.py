"""
test_utils.py - Neural Child Test Utilities
Created: 2024-03-21
Description: Advanced test utilities for managing imports, paths, and common test functionality.
Provides sophisticated test orchestration and result analysis for the Neural Child project.

Author: Dr. Celaya
Project: Neural Child + Meta-Learning
"""

import os
import sys
import logging
import datetime
import codecs
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

# Import path management (this will set up the Python path)
from path_fix import PROJECT_ROOT, IS_VALID, setup_test_paths

# Configure logging with file output
log_dir = Path(PROJECT_ROOT) / "logs"
log_dir.mkdir(exist_ok=True)

# Create timestamped log file
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = log_dir / f"test_run_{timestamp}.log"

class UnicodeStreamHandler(logging.StreamHandler):
    """Custom stream handler that properly handles Unicode on Windows"""
    def __init__(self, stream=None):
        super().__init__(stream)
        self.encoding = 'utf-8'
        
    def emit(self, record):
        try:
            msg = self.format(record)
            stream = self.stream
            # Ensure the message is a string and encode properly
            if isinstance(msg, bytes):
                msg = msg.decode(self.encoding)
            # Force proper encoding for Windows console
            if sys.platform == 'win32':
                try:
                    stream.buffer.write(msg.encode(self.encoding) + b'\n')
                except AttributeError:
                    stream.write(msg + '\n')
            else:
                stream.write(msg + '\n')
            self.flush()
        except Exception:
            self.handleError(record)

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

# Test status indicators with fallbacks
STATUS_PASS = "✓" if sys.platform != 'win32' else "[PASS]"
STATUS_FAIL = "❌" if sys.platform != 'win32' else "[FAIL]"
STATUS_TEST = "🧪" if sys.platform != 'win32' else "[TEST]"
STATUS_WARN = "⚠️" if sys.platform != 'win32' else "[WARN]"
STATUS_STATS = "📊" if sys.platform != 'win32' else "[STATS]"

def verify_project_structure() -> Tuple[bool, str]:
    """
    Verify the project structure and required files exist.
    
    Returns:
        Tuple[bool, str]: Success status and error message if any
    """
    required_files = [
        'main.py',
        'memory_module.py',
        'child_model.py',
        'llm_module.py'
    ]
    
    for file in required_files:
        file_path = Path(PROJECT_ROOT) / file
        if not file_path.exists():
            return False, f"Required file {file} not found in {PROJECT_ROOT}"
            
    return True, "Project structure verified"

def setup_test_env() -> bool:
    """
    Setup test environment and verify imports.
    
    This function performs comprehensive environment setup:
    1. Verifies project structure
    2. Sets up Python path
    3. Validates imports
    4. Configures logging
    
    Returns:
        bool: True if setup successful, False otherwise
    """
    try:
        # Verify project structure
        structure_valid, error_msg = verify_project_structure()
        if not structure_valid:
            logger.error(error_msg)
            return False
            
        # Log environment info
        logger.debug(f"Project root: {PROJECT_ROOT}")
        logger.debug(f"Python path: {sys.path}")
        logger.debug(f"Current directory: {os.getcwd()}")
        logger.debug(f"Log file: {log_file}")
        
        # Try imports
        from main import MotherLLM, DigitalChild
        logger.debug("Successfully imported main modules")
        
        # Log system info
        logger.debug(f"Python version: {sys.version}")
        logger.debug(f"Platform: {sys.platform}")
        
        return True
        
    except ImportError as e:
        logger.error(f"Failed to import main modules: {str(e)}")
        logger.error("Please ensure you're running tests from the project root directory")
        logger.error(f"Current PYTHONPATH: {os.environ.get('PYTHONPATH', 'Not set')}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error during setup: {str(e)}")
        return False

def run_test_scenario(scenario: Dict, mother: Any, child: Any, context: str = None) -> Dict:
    """
    Run a single test scenario and return detailed results.
    
    Args:
        scenario (Dict): Test scenario configuration
        mother (MotherLLM): Mother LLM instance
        child (DigitalChild): Digital Child instance
        context (str, optional): Additional context for the test
        
    Returns:
        Dict: Test results including response, matches, and metadata
    """
    try:
        logger.info(f"\nTesting: {scenario['context']}")
        logger.debug(f"Prompt: {scenario['prompt']}")
        
        # Record start time
        start_time = datetime.datetime.now()
        
        # Get response with timeout handling
        try:
            response = mother.respond(scenario['prompt'])
        except Exception as e:
            logger.error(f"Error getting response: {str(e)}")
            response = "ERROR: Failed to get response"
            
        # Calculate duration
        duration = (datetime.datetime.now() - start_time).total_seconds()
        
        # Check for expected responses
        matches = []
        for expected in scenario['expected_responses']:
            if expected.lower() in response.lower():
                matches.append(expected)
                
        result = {
            'scenario': scenario['context'],
            'response': response,
            'matches': matches,
            'success': len(matches) > 0,
            'timestamp': datetime.datetime.now(),
            'duration': duration,
            'context': context
        }
        
        logger.info(f"Response: {response}")
        logger.info(f"Matched concepts: {', '.join(matches)}")
        logger.info(f"Duration: {duration:.2f} seconds")
        logger.info(f"Success: {'✓' if result['success'] else '❌'}")
        
        return result
        
    except Exception as e:
        logger.error(f"Error in scenario {scenario['context']}: {str(e)}")
        return {
            'scenario': scenario['context'],
            'error': str(e),
            'success': False,
            'timestamp': datetime.datetime.now(),
            'context': context
        }

def print_test_summary(results: List[Dict]):
    """
    Generate and print comprehensive test results summary.
    
    Args:
        results (List[Dict]): List of test results
    """
    logger.info(f"\n{STATUS_STATS} Test Summary")
    logger.info("=" * 50)
    
    total_tests = len(results)
    successful_tests = sum(1 for r in results if r.get('success', False))
    failed_tests = total_tests - successful_tests
    
    # Calculate statistics
    avg_duration = sum(r.get('duration', 0) for r in results) / total_tests if total_tests > 0 else 0
    success_rate = (successful_tests / total_tests * 100) if total_tests > 0 else 0
    
    logger.info(f"Total scenarios: {total_tests}")
    logger.info(f"Successful: {successful_tests}")
    logger.info(f"Failed: {failed_tests}")
    logger.info(f"Average duration: {avg_duration:.2f} seconds")
    logger.info(f"Success rate: {success_rate:.1f}%")
    
    # Print failed scenarios
    if failed_tests > 0:
        logger.info("\nFailed scenarios:")
        for result in results:
            if not result.get('success', False):
                logger.error(f"- {result['scenario']}")
                if 'error' in result:
                    logger.error(f"  Error: {result['error']}")
                    
    # Save detailed results to file
    results_file = log_dir / f"test_results_{timestamp}.log"
    with open(results_file, 'w', encoding='utf-8') as f:
        f.write("=== Neural Child Test Results ===\n")
        f.write(f"Test Run Date: {datetime.datetime.now()}\n\n")
        
        f.write("Summary:\n")
        f.write(f"Total Tests: {total_tests}\n")
        f.write(f"Successful: {successful_tests}\n")
        f.write(f"Failed: {failed_tests}\n")
        f.write(f"Success Rate: {success_rate:.1f}%\n")
        f.write(f"Average Duration: {avg_duration:.2f} seconds\n\n")
        
        f.write("Detailed Results:\n")
        f.write("=" * 50 + "\n\n")
        for result in results:
            f.write(f"Scenario: {result['scenario']}\n")
            f.write(f"Success: {result['success']}\n")
            f.write(f"Timestamp: {result.get('timestamp', 'N/A')}\n")
            f.write(f"Duration: {result.get('duration', 'N/A')} seconds\n")
            if 'error' in result:
                f.write(f"Error: {result['error']}\n")
            f.write(f"Response: {result.get('response', 'N/A')}\n")
            f.write(f"Matches: {', '.join(result.get('matches', []))}\n")
            if result.get('context'):
                f.write(f"Context: {result['context']}\n")
            f.write("\n" + "-"*50 + "\n\n")
            
    logger.info(f"\nDetailed results saved to: {results_file}")

def setup_test_logger() -> logging.Logger:
    """
    Set up a logger for test execution with proper formatting and Unicode support.
    
    Returns:
        logging.Logger: Configured logger instance
    """
    # Create logger
    logger = logging.getLogger('quantum_tests')
    logger.setLevel(logging.DEBUG)
    
    # Create log directory if it doesn't exist
    log_dir = Path(PROJECT_ROOT) / "logs" / "quantum_tests"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Create timestamped log file
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"quantum_test_{timestamp}.log"
    
    # Create handlers with Unicode support
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    console_handler = UnicodeStreamHandler()
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Set formatter for handlers
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

def verify_project_root() -> bool:
    """
    Verify the project root directory is properly set up.
    
    Returns:
        bool: True if project root is valid, False otherwise
    """
    if not IS_VALID:
        logger = logging.getLogger('quantum_tests')
        logger.error("Project structure verification failed")
        return False
        
    required_files = [
        'main.py',
        'child_model.py',
        'meta_learning.py',
        'requirements.txt'
    ]
    
    for file in required_files:
        if not (Path(PROJECT_ROOT) / file).exists():
            logger = logging.getLogger('quantum_tests')
            logger.error(f"Required file {file} not found in {PROJECT_ROOT}")
            return False
            
    return True 