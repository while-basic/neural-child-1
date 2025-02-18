"""
Test runner for all emotional interaction tests.
Executes both positive and negative emotional scenarios and generates detailed reports.
"""

import unittest
import sys
import os
import logging
from datetime import datetime
from pathlib import Path
from test_mother_child_bonding import test_positive_bonding
from test_emotional_distress import test_emotional_distress

# Configure logging
log_dir = Path("logs/emotional_tests")
log_dir.mkdir(parents=True, exist_ok=True)
log_file = log_dir / f"emotional_test_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

class EmotionalTestSuite(unittest.TestCase):
    """Test suite for running all emotional interaction tests"""
    
    def setUp(self):
        """Set up test environment"""
        logger.info("Initializing Emotional Test Suite")
        self.start_time = datetime.now()
        
    def test_positive_scenarios(self):
        """Run positive emotional bonding tests"""
        logger.info("Starting positive emotional bonding tests")
        try:
            test_positive_bonding()
            logger.info("Positive emotional bonding tests completed successfully")
        except Exception as e:
            logger.error(f"Error in positive bonding tests: {str(e)}")
            raise
            
    def test_negative_scenarios(self):
        """Run emotional distress and recovery tests"""
        logger.info("Starting emotional distress tests")
        try:
            test_emotional_distress()
            logger.info("Emotional distress tests completed successfully")
        except Exception as e:
            logger.error(f"Error in emotional distress tests: {str(e)}")
            raise
            
    def tearDown(self):
        """Clean up and generate test report"""
        duration = datetime.now() - self.start_time
        logger.info(f"Test suite completed in {duration.total_seconds():.2f} seconds")
        
        # Generate detailed report
        report_path = log_dir / f"emotional_test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(report_path, 'w') as f:
            f.write("=== Neural Child Emotional Test Report ===\n\n")
            f.write(f"Test Run Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Duration: {duration.total_seconds():.2f} seconds\n\n")
            
            # Add test results
            f.write("Test Results:\n")
            f.write("- Positive Bonding Scenarios: COMPLETED\n")
            f.write("- Emotional Distress Scenarios: COMPLETED\n\n")
            
            # Add log summary
            f.write("Log File: {log_file}\n")
            f.write("\nFor detailed logs, please check the log file.\n")
            
        logger.info(f"Test report generated: {report_path}")

if __name__ == "__main__":
    unittest.main(verbosity=2) 