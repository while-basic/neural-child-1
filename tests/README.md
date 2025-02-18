# Neural Child Test Suite

## Overview
This directory contains automated tests for the Neural Child project, organized by test category. Each category focuses on different aspects of the AI's development and interaction capabilities.

## Directory Structure
```
tests/
├── emotional/           # Tests for emotional interactions and responses
│   ├── test_mother_child_bonding.py    # Positive emotional scenarios
│   ├── test_emotional_distress.py      # Negative emotional scenarios
│   └── run_emotional_tests.py          # Test runner for all emotional tests
├── memory/             # Tests for memory and learning capabilities
├── developmental/      # Tests for developmental milestones
└── integration/        # End-to-end integration tests
```

## Running Tests

### Emotional Tests
To run all emotional tests:
```bash
python tests/emotional/run_emotional_tests.py
```

Test results and detailed logs will be saved in `logs/emotional_tests/`.

### Individual Test Categories
- Run positive bonding tests:
  ```bash
  python tests/emotional/test_mother_child_bonding.py
  ```
- Run emotional distress tests:
  ```bash
  python tests/emotional/test_emotional_distress.py
  ```

## Test Reports
Test reports are automatically generated and stored in:
- Emotional test reports: `logs/emotional_tests/`
- Each report includes:
  - Test execution timestamp
  - Duration of test run
  - Test results summary
  - Detailed logs of emotional responses
  - Any errors or exceptions encountered

## Adding New Tests
1. Create a new test file in the appropriate category directory
2. Follow the existing test structure and naming conventions
3. Update the relevant test runner to include your new test
4. Add documentation for your test in this README

## Requirements
- Python 3.8+
- PyTorch
- Logging module
- Unittest framework

## Best Practices
- Each test should be independent and self-contained
- Include detailed logging for debugging
- Verify both positive and negative scenarios
- Test edge cases and boundary conditions
- Document expected behaviors and outcomes

## Contributing
When adding new tests:
1. Follow the established directory structure
2. Include appropriate logging and error handling
3. Update this README with any new test categories
4. Ensure all tests can be run independently

## Contact
For questions or issues related to testing, please contact Dr. Celaya. 