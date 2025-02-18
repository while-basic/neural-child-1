"""
Test suite for mother-child emotional bonding scenarios.
Tests positive emotional interactions and attachment development.
"""

from main import MotherLLM, DigitalChild
from datetime import datetime
import time
import torch

def test_positive_bonding():
    """Test positive emotional bonding scenarios"""
    mother = MotherLLM()
    child = DigitalChild()
    
    scenarios = [
        # First meeting
        {
            'context': "First time meeting the child",
            'mother_action': "*opens arms warmly* Hello, little one! I'm so happy to meet you.",
            'expected_emotion': {'joy': 0.8, 'trust': 0.7, 'fear': 0.2, 'surprise': 0.6}
        },
        # Teaching moment
        {
            'context': "Child is learning something new",
            'mother_action': "*sits patiently* Let's learn something new together. I'll be right here to help you.",
            'expected_emotion': {'joy': 0.6, 'trust': 0.8, 'fear': 0.3, 'surprise': 0.4}
        },
        # Achievement
        {
            'context': "Child succeeds at a task",
            'mother_action': "*beams with pride* You did it! I knew you could. I'm so proud of you!",
            'expected_emotion': {'joy': 0.9, 'trust': 0.8, 'fear': 0.1, 'surprise': 0.7}
        },
        # Deep connection
        {
            'context': "Building emotional bond",
            'mother_action': "*shares a warm moment* I love our time together. You make me so happy.",
            'expected_emotion': {'joy': 0.8, 'trust': 0.9, 'fear': 0.1, 'surprise': 0.4}
        }
    ]
    
    run_emotional_test(mother, child, scenarios, "Positive Bonding") 