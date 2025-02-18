"""
Test suite for emotional distress scenarios.
Tests negative emotional interactions and recovery patterns.
"""

from main import MotherLLM, DigitalChild
from datetime import datetime
import time
import torch
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def test_emotional_distress():
    """Test negative emotional scenarios and recovery patterns"""
    mother = MotherLLM()
    child = DigitalChild()
    
    scenarios = [
        # Initial Fear
        {
            'context': "Child is scared of new environment",
            'mother_action': "*approaches gently* It's okay to be scared. I'm right here with you.",
            'expected_emotion': {'joy': 0.1, 'trust': 0.3, 'fear': 0.9, 'anxiety': 0.8}
        },
        # Deep Sadness
        {
            'context': "Child's favorite toy is broken",
            'mother_action': "*holds child close* I see how sad you are. It's okay to cry.",
            'expected_emotion': {'joy': 0.1, 'trust': 0.5, 'sadness': 0.9, 'fear': 0.4}
        },
        # Anger and Frustration
        {
            'context': "Child is having difficulty with a task",
            'mother_action': "*stays calm* I understand you're frustrated. Let's take a deep breath together.",
            'expected_emotion': {'joy': 0.1, 'trust': 0.4, 'anger': 0.8, 'frustration': 0.9}
        },
        # Emotional Overwhelm
        {
            'context': "Child is having a meltdown",
            'mother_action': "*maintains loving presence* Everything feels big right now. I'm here while you feel these big feelings.",
            'expected_emotion': {'joy': 0.1, 'trust': 0.3, 'fear': 0.8, 'overwhelm': 0.9}
        },
        # Recovery Phase
        {
            'context': "Child begins to calm down",
            'mother_action': "*offers comfort* You're doing so well. Let's take all the time you need.",
            'expected_emotion': {'joy': 0.3, 'trust': 0.7, 'fear': 0.4, 'relief': 0.6}
        }
    ]
    
    for scenario in scenarios:
        logger.info(f"Testing scenario: {scenario['context']}")
        start_time = time.time()
        
        # Record initial emotional states
        mother_initial_state = mother.get_emotional_state()
        child_initial_state = child.get_emotional_state()
        
        # Process interaction
        mother_response = mother.process_scenario(scenario['context'], scenario['mother_action'])
        child_response = child.process_mother_action(mother_response)
        
        # Verify emotional changes
        child_final_state = child.get_emotional_state()
        emotional_shift = {
            emotion: child_final_state[emotion] - child_initial_state.get(emotion, 0)
            for emotion in child_final_state
        }
        
        # Log detailed results
        logger.debug(f"Emotional shift: {emotional_shift}")
        logger.debug(f"Expected emotions: {scenario['expected_emotion']}")
        logger.debug(f"Processing time: {time.time() - start_time:.2f} seconds")
        
        # Verify emotional alignment
        for emotion, expected_value in scenario['expected_emotion'].items():
            actual_value = child_final_state.get(emotion, 0)
            assert abs(actual_value - expected_value) <= 0.2, \
                f"Emotion {emotion} mismatch: expected {expected_value}, got {actual_value}"
            
        logger.info(f"Scenario completed successfully: {scenario['context']}")
        
if __name__ == "__main__":
    test_emotional_distress() 