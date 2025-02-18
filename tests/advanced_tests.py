import sys
import os

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from child_model import DynamicNeuralChild
from meta_learning import MetaLearningSystem, NegativeBehaviorModule
import time
import random
import math

def run_attachment_test():
    """Test attachment patterns and separation anxiety"""
    print("👶 Testing Attachment Patterns")
    child = DynamicNeuralChild()
    meta_learner = MetaLearningSystem(child)
    negative_behavior = NegativeBehaviorModule()
    
    # Simulate attachment-separation cycles
    attachment_cycles = [
        {'phase': 'secure', 'duration': 5, 'base_trust': 0.8},
        {'phase': 'separation', 'duration': 3, 'base_trust': 0.3},
        {'phase': 'reunion', 'duration': 4, 'base_trust': 0.6},
        {'phase': 'extended_separation', 'duration': 6, 'base_trust': 0.2}
    ]
    
    for cycle in attachment_cycles:
        print(f"\n🔄 Phase: {cycle['phase'].upper()}")
        for t in range(cycle['duration']):
            # Generate emotional state with some randomness
            emotional_state = torch.tensor([
                cycle['base_trust'] + 0.2 * math.sin(t),  # Oscillating joy
                cycle['base_trust'],                      # Base trust
                0.8 - cycle['base_trust'],                # Inverse fear
                0.5 + 0.3 * math.cos(t)                  # Varying surprise
            ], device=child.device)
            
            response = child.update_emotions(emotional_state)
            negative_behavior.update_emotional_state({
                'reward_score': cycle['base_trust'],
                'success_metric': cycle['base_trust'],
                'complexity_rating': 0.5
            })
            
            print(f"Time Step {t}:")
            print(f"Emotional State: {child.express_feeling()}")
            print(f"Trust Level: {response['trust_level']:.2f}")
            print(f"Behaviors: {negative_behavior.get_emotional_summary()['behaviors']}")
            time.sleep(1)

def run_learning_frustration_test():
    """Test response to repeated failures and learning challenges"""
    print("📚 Testing Learning Frustration Response")
    child = DynamicNeuralChild()
    meta_learner = MetaLearningSystem(child)
    negative_behavior = NegativeBehaviorModule()
    
    # Simulate learning scenarios with varying difficulty
    learning_scenarios = [
        {'task': 'Easy Math', 'success_rate': 0.8, 'attempts': 3},
        {'task': 'Medium Reading', 'success_rate': 0.5, 'attempts': 4},
        {'task': 'Hard Puzzle', 'success_rate': 0.2, 'attempts': 5},
        {'task': 'Recovery Period', 'success_rate': 0.9, 'attempts': 3}
    ]
    
    for scenario in learning_scenarios:
        print(f"\n📝 Task: {scenario['task']}")
        for attempt in range(scenario['attempts']):
            # Simulate success/failure
            success = random.random() < scenario['success_rate']
            
            # Generate input based on success
            input_tensor = torch.randn(1, 768, device=child.device)
            modified_input = meta_learner.forward(input_tensor)
            response = child(modified_input)
            
            # Update emotional state based on success
            emotional_impact = torch.tensor([
                0.8 if success else 0.2,  # joy
                0.7 if success else 0.3,  # trust
                0.2 if success else 0.8,  # fear
                0.5,                      # surprise
            ], device=child.device)
            
            child.update_emotions(emotional_impact)
            negative_behavior.update_emotional_state({
                'reward_score': 0.9 if success else 0.1,
                'success_metric': 1.0 if success else 0.0,
                'complexity_rating': 0.7
            })
            
            print(f"Attempt {attempt + 1}:")
            print(f"Success: {'✅' if success else '❌'}")
            print(f"Emotional State: {child.express_feeling()}")
            print(f"Negative Behaviors: {negative_behavior.get_emotional_summary()['behaviors']}")
            time.sleep(1)

def run_social_interaction_test():
    """Test response to various social situations"""
    print("🤝 Testing Social Interaction Patterns")
    child = DynamicNeuralChild()
    meta_learner = MetaLearningSystem(child)
    negative_behavior = NegativeBehaviorModule()
    
    social_scenarios = [
        {
            'scenario': 'Peer Play',
            'interactions': [
                {'type': 'Cooperative', 'emotion': [0.8, 0.7, 0.2, 0.6]},
                {'type': 'Competitive', 'emotion': [0.6, 0.5, 0.4, 0.7]},
                {'type': 'Conflict', 'emotion': [0.3, 0.4, 0.7, 0.8]}
            ]
        },
        {
            'scenario': 'Group Activity',
            'interactions': [
                {'type': 'Leading', 'emotion': [0.7, 0.6, 0.3, 0.5]},
                {'type': 'Following', 'emotion': [0.6, 0.8, 0.2, 0.4]},
                {'type': 'Excluded', 'emotion': [0.2, 0.3, 0.8, 0.6]}
            ]
        },
        {
            'scenario': 'New Environment',
            'interactions': [
                {'type': 'Exploration', 'emotion': [0.7, 0.5, 0.4, 0.9]},
                {'type': 'Meeting Strangers', 'emotion': [0.4, 0.3, 0.7, 0.8]},
                {'type': 'Finding Friends', 'emotion': [0.8, 0.7, 0.3, 0.6]}
            ]
        }
    ]
    
    for scenario in social_scenarios:
        print(f"\n👥 Scenario: {scenario['scenario']}")
        for interaction in scenario['interactions']:
            emotional_state = torch.tensor(interaction['emotion'], device=child.device)
            response = child.update_emotions(emotional_state)
            
            # Update negative behavior based on interaction type
            stress_level = 0.7 if 'Conflict' in interaction['type'] or 'Excluded' in interaction['type'] else 0.3
            negative_behavior.update_emotional_state({
                'reward_score': 1 - stress_level,
                'success_metric': 1 - stress_level,
                'complexity_rating': 0.6
            })
            
            print(f"\nInteraction: {interaction['type']}")
            print(f"Emotional State: {child.express_feeling()}")
            print(f"Trust Level: {response['trust_level']:.2f}")
            print(f"Behaviors: {negative_behavior.get_emotional_summary()['behaviors']}")
            time.sleep(1)

def run_emotional_regulation_test():
    """Test emotional regulation and recovery capabilities"""
    print("🎭 Testing Emotional Regulation")
    child = DynamicNeuralChild()
    meta_learner = MetaLearningSystem(child)
    negative_behavior = NegativeBehaviorModule()
    
    # Test emotional recovery after intense experiences
    emotional_sequences = [
        {
            'phase': 'Intense Joy',
            'start': [0.9, 0.8, 0.1, 0.7],
            'recovery_steps': 5
        },
        {
            'phase': 'Deep Fear',
            'start': [0.1, 0.2, 0.9, 0.8],
            'recovery_steps': 6
        },
        {
            'phase': 'Mixed Emotions',
            'start': [0.7, 0.3, 0.6, 0.8],
            'recovery_steps': 4
        }
    ]
    
    for sequence in emotional_sequences:
        print(f"\n😊 Phase: {sequence['phase']}")
        
        # Initial intense emotion
        emotional_state = torch.tensor(sequence['start'], device=child.device)
        response = child.update_emotions(emotional_state)
        
        print("Initial State:")
        print(f"Emotional State: {child.express_feeling()}")
        print(f"Trust Level: {response['trust_level']:.2f}")
        
        # Recovery period
        for step in range(sequence['recovery_steps']):
            # Gradually return to baseline
            recovery_factor = 1 - (step + 1) / sequence['recovery_steps']
            current_state = emotional_state * recovery_factor + torch.tensor([0.5, 0.5, 0.3, 0.4], device=child.device) * (1 - recovery_factor)
            
            response = child.update_emotions(current_state)
            negative_behavior.update_emotional_state({
                'reward_score': 0.5 + 0.3 * (1 - recovery_factor),
                'success_metric': 0.6,
                'complexity_rating': 0.5
            })
            
            print(f"\nRecovery Step {step + 1}:")
            print(f"Emotional State: {child.express_feeling()}")
            print(f"Trust Level: {response['trust_level']:.2f}")
            print(f"Behaviors: {negative_behavior.get_emotional_summary()['behaviors']}")
            time.sleep(1)

if __name__ == "__main__":
    print("🧪 Advanced Neural Child Testing Suite")
    print("\nSelect a test to run:")
    print("1. Attachment Patterns Test")
    print("2. Learning Frustration Test")
    print("3. Social Interaction Test")
    print("4. Emotional Regulation Test")
    print("5. Run All Tests")
    
    choice = input("\nEnter test number (1-5): ")
    
    if choice == '1':
        run_attachment_test()
    elif choice == '2':
        run_learning_frustration_test()
    elif choice == '3':
        run_social_interaction_test()
    elif choice == '4':
        run_emotional_regulation_test()
    elif choice == '5':
        run_attachment_test()
        time.sleep(2)
        run_learning_frustration_test()
        time.sleep(2)
        run_social_interaction_test()
        time.sleep(2)
        run_emotional_regulation_test()
    else:
        print("Invalid choice. Please select 1-5.") 