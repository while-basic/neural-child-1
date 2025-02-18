import torch
from child_model import DynamicNeuralChild
from meta_learning import MetaLearningSystem, NegativeBehaviorModule
import time
import random

def run_stress_test():
    print("🧪 Initializing Neural Child Stress Test")
    child = DynamicNeuralChild()
    meta_learner = MetaLearningSystem(child)
    negative_behavior = NegativeBehaviorModule()
    
    # Stress Test 1: Rapid Emotional Changes
    print("\n🔄 Test 1: Rapid Emotional Changes")
    emotional_sequences = [
        {'joy': 0.9, 'trust': 0.2, 'fear': 0.1, 'surprise': 0.8},
        {'joy': 0.1, 'trust': 0.1, 'fear': 0.9, 'surprise': 0.9},
        {'joy': 0.8, 'trust': 0.9, 'fear': 0.1, 'surprise': 0.1},
        {'joy': 0.2, 'trust': 0.3, 'fear': 0.8, 'surprise': 0.7}
    ]
    
    for emotion in emotional_sequences:
        # Convert emotion dict to tensor
        emotion_tensor = torch.tensor([
            emotion['joy'],
            emotion['trust'],
            emotion['fear'],
            emotion['surprise']
        ], device=child.device)
        
        # Update emotional state
        response = child.update_emotions(emotion_tensor)
        
        # Update negative behavior system
        negative_behavior.update_emotional_state({
            'reward_score': 0.3,  # Low reward to simulate stress
            'complexity_rating': 0.8,  # High complexity
            'success_metric': 0.2  # Low success
        })
        
        print(f"Emotional State: {child.express_feeling()}")
        print(f"Trust Level: {response['trust_level']:.2f}")
        print(f"Negative Behaviors: {negative_behavior.get_emotional_summary()['behaviors']}")
        time.sleep(1)
    
    # Stress Test 2: Conflicting Information
    print("\n🔀 Test 2: Conflicting Information")
    conflicting_stimuli = [
        {'text': '[COMFORT] but with fear emotion', 'emotional_vector': torch.tensor([0.1, 0.2, 0.8, 0.7], device=child.device)},
        {'text': '[PLAY] but with sadness', 'emotional_vector': torch.tensor([0.2, 0.3, 0.6, 0.4], device=child.device)},
        {'text': '[LOVE] with anxiety', 'emotional_vector': torch.tensor([0.3, 0.4, 0.7, 0.8], device=child.device)}
    ]
    
    for stimulus in conflicting_stimuli:
        # Process through meta-learning system
        modified_input = meta_learner.forward(stimulus['emotional_vector'])
        response = child(modified_input)
        
        # Update negative behavior
        behavior_modification = negative_behavior.get_behavior_modification(response)
        
        print(f"Stimulus: {stimulus['text']}")
        print(f"Response: {child.express_feeling()}")
        print(f"Behavior Modification: {behavior_modification.mean().item():.2f}")
        time.sleep(1)
    
    # Stress Test 3: Complex Tasks Beyond Stage
    print("\n📚 Test 3: Complex Tasks Beyond Stage")
    advanced_tasks = [
        {'text': '[SOLVE] complex puzzle', 'complexity_rating': 0.9},
        {'text': '[EXPLAIN] quantum physics', 'complexity_rating': 1.0},
        {'text': '[ANALYZE] philosophical concept', 'complexity_rating': 0.95}
    ]
    
    for task in advanced_tasks:
        # Generate challenging input
        input_tensor = torch.randn(1, 768, device=child.device)
        
        # Process through meta-learning with behavior modification
        modified_input = meta_learner.forward(input_tensor)
        response = child(modified_input)
        
        # Update negative behavior with high stress
        negative_behavior.update_emotional_state({
            'reward_score': 0.1,  # Very low reward
            'success_metric': 0.1,  # Very low success
            'complexity_rating': task['complexity_rating']  # Very high complexity
        })
        
        print(f"Task: {task['text']}")
        print(f"Emotional Response: {child.express_feeling()}")
        print(f"Negative Behaviors: {negative_behavior.get_emotional_summary()['behaviors']}")
        time.sleep(1)
    
    # Stress Test 4: Environmental Pressure
    print("\n🌪️ Test 4: Environmental Pressure")
    stressors = [
        {'noise_level': 0.9, 'social_pressure': 0.8, 'time_pressure': 0.9},
        {'noise_level': 0.7, 'social_pressure': 0.9, 'time_pressure': 0.8},
        {'noise_level': 0.8, 'social_pressure': 0.7, 'time_pressure': 0.9}
    ]
    
    for stressor in stressors:
        # Combine stressors into emotional impact
        combined_stress = torch.tensor([
            0.1,  # joy
            0.2,  # trust
            0.8 * (stressor['noise_level'] + stressor['social_pressure'])/2,  # fear
            0.7 * stressor['time_pressure']  # surprise
        ], device=child.device)
        
        # Process through both systems
        response = child.update_emotions(combined_stress)
        negative_behavior.update_emotional_state({
            'reward_score': 0.2,
            'success_metric': 0.3,
            'complexity_rating': 0.8
        })
        
        print(f"Environmental Stressors: {stressor}")
        print(f"Child's Response: {child.express_feeling()}")
        print(f"Emotional State: {child.get_emotional_state()}")
        print(f"Negative Behaviors: {negative_behavior.get_emotional_summary()['behaviors']}")
        time.sleep(1)
    
    # Final Analysis
    print("\n📊 Final Analysis")
    print(f"Final Emotional State: {child.get_emotional_state()}")
    print(f"Final Expression: {child.express_feeling()}")
    print("\nNegative Behavior Summary:")
    print(negative_behavior.get_emotional_summary())

if __name__ == "__main__":
    run_stress_test() 