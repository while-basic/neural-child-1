import sys
import os
import torch
import time
import random
import math
from typing import Dict, List, Any

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from child_model import DynamicNeuralChild
from meta_learning import MetaLearningSystem, NegativeBehaviorModule
from emotional_regulation import EmotionalRegulation
from psychological_components import TheoryOfMind, AttachmentSystem, DefenseMechanisms
from curriculum_manager import DevelopmentalSystem, DevelopmentalStage

class AdvancedDevelopmentalTester:
    def __init__(self):
        self.child = DynamicNeuralChild()
        self.meta_learner = MetaLearningSystem(self.child)
        self.negative_behavior = NegativeBehaviorModule()
        self.emotional_system = EmotionalRegulation()
        self.theory_of_mind = TheoryOfMind()
        self.attachment = AttachmentSystem()
        self.defense_mechanisms = DefenseMechanisms()
        self.developmental_system = DevelopmentalSystem()
        
    def run_early_elementary_test(self):
        """Test advanced learning capabilities (7-8 years)"""
        print("\n🎓 Testing Early Elementary Stage (7-8 years)")
        
        research_projects = [
            {
                'topic': 'Space Exploration',
                'complexity': 0.7,
                'collaboration_required': True,
                'presentation_needed': True,
                'steps': [
                    {'action': '[RESEARCH]', 'difficulty': 0.6},
                    {'action': '[PROJECT]', 'difficulty': 0.7},
                    {'action': '[COLLABORATE]', 'difficulty': 0.8},
                    {'action': '[PRESENT]', 'difficulty': 0.9}
                ]
            },
            {
                'topic': 'Environmental Science',
                'complexity': 0.8,
                'collaboration_required': True,
                'presentation_needed': True,
                'steps': [
                    {'action': '[INVESTIGATE]', 'difficulty': 0.7},
                    {'action': '[ANALYZE]', 'difficulty': 0.8},
                    {'action': '[TEAMWORK]', 'difficulty': 0.7},
                    {'action': '[PRESENT]', 'difficulty': 0.8}
                ]
            }
        ]
        
        for project in research_projects:
            print(f"\n📚 Project: {project['topic']}")
            
            for step in project['steps']:
                # Generate complex input based on project requirements
                input_tensor = torch.randn(1, 768, device=self.child.device) * step['difficulty']
                
                # Process through meta-learning system
                modified_input = self.meta_learner.forward(input_tensor)
                response = self.child(modified_input)
                
                # Update emotional state based on complexity
                emotional_impact = torch.tensor([
                    0.7 - step['difficulty'] * 0.3,  # joy decreases with difficulty
                    0.6,                             # maintain moderate trust
                    0.3 + step['difficulty'] * 0.3,  # fear increases with difficulty
                    0.5 + step['difficulty'] * 0.2   # surprise increases with difficulty
                ], device=self.child.device)
                
                self.child.update_emotions(emotional_impact)
                
                # Update negative behavior system
                self.negative_behavior.update_emotional_state({
                    'reward_score': 0.7 - step['difficulty'] * 0.2,
                    'success_metric': 0.8 - step['difficulty'] * 0.3,
                    'complexity_rating': step['difficulty']
                })
                
                print(f"\nStep: {step['action']}")
                print(f"Emotional State: {self.child.express_feeling()}")
                print(f"Behaviors: {self.negative_behavior.get_emotional_summary()['behaviors']}")
                time.sleep(1)
    
    def run_middle_elementary_test(self):
        """Test complex problem-solving and leadership (8-9 years)"""
        print("\n🧩 Testing Middle Elementary Stage (8-9 years)")
        
        challenges = [
            {
                'type': 'Scientific Method',
                'steps': [
                    {'action': '[HYPOTHESIZE]', 'leadership': 0.7, 'complexity': 0.8},
                    {'action': '[EXPERIMENT]', 'leadership': 0.8, 'complexity': 0.7},
                    {'action': '[ANALYZE]', 'leadership': 0.6, 'complexity': 0.9},
                    {'action': '[CONCLUDE]', 'leadership': 0.9, 'complexity': 0.8}
                ]
            },
            {
                'type': 'Group Project',
                'steps': [
                    {'action': '[PLAN]', 'leadership': 0.9, 'complexity': 0.7},
                    {'action': '[DELEGATE]', 'leadership': 0.8, 'complexity': 0.8},
                    {'action': '[COORDINATE]', 'leadership': 0.9, 'complexity': 0.7},
                    {'action': '[DELIVER]', 'leadership': 0.7, 'complexity': 0.9}
                ]
            }
        ]
        
        for challenge in challenges:
            print(f"\n🎯 Challenge: {challenge['type']}")
            
            for step in challenge['steps']:
                # Generate input incorporating leadership component
                base_input = torch.randn(1, 768, device=self.child.device)
                leadership_factor = step['leadership'] * torch.ones_like(base_input)
                input_tensor = base_input * leadership_factor
                
                # Process with meta-learning
                modified_input = self.meta_learner.forward(input_tensor)
                response = self.child(modified_input)
                
                # Complex emotional update based on leadership and complexity
                emotional_impact = torch.tensor([
                    0.6 + step['leadership'] * 0.3,  # joy increases with leadership
                    0.7,                             # maintain high trust
                    0.2 + step['complexity'] * 0.2,  # slight fear with complexity
                    0.4 + step['complexity'] * 0.3   # surprise with complexity
                ], device=self.child.device)
                
                self.child.update_emotions(emotional_impact)
                
                # Update negative behavior with leadership consideration
                self.negative_behavior.update_emotional_state({
                    'reward_score': step['leadership'],
                    'success_metric': 0.9 - step['complexity'] * 0.2,
                    'complexity_rating': step['complexity']
                })
                
                print(f"\nStep: {step['action']}")
                print(f"Leadership Score: {step['leadership']:.2f}")
                print(f"Emotional State: {self.child.express_feeling()}")
                print(f"Behaviors: {self.negative_behavior.get_emotional_summary()['behaviors']}")
                time.sleep(1)
    
    def run_late_elementary_test(self):
        """Test advanced analytical and mentorship capabilities (9-11 years)"""
        print("\n🔬 Testing Late Elementary Stage (9-11 years)")
        
        research_scenarios = [
            {
                'topic': 'Advanced Mathematics',
                'activities': [
                    {'action': '[SYNTHESIZE]', 'mentorship': 0.7, 'analysis': 0.9},
                    {'action': '[CRITIQUE]', 'mentorship': 0.8, 'analysis': 0.8},
                    {'action': '[INNOVATE]', 'mentorship': 0.6, 'analysis': 0.9},
                    {'action': '[MENTOR]', 'mentorship': 0.9, 'analysis': 0.7}
                ]
            },
            {
                'topic': 'Literature Analysis',
                'activities': [
                    {'action': '[ANALYZE]', 'mentorship': 0.6, 'analysis': 0.9},
                    {'action': '[INTERPRET]', 'mentorship': 0.7, 'analysis': 0.8},
                    {'action': '[TEACH]', 'mentorship': 0.9, 'analysis': 0.7},
                    {'action': '[GUIDE]', 'mentorship': 0.8, 'analysis': 0.8}
                ]
            }
        ]
        
        for scenario in research_scenarios:
            print(f"\n📚 Research Topic: {scenario['topic']}")
            
            for activity in scenario['activities']:
                # Generate complex analytical input
                analytical_input = torch.randn(1, 768, device=self.child.device) * activity['analysis']
                
                # Process through meta-learning with mentorship consideration
                modified_input = self.meta_learner.forward(analytical_input)
                response = self.child(modified_input)
                
                # Update emotional state considering both analysis and mentorship
                emotional_impact = torch.tensor([
                    0.5 + activity['mentorship'] * 0.4,  # joy from mentoring
                    0.6 + activity['mentorship'] * 0.3,  # trust from mentoring
                    0.2 + activity['analysis'] * 0.2,    # slight anxiety from analysis
                    0.4 + activity['analysis'] * 0.4     # surprise from discoveries
                ], device=self.child.device)
                
                self.child.update_emotions(emotional_impact)
                
                # Update negative behavior with complex factors
                self.negative_behavior.update_emotional_state({
                    'reward_score': (activity['mentorship'] + activity['analysis']) / 2,
                    'success_metric': activity['mentorship'],
                    'complexity_rating': activity['analysis']
                })
                
                print(f"\nActivity: {activity['action']}")
                print(f"Analysis Level: {activity['analysis']:.2f}")
                print(f"Mentorship Level: {activity['mentorship']:.2f}")
                print(f"Emotional State: {self.child.express_feeling()}")
                print(f"Behaviors: {self.negative_behavior.get_emotional_summary()['behaviors']}")
                time.sleep(1)
    
    def run_early_adolescence_test(self):
        """Test identity development and abstract reasoning (11-13 years)"""
        print("\n🤔 Testing Early Adolescence Stage (11-13 years)")
        
        identity_scenarios = [
            {
                'theme': 'Personal Values',
                'explorations': [
                    {'action': '[EXPLORE]', 'abstraction': 0.8, 'emotional_intensity': 0.7},
                    {'action': '[QUESTION]', 'abstraction': 0.9, 'emotional_intensity': 0.8},
                    {'action': '[CHALLENGE]', 'abstraction': 0.7, 'emotional_intensity': 0.9},
                    {'action': '[UNDERSTAND]', 'abstraction': 0.8, 'emotional_intensity': 0.7}
                ]
            },
            {
                'theme': 'Social Identity',
                'explorations': [
                    {'action': '[REFLECT]', 'abstraction': 0.7, 'emotional_intensity': 0.8},
                    {'action': '[EXPRESS]', 'abstraction': 0.8, 'emotional_intensity': 0.9},
                    {'action': '[ADAPT]', 'abstraction': 0.9, 'emotional_intensity': 0.7},
                    {'action': '[INTEGRATE]', 'abstraction': 0.8, 'emotional_intensity': 0.8}
                ]
            }
        ]
        
        for scenario in identity_scenarios:
            print(f"\n🎭 Identity Theme: {scenario['theme']}")
            
            for exploration in scenario['explorations']:
                # Generate abstract reasoning input
                abstract_input = torch.randn(1, 768, device=self.child.device) * exploration['abstraction']
                
                # Process through meta-learning with emotional consideration
                modified_input = self.meta_learner.forward(abstract_input)
                response = self.child(modified_input)
                
                # Complex emotional update based on identity exploration
                emotional_impact = torch.tensor([
                    0.4 + random.random() * 0.4,     # variable joy
                    0.5 + random.random() * 0.3,     # variable trust
                    0.3 + random.random() * 0.4,     # variable fear
                    0.6 + random.random() * 0.3      # high surprise
                ], device=self.child.device)
                
                self.child.update_emotions(emotional_impact)
                
                # Update negative behavior with identity exploration factors
                self.negative_behavior.update_emotional_state({
                    'reward_score': 0.5 + random.random() * 0.4,
                    'success_metric': exploration['abstraction'],
                    'complexity_rating': exploration['emotional_intensity']
                })
                
                print(f"\nExploration: {exploration['action']}")
                print(f"Abstraction Level: {exploration['abstraction']:.2f}")
                print(f"Emotional Intensity: {exploration['emotional_intensity']:.2f}")
                print(f"Emotional State: {self.child.express_feeling()}")
                print(f"Behaviors: {self.negative_behavior.get_emotional_summary()['behaviors']}")
                time.sleep(1)

def main():
    tester = AdvancedDevelopmentalTester()
    
    print("🧪 Advanced Developmental Stage Testing Suite")
    print("\nSelect a test to run:")
    print("1. Early Elementary Stage Test (7-8 years)")
    print("2. Middle Elementary Stage Test (8-9 years)")
    print("3. Late Elementary Stage Test (9-11 years)")
    print("4. Early Adolescence Stage Test (11-13 years)")
    print("5. Run All Tests")
    
    choice = input("\nEnter test number (1-5): ")
    
    if choice == '1':
        tester.run_early_elementary_test()
    elif choice == '2':
        tester.run_middle_elementary_test()
    elif choice == '3':
        tester.run_late_elementary_test()
    elif choice == '4':
        tester.run_early_adolescence_test()
    elif choice == '5':
        print("\n🚀 Running all advanced developmental tests...")
        tester.run_early_elementary_test()
        time.sleep(2)
        tester.run_middle_elementary_test()
        time.sleep(2)
        tester.run_late_elementary_test()
        time.sleep(2)
        tester.run_early_adolescence_test()
    else:
        print("Invalid choice. Please select 1-5.")

if __name__ == "__main__":
    main() 