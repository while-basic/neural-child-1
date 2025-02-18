import torch
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from main import MotherLLM, DigitalChild

class MemoryTestSuite:
    def __init__(self):
        self.mother = MotherLLM()
        self.child = DigitalChild()
        self.test_results = []
        
    def run_all_tests(self):
        """Run all memory system tests"""
        print("🧪 Starting Memory System Tests\n")
        
        # Short-term Memory Tests
        self._test_short_term_capacity()
        self._test_short_term_retrieval()
        self._test_importance_calculation()
        
        # Long-term Memory Tests
        self._test_long_term_categorization()
        self._test_memory_importance_threshold()
        self._test_category_capacity()
        
        # Integration Tests
        self._test_memory_enhanced_responses()
        self._test_emotional_memory_integration()
        self._test_developmental_tracking()
        self._test_personal_info_memory()
        self._test_emotional_response_memory()
        
        self._print_test_summary()
    
    def _test_short_term_capacity(self):
        """Test short-term memory capacity limits"""
        print("📝 Testing Short-term Memory Capacity")
        
        # Generate more than max capacity interactions
        for i in range(60):  # Max is 50
            interaction = {
                'text': f"Test message {i}",
                'timestamp': datetime.now(),
                'importance': 0.5
            }
            self.mother._store_memory('interaction', interaction, 0.5)
            
        passed = len(self.mother.short_term_memory) == 50
        self.test_results.append({
            'name': 'Short-term Memory Capacity',
            'passed': passed,
            'expected': 50,
            'actual': len(self.mother.short_term_memory)
        })
        print(f"✓ Capacity maintained: {passed}")
    
    def _test_short_term_retrieval(self):
        """Test recent memory retrieval"""
        print("\n📝 Testing Short-term Memory Retrieval")
        
        # Add test memories
        test_messages = [
            "Hello, how are you?",
            "I'm learning about science",
            "Can you help me with math?"
        ]
        
        for msg in test_messages:
            interaction = {
                'text': msg,
                'timestamp': datetime.now(),
                'importance': 0.5
            }
            self.mother._store_memory('interaction', interaction, 0.5)
            
        # Test retrieval
        current_context = {'text': 'test context'}
        memories = self.mother._retrieve_relevant_memories(current_context)
        
        passed = any(test_messages[-1] in memory['content']['text'] for memory in memories)
        self.test_results.append({
            'name': 'Short-term Memory Retrieval',
            'passed': passed,
            'details': 'Recent memory successfully retrieved'
        })
        print(f"✓ Recent memory retrieval: {passed}")
    
    def _test_importance_calculation(self):
        """Test memory importance calculation"""
        print("\n📝 Testing Memory Importance Calculation")
        
        test_cases = [
            {
                'content': {
                    'text': 'Normal interaction',
                    'emotional_state': {'confidence': 0.5}
                },
                'expected_range': (0.5, 0.7)
            },
            {
                'content': {
                    'text': 'Emotional milestone',
                    'emotional_state': {'confidence': 0.9},
                    'stage_transition': True
                },
                'expected_range': (0.8, 1.0)
            }
        ]
        
        for case in test_cases:
            importance = self.mother._calculate_memory_importance(case['content'])
            min_expected, max_expected = case['expected_range']
            passed = min_expected <= importance <= max_expected
            
            self.test_results.append({
                'name': f"Importance Calculation - {case['content']['text']}",
                'passed': passed,
                'expected_range': case['expected_range'],
                'actual': importance
            })
            print(f"✓ Importance calculation ({case['content']['text']}): {passed}")
    
    def _test_long_term_categorization(self):
        """Test memory categorization"""
        print("\n📝 Testing Long-term Memory Categorization")
        
        test_cases = [
            {
                'text': 'I feel happy today',
                'expected_category': 'emotional_milestones'
            },
            {
                'text': 'I learned multiplication',
                'expected_category': 'learning_achievements'
            },
            {
                'text': 'I made a new friend',
                'expected_category': 'relationship_dynamics'
            }
        ]
        
        for case in test_cases:
            category = self.mother._determine_memory_category({'text': case['text']})
            passed = category == case['expected_category']
            
            self.test_results.append({
                'name': f"Memory Categorization - {case['text']}",
                'passed': passed,
                'expected': case['expected_category'],
                'actual': category
            })
            print(f"✓ Category determination ({case['text']}): {passed}")
    
    def _test_memory_importance_threshold(self):
        """Test memory importance threshold filtering"""
        print("\n📝 Testing Memory Importance Threshold")
        
        # Test with below-threshold memory
        low_importance = {
            'text': 'Low importance memory',
            'timestamp': datetime.now()
        }
        self.mother._store_memory('interaction', low_importance, 0.5)
        
        # Test with above-threshold memory
        high_importance = {
            'text': 'High importance memory',
            'timestamp': datetime.now(),
            'emotional_state': {'confidence': 0.9},
            'stage_transition': True
        }
        self.mother._store_memory('interaction', high_importance, 0.9)
        
        # Check if high importance memory was stored in long-term memory
        found_high_importance = False
        for category in self.mother.long_term_memory.values():
            if any(mem['content']['text'] == 'High importance memory' for mem in category):
                found_high_importance = True
                break
                
        self.test_results.append({
            'name': 'Memory Importance Threshold',
            'passed': found_high_importance,
            'details': 'High importance memory stored in long-term memory'
        })
        print(f"✓ Importance threshold filtering: {found_high_importance}")
    
    def _test_category_capacity(self):
        """Test category capacity limits"""
        print("\n📝 Testing Category Capacity")
        
        category = 'learning_achievements'
        max_category_size = self.mother.max_long_term_memories // 5
        
        # Add more than maximum memories to category
        for i in range(max_category_size + 10):
            memory = {
                'text': f'Learning achievement {i}',
                'timestamp': datetime.now(),
                'importance': 0.8
            }
            self.mother._store_memory('learning', memory, 0.8)
            
        passed = len(self.mother.long_term_memory[category]) <= max_category_size
        self.test_results.append({
            'name': 'Category Capacity',
            'passed': passed,
            'expected': max_category_size,
            'actual': len(self.mother.long_term_memory[category])
        })
        print(f"✓ Category capacity maintained: {passed}")
    
    def _test_memory_enhanced_responses(self):
        """Test if responses are enhanced with memory context"""
        print("\n📝 Testing Memory-Enhanced Responses")
        
        # Add some memories
        context = [
            {"role": "user", "content": "I love learning about space!"},
            {"role": "assistant", "content": "That's wonderful! Space is fascinating."}
        ]
        
        # Get response with memory context
        response = self.mother.respond("Tell me more about space", context)
        
        # Check if response contains reference to previous space discussion
        memory_enhanced = 'space' in response.lower()
        self.test_results.append({
            'name': 'Memory-Enhanced Responses',
            'passed': memory_enhanced,
            'details': 'Response contains context from previous interactions'
        })
        print(f"✓ Memory enhancement in responses: {memory_enhanced}")
    
    def _test_emotional_memory_integration(self):
        """Test emotional state integration with memories"""
        print("\n📝 Testing Emotional Memory Integration")
        
        # Create emotional interaction
        emotional_state = torch.tensor(
            [0.8, 0.7, 0.2, 0.3],
            device=self.child.brain.device
        )  # joy, trust, fear, surprise
        
        self.child.update_emotions(emotional_state)
        
        # Store memory with emotional state
        memory = {
            'text': 'Happy learning experience',
            'emotional_state': self.child.emotional_state,
            'timestamp': datetime.now()
        }
        self.mother._store_memory('emotional', memory, 0.8)
        
        # Verify emotional state was stored
        has_emotion = any(
            isinstance(mem['context'].get('emotional_state'), torch.Tensor)
            for category in self.mother.long_term_memory.values()
            for mem in category
        )
        
        self.test_results.append({
            'name': 'Emotional Memory Integration',
            'passed': has_emotion,
            'details': 'Emotional state successfully integrated with memory'
        })
        print(f"✓ Emotional state integration: {has_emotion}")
    
    def _test_developmental_tracking(self):
        """Test developmental stage tracking in memories"""
        print("\n📝 Testing Developmental Tracking")
        
        # Simulate development over time
        stages = [
            (datetime.now() - timedelta(days=60), 'EARLY_ELEMENTARY'),
            (datetime.now() - timedelta(days=30), 'MIDDLE_ELEMENTARY'),
            (datetime.now(), 'LATE_ELEMENTARY')
        ]
        
        for timestamp, stage in stages:
            memory = {
                'text': f'Development checkpoint: {stage}',
                'developmental_stage': stage,
                'timestamp': timestamp
            }
            self.mother._store_memory('developmental', memory, 0.9)
            
        # Verify developmental progression is tracked
        has_progression = any(
            mem['content'].get('developmental_stage') == 'LATE_ELEMENTARY'
            for category in self.mother.long_term_memory.values()
            for mem in category
        )
        
        self.test_results.append({
            'name': 'Developmental Tracking',
            'passed': has_progression,
            'details': 'Developmental stages successfully tracked in memories'
        })
        print(f"✓ Developmental tracking: {has_progression}")
    
    def _test_personal_info_memory(self):
        """Test remembering and recalling personal information"""
        print("\n📝 Testing Personal Information Memory")
        
        # Test sequence of interactions
        interactions = [
            "Hi, my name is Alice!",
            "I'm 8 years old",
            "I love playing piano",
            "What's my name?",
            "How old am I?",
            "What instrument do I play?"
        ]
        
        # Store initial interactions
        for msg in interactions[:3]:
            memory = {
                'text': msg,
                'timestamp': datetime.now(),
                'personal_info': True
            }
            self.mother._store_memory('personal_info', memory, 0.9)  # High importance for personal info
            
        # Test recall through responses
        for question in interactions[3:]:
            response = self.mother.respond(question, context=[
                {"role": "user", "content": msg} for msg in interactions[:3]
            ])
            
            # Check if response contains relevant information
            if "name" in question.lower():
                passed = "alice" in response.lower()
                detail = "remembered name"
            elif "old" in question.lower():
                passed = "8" in response
                detail = "remembered age"
            elif "instrument" in question.lower():
                passed = "piano" in response.lower()
                detail = "remembered hobby"
                
            self.test_results.append({
                'name': f'Personal Info - {detail}',
                'passed': passed,
                'details': f'Successfully {detail}' if passed else f'Failed to {detail}'
            })
            print(f"✓ Personal information ({detail}): {passed}")
    
    def _test_emotional_response_memory(self):
        """Test emotional response and memory system"""
        print("\n📝 Testing Emotional Response Memory")
        
        # Test sequence of emotional interactions
        emotional_messages = [
            ("I'm so happy today!", "JOY"),
            ("I'm feeling scared about the test tomorrow.", "FEAR"),
            ("I'm really proud of my drawing!", "PRIDE"),
            ("I miss my friend.", "SADNESS")
        ]
        
        # Test emotional responses and memory
        for message, expected_emotion in emotional_messages:
            # Get response and check for emotional expression
            response = self.mother.respond(message)
            
            # Verify emotional analysis
            current_emotion = self.mother.current_emotion
            emotion_correct = current_emotion['type'] == expected_emotion
            
            # Verify emotional memory storage
            memory_stored = any(
                mem['content'].get('emotion', {}).get('type') == expected_emotion
                for mem in self.mother.long_term_memory['emotional_memories']
            )
            
            # Verify emotional expression in response
            has_expression = '*' in response  # Check for emotional expression markers
            
            # Add test results
            self.test_results.append({
                'name': f'Emotional Response - {expected_emotion}',
                'passed': emotion_correct and memory_stored and has_expression,
                'details': f'Emotion: {current_emotion["type"]}, Memory Stored: {memory_stored}, Has Expression: {has_expression}'
            })
            print(f"✓ Emotional processing ({expected_emotion}): {emotion_correct and memory_stored and has_expression}")
            
    def _print_test_summary(self):
        """Print summary of all test results"""
        print("\n📊 Test Summary")
        print("=" * 50)
        
        passed_tests = sum(1 for test in self.test_results if test['passed'])
        total_tests = len(self.test_results)
        
        print(f"\nTotal Tests: {total_tests}")
        print(f"Passed: {passed_tests}")
        print(f"Failed: {total_tests - passed_tests}")
        print(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%")
        
        if total_tests - passed_tests > 0:
            print("\nFailed Tests:")
            for test in self.test_results:
                if not test['passed']:
                    print(f"- {test['name']}")
                    if 'expected' in test:
                        print(f"  Expected: {test['expected']}")
                        print(f"  Actual: {test['actual']}")
                    if 'details' in test:
                        print(f"  Details: {test['details']}")

if __name__ == "__main__":
    test_suite = MemoryTestSuite()
    test_suite.run_all_tests() 