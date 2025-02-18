"""
quantum_evolution_tests.py - Test suite for quantum emotional processing and neural evolution capabilities
Created: 2025-02-18
Description: Tests quantum emotional processing and neural evolution capabilities.
Tests both individual components and their integration.

Author: Dr. Celaya
Project: Neural Child + Meta-Learning
"""

import sys
import os
import torch
import logging
from datetime import datetime
from typing import Dict, Any
from pathlib import Path

# Add project root to Python path
project_root = str(Path(__file__).parent.parent.parent.absolute())
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import test utilities
from tests.test_utils import setup_test_logger, verify_project_root
from child_model import DynamicNeuralChild
from meta_learning import MetaLearningSystem

# Initialize logging
logger = setup_test_logger()

class QuantumEvolutionTests:
    def __init__(self):
        """Initialize test environment"""
        try:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.hidden_size = 256
            
            logger.info(f"Initializing test environment on device: {self.device}")
            self.child = DynamicNeuralChild(device=str(self.device), hidden_size=self.hidden_size)
            self.meta_learner = MetaLearningSystem(self.child)
            
            # Initialize quantum metrics
            self.quantum_metrics = {
                'coherence_history': [],
                'entanglement_strength': [],
                'superposition_stability': []
            }
            
            # Initialize test data
            self.test_input = torch.randn(1, 768, device=self.device)
            self.test_emotional_state = torch.tensor([[0.5, 0.3, 0.2, 0.4]], device=self.device)
            
            logger.info("✓ Test environment initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Error initializing test environment: {str(e)}")
            raise
        
    def test_quantum_emotional_processing(self):
        """Test quantum emotional state processing"""
        try:
            logger.info("\nTesting: Superposition stability")
            # Test superposition stability
            output = self.child(self.test_input)
            coherence = torch.mean(torch.abs(output)).item()
            stability = torch.std(output).item()
            max_prob = torch.max(torch.softmax(output, dim=1)).item()
            
            logger.info(f"Coherence: {coherence:.3f}")
            logger.info(f"Stability: {stability:.3f}")
            logger.info(f"Max Probability: {max_prob:.3f}")
            
            # Test emotional collapse
            logger.info("\nTesting: Emotional collapse")
            collapse_input = torch.randn(1, 768, device=self.device) * 2.0
            collapse_output = self.child(collapse_input)
            coherence = torch.mean(torch.abs(collapse_output)).item()
            stability = torch.std(collapse_output).item()
            max_prob = torch.max(torch.softmax(collapse_output, dim=1)).item()
            
            logger.info(f"Coherence: {coherence:.3f}")
            logger.info(f"Stability: {stability:.3f}")
            logger.info(f"Max Probability: {max_prob:.3f}")
            
            # Test entanglement preservation
            logger.info("\nTesting: Entanglement preservation")
            entangled_input = self.test_input + 0.1 * torch.randn_like(self.test_input)
            entangled_output = self.child(entangled_input)
            coherence = torch.mean(torch.abs(entangled_output)).item()
            stability = torch.std(entangled_output).item()
            max_prob = torch.max(torch.softmax(entangled_output, dim=1)).item()
            
            logger.info(f"Coherence: {coherence:.3f}")
            logger.info(f"Stability: {stability:.3f}")
            logger.info(f"Max Probability: {max_prob:.3f}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error in quantum emotional processing test: {str(e)}")
            return False
        
    def test_neural_evolution(self):
        """Test neural architecture evolution"""
        try:
            logger.info("\n🧬 Testing Neural Evolution")
            
            # Test network growth
            logger.info("\nTesting: Network growth")
            initial_layers = len(self.child.layers)
            self.child.add_layer()
            complexity_change = (len(self.child.layers) - initial_layers) / initial_layers
            
            logger.info(f"Complexity Change: {complexity_change:.3f}")
            logger.info(f"Mutations: {len(self.meta_learner.metric_history.get('adaptation_scores', []))}")
            
            # Test network pruning
            logger.info("\nTesting: Network pruning")
            performance_metrics = {
                'reward_score': 0.8,
                'success_metric': 0.7,
                'complexity_rating': 0.6
            }
            
            evolution_result = self.meta_learner.meta_update(performance_metrics)
            complexity_change = evolution_result if isinstance(evolution_result, float) else 0.0
            
            logger.info(f"Complexity Change: {complexity_change:.3f}")
            logger.info(f"Mutations: {len(self.meta_learner.metric_history.get('adaptation_scores', []))}")
            
            # Test mutation events
            logger.info("\nTesting: Mutation events")
            for _ in range(3):  # Test multiple mutations
                layer_idx = 1  # Test modifying the first hidden layer
                self.child.modify_layer(layer_idx)
                
                # Verify layer modification
                modified_output = self.child(self.test_input)
                assert modified_output.size() == (1, 4), f"Unexpected output size: {modified_output.size()}"
                
            logger.info("✓ Mutation tests completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error in neural evolution test: {str(e)}")
            return False
        
    def test_integrated_learning(self):
        """Test integration of quantum processing and neural evolution"""
        try:
            logger.info("\n🔄 Testing Integrated Learning")
            
            # Prepare test data with correct shapes
            test_sequence = [
                (self.test_input, self.test_emotional_state.squeeze(0)) for _ in range(5)
            ]
            
            # Run learning sequence
            for input_data, target_state in test_sequence:
                # Forward pass
                output = self.child(input_data)
                
                # Ensure target state has batch dimension
                if target_state.dim() == 1:
                    target_state = target_state.unsqueeze(0)
                
                # Update emotional state
                emotional_state = self.child.update_emotions(target_state)
                
                # Verify output and state
                assert output.size() == (1, 4), f"Invalid output size: {output.size()}"
                assert isinstance(emotional_state, dict), "Invalid emotional state type"
                
                # Calculate mean squared error for reward
                mse = torch.nn.functional.mse_loss(output, target_state)
                reward = 1.0 - mse.item()  # Convert loss to reward (0 to 1)
                
                # Update meta-learning system
                performance_metrics = {
                    'reward_score': reward,
                    'success_metric': 0.7,
                    'complexity_rating': 0.6
                }
                self.meta_learner.meta_update(performance_metrics)
                
                # Log progress
                logger.info(f"Step MSE: {mse.item():.3f}, Reward: {reward:.3f}")
            
            logger.info("✓ Integrated learning tests completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error in integrated learning test: {str(e)}")
            logger.error(f"Input shape: {input_data.size()}")
            logger.error(f"Target shape: {target_state.size()}")
            logger.error(f"Output shape: {output.size() if 'output' in locals() else 'N/A'}")
            return False
        
    def save_test_results(self):
        """Save detailed test results"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_dir = Path(project_root) / "logs" / "quantum_tests"
            log_dir.mkdir(parents=True, exist_ok=True)
            
            results_path = log_dir / f"quantum_evolution_test_{timestamp}.log"
            
            with open(results_path, 'w', encoding='utf-8') as f:
                f.write(f"Quantum Evolution Test Results - {timestamp}\n")
                f.write("-" * 50 + "\n")
                f.write(f"Device: {self.device}\n")
                f.write(f"Hidden Size: {self.hidden_size}\n")
                f.write(f"Network Depth: {len(self.child.layers)}\n")
                f.write(f"Quantum Metrics:\n")
                for metric, values in self.quantum_metrics.items():
                    f.write(f"  {metric}: {values}\n")
                    
            logger.info(f"✓ Test results saved to {results_path}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error saving test results: {str(e)}")
            return False

def run_quantum_evolution_tests():
    """Run all quantum evolution tests"""
    try:
        if not verify_project_root():
            logger.error("❌ Project root verification failed")
            return False
            
        logger.info("\n🧪 Starting Quantum Evolution Test Suite")
        
        # Initialize test environment
        logger.info("Initializing quantum and evolution test environment...")
        test_suite = QuantumEvolutionTests()
        logger.info("✓ Test environment initialized")
        
        # Run tests
        results = []
        results.append(test_suite.test_quantum_emotional_processing())
        results.append(test_suite.test_neural_evolution())
        results.append(test_suite.test_integrated_learning())
        
        # Save results
        test_suite.save_test_results()
        
        success = all(results)
        if success:
            logger.info("\n✨ All tests completed successfully!")
        else:
            logger.error("\n❌ Some tests failed. Check logs for details.")
            
        return success
        
    except Exception as e:
        logger.error(f"❌ Critical error in test execution: {str(e)}")
        return False

if __name__ == "__main__":
    success = run_quantum_evolution_tests()
    sys.exit(0 if success else 1) 