"""
quantum_evolution_tests.py - Advanced Quantum and Evolution Test Suite
Created: 2024-03-21
Description: This file contains systems that help the digital child learn and adapt over time,
similar to how a human child learns from experience and develops better learning strategies.

Author: Dr. Celaya
Project: Neural Child + Meta-Learning
"""

import torch
import torch.nn as nn
import numpy as np
from copy import deepcopy
import random
from typing import List, Dict, Any, Tuple, Optional
import time
from datetime import datetime
import math

class NegativeBehaviorModule:
    """
    Advanced system for simulating realistic negative behaviors and emotional development.
    Implements interconnected emotional states and their behavioral manifestations.
    """
    def __init__(self, base_rebellion_rate: float = 0.1):
        # Core emotional states with initial values
        self.emotional_state = {
            'frustration': 0.0,  # Builds up from failures and restrictions
            'anger': 0.0,        # Intensifies with sustained frustration
            'anxiety': 0.0,      # Increases with uncertainty and pressure
            'boredom': 0.0,      # Grows with repetitive or unchallenging tasks
            'defiance': 0.0      # Develops from frustration and anger
        }
        
        # Behavioral manifestations
        self.behaviors = {
            'withdrawal': 0.0,     # Tendency to become unresponsive
            'aggression': 0.0,     # Tendency for aggressive responses
            'manipulation': 0.0,    # Tendency to manipulate outcomes
            'attention_seeking': 0.0  # Tendency to act out for attention
        }
        
        # Personality traits that influence behavior
        self.personality = {
            'sensitivity': random.uniform(0.3, 0.7),    # How strongly emotions are felt
            'resilience': random.uniform(0.4, 0.8),     # Recovery from negative states
            'impulsivity': random.uniform(0.2, 0.6),    # Tendency to act on emotions
            'adaptability': random.uniform(0.3, 0.7)    # Ability to adjust to changes
        }
        
        # Track environmental factors
        self.environment = {
            'stress_level': 0.0,
            'social_support': 0.5,
            'previous_successes': [],
            'failure_streak': 0
        }
        
        self.rebellion_rate = base_rebellion_rate
        self.last_update_time = time.time()
        
    def update_emotional_state(self, performance_metrics: Dict[str, float]):
        """Update emotional state based on performance, environment, and personality"""
        current_time = time.time()
        time_delta = current_time - self.last_update_time
        
        # Update environmental stress based on performance
        reward = performance_metrics.get('reward_score', 0.5)
        complexity = performance_metrics.get('complexity_rating', 0.5)
        
        # Track success/failure streaks
        if reward < 0.3:
            self.environment['failure_streak'] += 1
        else:
            self.environment['failure_streak'] = 0
            
        # Update emotional states with interconnected effects
        
        # Frustration increases with failures and high stress
        if reward < 0.3 or self.environment['stress_level'] > 0.7:
            self.emotional_state['frustration'] += 0.2 * self.personality['sensitivity']
            # Anger builds up from sustained frustration
            if self.emotional_state['frustration'] > 0.6:
                self.emotional_state['anger'] += 0.15 * self.personality['impulsivity']
        
        # Anxiety responds to complexity and stress
        if complexity > 0.7 or self.environment['stress_level'] > 0.6:
            self.emotional_state['anxiety'] += 0.1 * (2 - self.personality['resilience'])
        
        # Boredom increases with low complexity or repetitive success
        if complexity < 0.4 or len(self.environment['previous_successes']) > 5:
            self.emotional_state['boredom'] += 0.15 * (1 - self.personality['adaptability'])
        
        # Defiance grows from frustration and anger
        if self.emotional_state['frustration'] > 0.5 and self.emotional_state['anger'] > 0.4:
            self.emotional_state['defiance'] += 0.2 * self.personality['impulsivity']
        
        # Update behavioral manifestations
        self._update_behaviors()
        
        # Natural recovery based on personality
        self._apply_recovery(time_delta)
        
        # Keep track of recent performance
        self.environment['previous_successes'].append(reward > 0.7)
        if len(self.environment['previous_successes']) > 10:
            self.environment['previous_successes'].pop(0)
            
        self.last_update_time = current_time
        
    def _update_behaviors(self):
        """Update behavioral manifestations based on emotional state"""
        # Withdrawal behavior
        self.behaviors['withdrawal'] = (
            self.emotional_state['anxiety'] * 0.4 +
            self.emotional_state['frustration'] * 0.3 +
            self.emotional_state['boredom'] * 0.3
        ) * self.personality['sensitivity']
        
        # Aggressive behavior
        self.behaviors['aggression'] = (
            self.emotional_state['anger'] * 0.5 +
            self.emotional_state['frustration'] * 0.3 +
            self.emotional_state['defiance'] * 0.2
        ) * self.personality['impulsivity']
        
        # Manipulative behavior
        self.behaviors['manipulation'] = (
            self.emotional_state['defiance'] * 0.4 +
            self.emotional_state['anxiety'] * 0.3 +
            self.emotional_state['boredom'] * 0.3
        ) * (1 - self.personality['resilience'])
        
        # Attention-seeking behavior
        self.behaviors['attention_seeking'] = (
            self.emotional_state['boredom'] * 0.4 +
            self.emotional_state['anxiety'] * 0.3 +
            self.emotional_state['frustration'] * 0.3
        ) * (1 - self.personality['adaptability'])
        
    def _apply_recovery(self, time_delta: float):
        """Apply natural emotional recovery based on personality traits"""
        recovery_rate = self.personality['resilience'] * 0.1 * time_delta
        
        # Emotions recover at different rates
        for emotion in self.emotional_state:
            if emotion != 'defiance':  # Defiance recovers more slowly
                self.emotional_state[emotion] *= (1 - recovery_rate)
            else:
                self.emotional_state[emotion] *= (1 - recovery_rate * 0.5)
                
        # Behaviors also naturally decay
        for behavior in self.behaviors:
            self.behaviors[behavior] *= (1 - recovery_rate * 0.7)
            
    def get_behavior_modification(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Generate behavior modifications based on current emotional and behavioral state"""
        behavior_mask = torch.ones_like(input_tensor)
        
        # Only apply modifications if emotions are strong enough
        if random.random() < self.rebellion_rate:
            # Withdrawal behavior - suppress responses
            if self.behaviors['withdrawal'] > 0.5:
                behavior_mask *= (1 - self.behaviors['withdrawal'] * 0.5)
                
            # Aggressive behavior - amplify negative responses
            if self.behaviors['aggression'] > 0.4:
                negative_mask = torch.where(
                    input_tensor < 0,
                    1 + self.behaviors['aggression'],
                    1.0
                )
                behavior_mask *= negative_mask
                
            # Manipulative behavior - strategically modify responses
            if self.behaviors['manipulation'] > 0.4:
                manipulation_mask = torch.where(
                    torch.rand_like(input_tensor) < self.behaviors['manipulation'],
                    -1.0,  # Flip responses
                    1.0
                )
                behavior_mask *= manipulation_mask
                
            # Attention-seeking behavior - exaggerate responses
            if self.behaviors['attention_seeking'] > 0.5:
                attention_mask = 1 + (torch.rand_like(input_tensor) * 
                                    self.behaviors['attention_seeking'])
                behavior_mask *= attention_mask
                
            # Add random noise based on emotional intensity
            emotional_intensity = sum(self.emotional_state.values()) / len(self.emotional_state)
            if emotional_intensity > 0.6:
                noise = torch.randn_like(input_tensor) * emotional_intensity * 0.2
                behavior_mask += noise
                
        return torch.clamp(behavior_mask, -1.0, 2.0)  # Limit the modification range
        
    def get_emotional_summary(self) -> Dict[str, float]:
        """Get a summary of current emotional and behavioral state"""
        return {
            'emotional_state': self.emotional_state.copy(),
            'behaviors': self.behaviors.copy(),
            'personality': self.personality.copy(),
            'environment': {
                'stress_level': self.environment['stress_level'],
                'social_support': self.environment['social_support'],
                'recent_failures': self.environment['failure_streak']
            }
        }

class MetaLearningSystem:
    """
    This system helps the digital child learn how to learn better over time,
    similar to how children develop better study habits and learning strategies as they grow
    """
    def __init__(self, model: nn.Module, learning_rate: float = 0.001):
        # Initialize the system with the child's brain (model) and how quickly it should learn
        self.model = model
        self.meta_optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.performance_history = []  # Keep track of how well the child is doing
        self.meta_learning_rate = learning_rate  # How quickly to adjust learning strategies
        
        # Add shape projection layer for handling emotional inputs
        self.shape_projection = nn.Sequential(
            nn.Linear(4, 128),  # Project from emotional state (4) to intermediate
            nn.ReLU(),
            nn.Linear(128, 768)  # Project to expected input size (768)
        ).to(model.device)
        
        # Add negative behavior module
        self.negative_behavior = NegativeBehaviorModule()
        
        # Add performance tracking
        self.performance_metrics = {
            'success_rate': 0.5,  # Initial baseline
            'learning_efficiency': 0.0,
            'adaptation_speed': 0.0,
            'stability_score': 1.0,
            'last_update': datetime.now()
        }
        
        # Add performance history
        self.metric_history = {
            'success_rates': [],
            'learning_curves': [],
            'adaptation_scores': [],
            'stability_metrics': []
        }
        
    def _validate_and_project_input(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Validate input shape and project if necessary
        """
        try:
            if input_tensor is None:
                print("⚠️ Received None input, using zero tensor")
                return torch.zeros((1, 768), device=self.model.device)
                
            if isinstance(input_tensor, dict):
                # Handle dictionary inputs (like MotherResponse)
                if 'emotional_context' in input_tensor:
                    emotional_values = [
                        input_tensor['emotional_context'].get('joy', 0.5),
                        input_tensor['emotional_context'].get('trust', 0.5),
                        input_tensor['emotional_context'].get('fear', 0.0),
                        input_tensor['emotional_context'].get('surprise', 0.2)
                    ]
                    input_tensor = torch.tensor(emotional_values, device=self.model.device)
                
            if not isinstance(input_tensor, torch.Tensor):
                input_tensor = torch.tensor(input_tensor, device=self.model.device)
                
            if input_tensor.dim() == 1:
                input_tensor = input_tensor.unsqueeze(0)  # Add batch dimension
                
            # Print shape for debugging
            print(f"💭 Input shape check: {input_tensor.size()}")
                
            if input_tensor.size(1) == 4:  # If it's an emotional input
                print("🔄 Projecting emotional input to match network dimensions")
                return self.shape_projection(input_tensor)
            elif input_tensor.size(1) != 768:  # If it's not the expected size
                print(f"⚠️ Unexpected input size: got {input_tensor.size(1)}, expected 768 or 4")
                # Try to reshape or pad if possible
                if input_tensor.numel() <= 768:
                    padded = torch.zeros((1, 768), device=self.model.device)
                    padded[0, :input_tensor.numel()] = input_tensor.view(-1)
                    return padded
                else:
                    return input_tensor[:, :768]  # Truncate to expected size
                
            return input_tensor
            
        except Exception as e:
            print(f"❌ Error in input validation: {str(e)}")
            return torch.zeros((1, 768), device=self.model.device)  # Safe fallback
        
    def get_performance_metrics(self) -> Dict[str, float]:
        """
        Get current performance metrics of the learning system.
        
        Returns:
            Dict[str, float]: Dictionary containing performance metrics:
                - success_rate: Overall success rate of recent interactions
                - learning_efficiency: Rate of improvement over time
                - adaptation_speed: How quickly the system adapts to new scenarios
                - stability_score: Measure of learning stability
        """
        # Calculate learning efficiency from history
        if len(self.performance_history) > 10:
            recent_performance = np.mean(self.performance_history[-10:])
            previous_performance = np.mean(self.performance_history[-20:-10])
            self.performance_metrics['learning_efficiency'] = (recent_performance - previous_performance)
            
        # Calculate adaptation speed
        time_since_update = (datetime.now() - self.performance_metrics['last_update']).total_seconds()
        adaptation_factor = math.exp(-time_since_update / 3600)  # Decay over an hour
        self.performance_metrics['adaptation_speed'] = adaptation_factor
        
        # Update stability score based on performance variance
        if len(self.performance_history) > 5:
            stability = 1.0 - np.std(self.performance_history[-5:])
            self.performance_metrics['stability_score'] = max(0.1, stability)
            
        # Update success rate based on recent performance
        if self.performance_history:
            self.performance_metrics['success_rate'] = np.mean(self.performance_history[-5:])
            
        # Update timestamp
        self.performance_metrics['last_update'] = datetime.now()
        
        # Record metrics in history
        self.metric_history['success_rates'].append(self.performance_metrics['success_rate'])
        self.metric_history['learning_curves'].append(self.performance_metrics['learning_efficiency'])
        self.metric_history['adaptation_scores'].append(self.performance_metrics['adaptation_speed'])
        self.metric_history['stability_metrics'].append(self.performance_metrics['stability_score'])
        
        return self.performance_metrics
        
    def meta_update(self, performance_metrics: Dict[str, float]) -> float:
        """
        Update how the child learns based on how well they're doing,
        like adjusting teaching methods based on a child's test scores
        """
        try:
            # Update negative behavior emotional state
            self.negative_behavior.update_emotional_state(performance_metrics)
            
            # Calculate how to improve based on the reward received
            meta_loss = -torch.tensor(performance_metrics.get('reward_score', 0.0), 
                                    requires_grad=True, 
                                    device=self.model.device)
            
            # Update the learning strategy
            self.meta_optimizer.zero_grad()
            meta_loss.backward()
            self.meta_optimizer.step()
            
            # Adjust learning speed based on recent performance
            # Like giving a child more challenging material if they're doing well
            if len(self.performance_history) > 10:
                if np.mean(self.performance_history[-10:]) > np.mean(self.performance_history[-20:-10]):
                    self.meta_learning_rate *= 1.1  # Speed up learning if doing well
                else:
                    self.meta_learning_rate *= 0.9  # Slow down if struggling
            
            self.performance_history.append(performance_metrics.get('reward_score', 0.0))
            return meta_loss.item()
            
        except Exception as e:
            print(f"❌ Error in meta_update: {str(e)}")
            return 0.0  # Return safe default

    def forward(self, input_data: Any) -> torch.Tensor:
        """
        Process input through the meta-learning system with behavior modifications
        """
        # Validate and project input
        processed_input = self._validate_and_project_input(input_data)
        
        # Get behavior modifications
        behavior_mask = self.negative_behavior.get_behavior_modification(processed_input)
        
        # Apply behavior modifications to input
        modified_input = processed_input * behavior_mask
        
        # Process through model
        return self.model(modified_input)

    def _process_quantum_emotions(self, stimulus_vector):
        """Process emotions using quantum-inspired algorithms"""
        # Ensure stimulus vector is properly shaped
        if stimulus_vector.dim() == 0:
            stimulus_vector = stimulus_vector.unsqueeze(0)  # Make it 1D
        if stimulus_vector.dim() == 1:
            stimulus_vector = stimulus_vector.unsqueeze(0)  # Add batch dimension
            
        # Reshape entanglement matrix if needed
        if self.quantum_emotional_state['entanglement_matrix'].dim() == 2:
            entanglement_matrix = self.quantum_emotional_state['entanglement_matrix'].unsqueeze(0)
        else:
            entanglement_matrix = self.quantum_emotional_state['entanglement_matrix']
            
        # Update superposition state with proper broadcasting
        self.quantum_emotional_state['superposition'] = torch.nn.functional.softmax(
            torch.matmul(stimulus_vector, entanglement_matrix).squeeze(), 
            dim=-1
        )
        
        # Apply quantum noise (decoherence)
        noise = torch.randn_like(self.quantum_emotional_state['superposition']) * (1 - self.quantum_emotional_state['coherence_factor'])
        self.quantum_emotional_state['superposition'] += noise
        
        # Update coherence
        self.quantum_emotional_state['coherence_factor'] *= 0.99  # Gradual decoherence
        
        # Check for emotional collapse
        if torch.max(self.quantum_emotional_state['superposition']) > self.quantum_emotional_state['collapse_threshold']:
            # Collapse to classical emotional state
            classical_state = torch.zeros_like(self.quantum_emotional_state['superposition'])
            max_idx = torch.argmax(self.quantum_emotional_state['superposition'])
            classical_state[max_idx] = 1.0
            self.quantum_emotional_state['superposition'] = classical_state
            
        # Update metrics
        self.quantum_metrics['coherence_history'].append(self.quantum_emotional_state['coherence_factor'])
        self.quantum_metrics['entanglement_strength'].append(
            torch.trace(self.quantum_emotional_state['entanglement_matrix']).item()
        )
        self.quantum_metrics['superposition_stability'].append(
            torch.std(self.quantum_emotional_state['superposition']).item()
        )
        
        return self.quantum_emotional_state['superposition']

    def learn(self, mother_feedback):
        """Process learning feedback and update model"""
        try:
            # Ensure input tensor exists and is properly shaped
            input_tensor = mother_feedback.get('input')
            if input_tensor is None:
                input_tensor = torch.zeros((1, self.model.input_size), device=self.model.device)
                
            # Record experience in memory
            self.memory.record_experience(
                input_tensor,
                mother_feedback.get('internal_state'),
                mother_feedback.get('reward', 0.0),
                time.time(),
                self.emotional_state  
            )
            
            # Process quantum emotions with proper tensor handling
            reward_tensor = torch.tensor(mother_feedback.get('reward', 0.0), 
                                       device=self.model.device)
            quantum_emotional_state = self._process_quantum_emotions(reward_tensor)
            
            # Evolve neural architecture
            evolution_metrics = self._evolve_neural_architecture()
            
            # Debug print
            print(f"Input tensor shape before projection: {input_tensor.size()}")
            print(f"Quantum emotional state: {quantum_emotional_state}")
            print(f"Evolution metrics: {evolution_metrics}")
            
            # Handle emotional input projection
            if isinstance(input_tensor, torch.Tensor) and input_tensor.size(1) == 4:
                input_tensor = self.shape_projection(input_tensor)
                print(f"Input tensor shape after projection: {input_tensor.size()}")
            
            # Verify input shape matches expected
            if isinstance(input_tensor, torch.Tensor) and input_tensor.size(1) != self.model.input_size:
                print(f"Shape mismatch: got {input_tensor.size()}, expected {self.model.input_size}")
                input_tensor = torch.zeros((1, self.model.input_size), device=self.model.device)
            
            # Enhanced training step with quantum and evolution integration
            loss = self.trainer.training_step(
                input_tensor, 
                quantum_state=quantum_emotional_state,
                architecture_complexity=evolution_metrics['complexity']
            )
            
            # Meta-learning update with enhanced metrics
            meta_loss = self.meta_update({
                'reward_score': mother_feedback.get('reward', 0.0),
                'success_metric': mother_feedback.get('success_metric', 0.5),
                'complexity_rating': mother_feedback.get('complexity_rating', 0.3),
                'quantum_coherence': self.quantum_emotional_state['coherence_factor'],
                'architecture_adaptation': evolution_metrics['adaptation_score']
            })
            
            return loss
            
        except Exception as e:
            print(f"Detailed error in learn step: {str(e)}")
            # Return safe default loss value
            return torch.tensor(0.0, device=self.model.device)

class NeuralArchitectureSearch:
    """
    This system experiments with different brain structures to find the best one,
    like trying different teaching methods to find what works best for a particular child
    """
    def __init__(self, 
                 input_size: int,
                 output_size: int,
                 max_layers: int = 5,
                 device: str = 'cuda'):
        self.input_size = input_size
        self.output_size = output_size
        self.max_layers = max_layers
        self.device = device
        self.best_architecture = None  # Remember the best structure found
        self.best_performance = float('-inf')
        
    def generate_architecture(self) -> List[Dict[str, Any]]:
        """
        Create a new brain structure to try out,
        like designing a new learning approach
        """
        num_layers = random.randint(2, self.max_layers)
        architecture = []
        current_size = self.input_size
        
        # Build each layer of the brain
        for i in range(num_layers - 1):
            layer_size = random.choice([32, 64, 128, 256, 512])
            architecture.append({
                'type': 'linear',
                'in_features': current_size,
                'out_features': layer_size,
                'activation': random.choice(['relu', 'tanh', 'gelu'])
            })
            current_size = layer_size
        
        # Add the final output layer
        architecture.append({
            'type': 'linear',
            'in_features': current_size,
            'out_features': self.output_size,
            'activation': 'tanh'  # Keep outputs in a reasonable range
        })
        
        return architecture
    
    def evaluate_architecture(self, 
                            architecture: List[Dict[str, Any]], 
                            performance_metrics: Dict[str, float]) -> float:
        """
        Test how well a particular brain structure works,
        like evaluating if a teaching method is effective
        """
        # Calculate overall performance score
        performance_score = (
            performance_metrics['reward_score'] * 0.4 +      # How well the child did
            performance_metrics['success_metric'] * 0.3 +    # How successful they were
            performance_metrics['complexity_rating'] * 0.3    # How complex their thinking was
        )
        
        # Keep track of the best structure found
        if performance_score > self.best_performance:
            self.best_architecture = architecture
            self.best_performance = performance_score
            
        return performance_score

class GeneticOptimizer:
    """
    This system evolves better versions of the child's brain over time,
    similar to how evolution selects for beneficial traits
    """
    def __init__(self, 
                 population_size: int = 10,
                 mutation_rate: float = 0.1,
                 crossover_rate: float = 0.7):
        self.population_size = population_size  # How many versions to try
        self.mutation_rate = mutation_rate      # How often to make random changes
        self.crossover_rate = crossover_rate    # How often to combine good features
        self.population = []
        self.fitness_scores = []
        
    def initialize_population(self, model: nn.Module) -> List[nn.Module]:
        """
        Create initial versions of the brain to experiment with,
        like starting with different teaching approaches
        """
        # Create copies of the original brain
        self.population = [deepcopy(model) for _ in range(self.population_size)]
        
        # Add some random variations to each copy
        for individual in self.population[1:]:  # Keep one unchanged
            self._mutate(individual)
            
        return self.population
    
    def _mutate(self, model: nn.Module) -> None:
        """
        Make random changes to a brain version,
        like trying small variations in teaching methods
        """
        with torch.no_grad():
            for param in model.parameters():
                if random.random() < self.mutation_rate:
                    mutation = torch.randn_like(param) * 0.1
                    param.add_(mutation)
    
    def crossover(self, parent1: nn.Module, parent2: nn.Module) -> nn.Module:
        """
        Combine features from two successful brain versions,
        like combining effective aspects of different teaching methods
        """
        child = deepcopy(parent1)
        
        with torch.no_grad():
            for p1, p2, c in zip(parent1.parameters(), 
                               parent2.parameters(),
                               child.parameters()):
                mask = torch.rand_like(p1) < self.crossover_rate
                c.data = torch.where(mask, p1.data, p2.data)
                
        return child
    
    def evolve(self, fitness_scores: List[float]) -> List[nn.Module]:
        """
        Create a new generation of improved brain versions,
        like refining teaching methods based on what worked best
        """
        self.fitness_scores = fitness_scores
        
        # Sort by how well each version performed
        sorted_population = [x for _, x in sorted(
            zip(fitness_scores, self.population),
            key=lambda pair: pair[0],
            reverse=True
        )]
        
        # Keep the best performers
        new_population = sorted_population[:2]
        
        # Create new versions by combining successful ones
        while len(new_population) < self.population_size:
            parent1, parent2 = random.choices(
                sorted_population[:5],  # Select from top 5
                k=2
            )
            child = self.crossover(parent1, parent2)
            self._mutate(child)
            new_population.append(child)
            
        self.population = new_population
        return self.population

def create_model_from_architecture(architecture: List[Dict[str, Any]], 
                                 device: str = 'cuda') -> nn.Module:
    """
    Build a new brain from a blueprint,
    like implementing a new teaching strategy
    """
    layers = []
    
    for layer_spec in architecture:
        layers.append(nn.Linear(
            layer_spec['in_features'],
            layer_spec['out_features']
        ))
        
        if layer_spec['activation'] == 'relu':
            layers.append(nn.ReLU())
        elif layer_spec['activation'] == 'tanh':
            layers.append(nn.Tanh())
        elif layer_spec['activation'] == 'gelu':
            layers.append(nn.GELU())
            
    return nn.Sequential(*layers).to(device) 