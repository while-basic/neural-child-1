import torch
import torch.nn as nn
from typing import Dict, Any
import numpy as np
from config import config

class AutonomousLearner:
    def __init__(self, child_model):
        self.child = child_model
        self.learning_parameters = {
            'learning_rate': config.learning_rate,
            'exploration_rate': 0.3,
            'curriculum_difficulty': 0.1
        }
        self.performance_history = []
        self.exploration_decay = 0.995
        self.min_exploration = 0.05
        
    def get_learning_rate(self) -> float:
        """Get the current learning rate.
        
        Returns:
            The current learning rate value.
        """
        return self.learning_parameters['learning_rate']

    def learn_independently(self) -> Dict[str, float]:
        """Execute one cycle of autonomous learning"""
        try:
            # Generate self-directed learning task
            task = self._generate_task()
            
            # Attempt the task
            with torch.no_grad():
                response = self.child.brain(task['input'])
            
            # Self-evaluate performance
            performance = self._evaluate_performance(response, task['target'])
            
            # Update learning parameters based on performance
            self._adapt_parameters(performance)
            
            # Record performance
            self.performance_history.append(performance)
            
            # Decay exploration rate
            self.learning_parameters['exploration_rate'] = max(
                self.min_exploration,
                self.learning_parameters['exploration_rate'] * self.exploration_decay
            )
            
            return {
                'performance': performance,
                'learning_rate': self.learning_parameters['learning_rate'],
                'exploration_rate': self.learning_parameters['exploration_rate'],
                'curriculum_difficulty': self.learning_parameters['curriculum_difficulty']
            }
            
        except Exception as e:
            print(f"Error in autonomous learning: {str(e)}")
            return {
                'performance': 0.0,
                'learning_rate': self.learning_parameters['learning_rate'],
                'exploration_rate': self.learning_parameters['exploration_rate'],
                'curriculum_difficulty': self.learning_parameters['curriculum_difficulty']
            }
    
    def _generate_task(self) -> Dict[str, torch.Tensor]:
        """Generate a learning task based on current capabilities"""
        # Get current stage characteristics
        stage = self.child.curriculum.current_stage
        stage_char = self.child.curriculum.get_stage_characteristics()
        
        # Generate task difficulty based on current performance
        if self.performance_history:
            avg_performance = np.mean(self.performance_history[-10:])
            difficulty = self.learning_parameters['curriculum_difficulty']
            
            # Adjust difficulty based on performance
            if avg_performance > 0.8:
                difficulty = min(1.0, difficulty + 0.1)
            elif avg_performance < 0.4:
                difficulty = max(0.1, difficulty - 0.1)
                
            self.learning_parameters['curriculum_difficulty'] = difficulty
        
        # Generate input tensor
        input_dim = config.embedding_dim
        noise_scale = self.learning_parameters['exploration_rate']
        
        base_input = torch.randn(1, input_dim, device=self.child.device)
        noise = torch.randn_like(base_input) * noise_scale
        task_input = base_input + noise
        
        # Generate target based on stage requirements
        target = self._generate_target(stage_char)
        
        return {
            'input': task_input,
            'target': target,
            'difficulty': self.learning_parameters['curriculum_difficulty']
        }
    
    def _generate_target(self, stage_char) -> torch.Tensor:
        """Generate target tensor based on stage characteristics"""
        target_dim = config.embedding_dim
        complexity = np.interp(
            stage_char.complexity_range[0],
            [0, 1],
            [0.1, 0.9]
        )
        
        # Generate structured target
        target = torch.zeros(1, target_dim, device=self.child.device)
        num_active = int(target_dim * complexity)
        active_indices = torch.randperm(target_dim)[:num_active]
        target[0, active_indices] = torch.randn(num_active)
        
        return target
    
    def _evaluate_performance(self, 
                            response: torch.Tensor,
                            target: torch.Tensor) -> float:
        """Evaluate the performance of the response against the target"""
        with torch.no_grad():
            # Calculate cosine similarity
            similarity = torch.nn.functional.cosine_similarity(
                response.flatten(),
                target.flatten(),
                dim=0
            )
            
            # Scale to [0, 1] range
            performance = (similarity + 1) / 2
            
            return performance.item()
    
    def _adapt_parameters(self, performance: float):
        """Adapt learning parameters based on performance"""
        # Adjust learning rate
        if performance < 0.3:
            self.learning_parameters['learning_rate'] *= 0.9
        elif performance > 0.8:
            self.learning_parameters['learning_rate'] *= 1.1
            
        # Clip learning rate
        self.learning_parameters['learning_rate'] = np.clip(
            self.learning_parameters['learning_rate'],
            1e-5,
            1e-2
        )
    
    def reset_learning_parameters(self):
        """Reset learning parameters to default values"""
        self.learning_parameters = {
            'learning_rate': config.learning_rate,
            'exploration_rate': 0.3,
            'curriculum_difficulty': 0.1
        }
        self.performance_history = []

    def process_feedback(self, feedback: str, current_stage: Any, learning_objectives: Dict[str, Any]) -> None:
        """Process feedback from mother and update learning parameters.
        
        Args:
            feedback: The feedback text from the mother
            current_stage: The current developmental stage
            learning_objectives: Dictionary of current learning objectives
        """
        try:
            # Extract learning signals from feedback
            performance = self._evaluate_feedback(feedback)
            
            # Update learning parameters based on performance
            self._adapt_parameters(performance)
            
            # Record performance
            self.performance_history.append(performance)
            
            # Adjust curriculum difficulty based on performance
            if len(self.performance_history) >= 5:
                recent_performance = np.mean(self.performance_history[-5:])
                if recent_performance > 0.8:
                    self.learning_parameters['curriculum_difficulty'] = min(
                        1.0,
                        self.learning_parameters['curriculum_difficulty'] + 0.1
                    )
                elif recent_performance < 0.4:
                    self.learning_parameters['curriculum_difficulty'] = max(
                        0.1,
                        self.learning_parameters['curriculum_difficulty'] - 0.1
                    )
            
            # Decay exploration rate
            self.learning_parameters['exploration_rate'] = max(
                self.min_exploration,
                self.learning_parameters['exploration_rate'] * self.exploration_decay
            )
            
        except Exception as e:
            print(f"Error processing feedback: {str(e)}")
    
    def _evaluate_feedback(self, feedback: str) -> float:
        """Evaluate feedback to determine learning performance.
        
        Args:
            feedback: The feedback text to evaluate
            
        Returns:
            Float between 0 and 1 indicating performance
        """
        # Simple sentiment-based evaluation for now
        positive_words = {'good', 'great', 'excellent', 'well', 'correct', 'right', 'yes'}
        negative_words = {'bad', 'wrong', 'incorrect', 'no', 'not', "don't", 'stop'}
        
        words = set(feedback.lower().split())
        positive_count = len(words.intersection(positive_words))
        negative_count = len(words.intersection(negative_words))
        
        total_count = positive_count + negative_count
        if total_count == 0:
            return 0.5  # Neutral feedback
            
        return positive_count / total_count
