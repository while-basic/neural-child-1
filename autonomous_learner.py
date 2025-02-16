import torch
import random
import numpy as np
from config import config

class AutonomousLearner:
    def __init__(self, child_model):
        self.model = child_model
        self.exploration_rate = 0.3
        self.curiosity_threshold = 0.7
        self.learning_topics = [
            "emotions", "objects", "actions", "concepts",
            "relationships", "communication", "problem-solving"
        ]
        self.learning_history = []

    def generate_self_prompt(self) -> str:
        """Generate learning prompts based on curiosity and current development stage"""
        stage = self.model.curriculum.current_stage
        complexity = min(1.0, stage.value / 17.0)  # Normalize stage value
        
        topic = random.choice(self.learning_topics)
        return f"I want to explore and learn about {topic} at complexity level {complexity:.2f}"

    def evaluate_learning(self, response: torch.Tensor) -> float:
        """Evaluate learning progress based on response quality"""
        try:
            # Convert tensor response to evaluation metrics
            response_vector = response.detach().cpu().numpy()
            metrics = {
                'coherence': float(np.mean(response_vector)),  # Average activation
                'complexity': float(np.std(response_vector)),  # Response variation
                'novelty': float(np.max(response_vector)),     # Peak activation
                'emotional_engagement': float(np.abs(np.mean(response_vector)))  # Emotional intensity
            }
            return sum(metrics.values()) / len(metrics)
        except Exception as e:
            print(f"Error in evaluate_learning: {e}")
            return 0.5  # Return default score on error

    def adjust_learning_strategy(self, performance: float):
        """Adjust learning parameters based on performance"""
        if performance < 0.3:
            self.exploration_rate = max(0.1, self.exploration_rate * 0.9)
        elif performance > 0.7:
            self.exploration_rate = min(0.5, self.exploration_rate * 1.1)

    def reset_learning_parameters(self):
        """Reset learning parameters to default values to recover from stuck states"""
        self.exploration_rate = 0.3
        self.curiosity_threshold = 0.7
        # Clear recent learning history to avoid being stuck in a bad pattern
        if len(self.learning_history) > 0:
            self.learning_history = self.learning_history[:-10]  # Keep all but last 10 entries
        print("Reset learning parameters to default values")

    def learn_independently(self) -> dict:
        """Execute one cycle of autonomous learning"""
        prompt = self.generate_self_prompt()
        
        # Generate self-directed response
        with torch.no_grad():
            perception = self.model.perceive({'text': prompt})
            response = self.model.respond(perception)
            
            # Evaluate learning
            performance = self.evaluate_learning(response)
            
            # Update learning strategy
            self.adjust_learning_strategy(performance)
            
            # Record learning experience
            self.learning_history.append({
                'prompt': prompt,
                'performance': performance,
                'timestamp': self.model.age()
            })
        
        return {
            'prompt': prompt,
            'performance': performance,
            'exploration_rate': self.exploration_rate
        }
