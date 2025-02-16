import torch
import random
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

    def evaluate_learning(self, response: dict) -> float:
        """Evaluate learning progress based on response quality"""
        metrics = {
            'coherence': response.get('coherence', 0.5),
            'complexity': response.get('complexity_rating', 0.5),
            'novelty': response.get('novelty', 0.5),
            'emotional_engagement': response.get('emotional_intensity', 0.5)
        }
        return sum(metrics.values()) / len(metrics)

    def adjust_learning_strategy(self, performance: float):
        """Adjust learning parameters based on performance"""
        if performance < 0.3:
            self.exploration_rate = max(0.1, self.exploration_rate * 0.9)
        elif performance > 0.7:
            self.exploration_rate = min(0.5, self.exploration_rate * 1.1)

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
