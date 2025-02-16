import torch
from typing import List, Dict
from config import config

class AutonomousLearner:
    def __init__(self, child_model):
        self.model = child_model
        self.exploration_rate = 0.3
        self.curiosity_threshold = 0.7
        self.learning_history = []

    def generate_self_prompt(self) -> str:
        """Generate learning prompts based on curiosity"""
        topics = [
            "feelings", "objects", "people", "actions",
            "colors", "numbers", "words", "concepts"
        ]
        
        # Select topic based on current development stage
        stage = self.model.curriculum.current_stage
        complexity = min(1.0, stage.value / 17.0)  # Normalize stage value
        
        return f"I want to learn about {torch.choice(topics)} at complexity level {complexity:.2f}"

    def evaluate_learning(self, response: Dict) -> float:
        """Self-evaluate learning progress"""
        confidence = response.get('confidence', 0.0)
        coherence = response.get('coherence', 0.0)
        novelty = response.get('novelty', 0.0)
        
        return (confidence + coherence + novelty) / 3.0

    def adjust_learning_path(self, performance: float):
        """Adjust learning parameters based on performance"""
        if performance < 0.3:
            self.exploration_rate *= 0.9  # Reduce exploration
        elif performance > 0.7:
            self.exploration_rate = min(0.5, self.exploration_rate * 1.1)  # Increase exploration

    def learn_independently(self):
        """Execute autonomous learning cycle"""
        prompt = self.generate_self_prompt()
        
        with torch.no_grad():
            response = self.model.respond(self.model.perceive({'text': prompt}))
            performance = self.evaluate_learning(response)
            self.adjust_learning_path(performance)
            
        return {
            'prompt': prompt,
            'performance': performance,
            'exploration_rate': self.exploration_rate
        }
