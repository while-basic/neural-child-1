import torch
import torch.nn as nn
from typing import Dict, Any, Optional
import numpy as np

class MetacognitionSystem(nn.Module):
    def __init__(self, hidden_dim: int = 32):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Self-awareness components
        self.state_encoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2)
        )
        
        self.confidence_predictor = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid()
        )
        
        self.error_predictor = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid()
        )
        
        # Learning regulation
        self.attention_controller = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Metrics tracking
        self.self_awareness_score = 0.0
        self.learning_efficiency = 0.0
        self.attention_focus = 0.0
        
    def forward(self, 
                state: torch.Tensor,
                target: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Process current state through metacognitive systems"""
        # Encode current state
        encoded_state = self.state_encoder(state)
        
        # Predict confidence and potential error
        confidence = self.confidence_predictor(encoded_state)
        error_prediction = self.error_predictor(encoded_state)
        
        # Calculate attention focus
        attention = self.attention_controller(state)
        
        # Update metrics if target is provided
        if target is not None:
            self._update_metrics(confidence, error_prediction, target)
        
        return {
            'confidence': confidence,
            'error_prediction': error_prediction,
            'attention': attention
        }
    
    def _update_metrics(self,
                       confidence: torch.Tensor,
                       error_prediction: torch.Tensor,
                       target: torch.Tensor):
        """Update internal metrics based on predictions and actual outcomes"""
        # Calculate actual error
        actual_error = torch.mean((target - error_prediction) ** 2)
        
        # Update self-awareness based on prediction accuracy
        prediction_accuracy = 1 - torch.abs(actual_error - error_prediction).item()
        self.self_awareness_score = 0.9 * self.self_awareness_score + 0.1 * prediction_accuracy
        
        # Update learning efficiency
        confidence_value = confidence.item()
        self.learning_efficiency = 0.9 * self.learning_efficiency + 0.1 * (
            confidence_value if prediction_accuracy > 0.5 else 1 - confidence_value
        )
        
        # Update attention focus
        self.attention_focus = self.attention_focus * 0.9 + 0.1 * (
            1.0 if prediction_accuracy > 0.7 else 0.5
        )
    
    def get_metrics(self) -> Dict[str, float]:
        """Return current metacognitive metrics"""
        return {
            'self_awareness': self.self_awareness_score,
            'learning_efficiency': self.learning_efficiency,
            'attention_focus': self.attention_focus
        }
    
    def regulate_learning(self, 
                         current_performance: float,
                         target_performance: float) -> Dict[str, float]:
        """Adjust learning parameters based on metacognitive assessment"""
        performance_gap = target_performance - current_performance
        
        # Calculate adjustment factors
        learning_rate_factor = np.clip(1.0 + performance_gap, 0.5, 2.0)
        exploration_factor = np.clip(1.0 - abs(performance_gap), 0.1, 1.0)
        
        return {
            'learning_rate_adjustment': learning_rate_factor,
            'exploration_rate': exploration_factor,
            'attention_threshold': self.attention_focus
        }
