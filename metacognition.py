import torch
import torch.nn as nn
from typing import Dict, Any, Optional
import numpy as np
from collections import deque
from datetime import datetime

class MetacognitionSystem(nn.Module):
    """System for tracking and developing metacognitive abilities."""
    
    def __init__(self, hidden_dim: int = 32):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Core metacognitive networks
        self.self_awareness_network = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
        self.learning_efficiency_network = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
        # Attention and memory systems
        self.attention_network = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
        # State tracking
        self.self_awareness_score = 0.0
        self.learning_efficiency = 0.0
        self.attention_focus = 0.0
        self.experience_buffer = deque(maxlen=100)
        self.reflection_history = []
        
        # Initialize metrics history
        self.metrics_history = {
            'self_awareness': deque(maxlen=100),
            'learning_efficiency': deque(maxlen=100),
            'attention_focus': deque(maxlen=100)
        }
    
    def update(self, 
              current_state: torch.Tensor,
              feedback: str,
              learning_outcome: Optional[float] = None) -> None:
        """Update metacognitive understanding based on experience.
        
        Args:
            current_state: Current emotional and cognitive state
            feedback: Feedback received from interaction
            learning_outcome: Optional measure of learning success
        """
        # Encode current experience
        state_encoding = self._encode_state(current_state)
        feedback_encoding = self._encode_feedback(feedback)
        
        # Update self-awareness based on state-feedback alignment
        self_awareness_input = torch.cat([state_encoding, feedback_encoding], dim=-1)
        new_self_awareness = self.self_awareness_network(self_awareness_input).item()
        
        # Update learning efficiency if outcome provided
        if learning_outcome is not None:
            learning_input = torch.cat([
                state_encoding,
                torch.tensor([learning_outcome]).float().unsqueeze(0)
            ], dim=-1)
            new_learning_efficiency = self.learning_efficiency_network(learning_input).item()
            self.learning_efficiency = 0.9 * self.learning_efficiency + 0.1 * new_learning_efficiency
        
        # Update attention focus
        attention_score = self.attention_network(state_encoding).item()
        self.attention_focus = 0.9 * self.attention_focus + 0.1 * attention_score
        
        # Progressive self-awareness development
        if len(self.metrics_history['self_awareness']) > 0:
            # Consider history for smoother progression
            historical_awareness = np.mean(list(self.metrics_history['self_awareness']))
            # Allow small increments but prevent large jumps
            max_increase = 0.1
            awareness_delta = new_self_awareness - historical_awareness
            capped_delta = np.clip(awareness_delta, -max_increase, max_increase)
            self.self_awareness_score = historical_awareness + capped_delta
        else:
            self.self_awareness_score = new_self_awareness
        
        # Store experience
        self.experience_buffer.append({
            'state': current_state.detach().cpu().numpy(),
            'feedback': feedback,
            'learning_outcome': learning_outcome,
            'self_awareness': self.self_awareness_score,
            'learning_efficiency': self.learning_efficiency,
            'attention_focus': self.attention_focus
        })
        
        # Update metrics history
        self.metrics_history['self_awareness'].append(self.self_awareness_score)
        self.metrics_history['learning_efficiency'].append(self.learning_efficiency)
        self.metrics_history['attention_focus'].append(self.attention_focus)
        
        # Periodic reflection
        if len(self.experience_buffer) % 10 == 0:
            self._reflect_on_experiences()
    
    def _encode_state(self, state: torch.Tensor) -> torch.Tensor:
        """Encode current state for metacognitive processing."""
        # Ensure state has correct dimensionality
        if state.dim() == 1:
            state = state.unsqueeze(0)
        
        # Project to hidden dimension if necessary
        if state.size(-1) != self.hidden_dim:
            projection = nn.Linear(state.size(-1), self.hidden_dim).to(state.device)
            state = projection(state)
        
        return state
    
    def _encode_feedback(self, feedback: str) -> torch.Tensor:
        """Encode feedback for metacognitive processing."""
        # Simple bag-of-words encoding for now
        # TODO: NEURAL-126 - Implement more sophisticated feedback encoding
        words = set(feedback.lower().split())
        encoding = torch.zeros(self.hidden_dim)
        for i, word in enumerate(words):
            if i < self.hidden_dim:
                encoding[i] = 1.0
        return encoding.unsqueeze(0)
    
    def _reflect_on_experiences(self) -> None:
        """Periodically reflect on recent experiences to update understanding."""
        if len(self.experience_buffer) < 10:
            return
            
        recent_experiences = list(self.experience_buffer)[-10:]
        
        # Analyze patterns in recent experiences
        state_consistency = np.mean([
            np.std(exp['state']) for exp in recent_experiences
        ])
        learning_progress = np.mean([
            exp['learning_outcome'] for exp in recent_experiences 
            if exp['learning_outcome'] is not None
        ])
        
        reflection = {
            'timestamp': datetime.now(),
            'state_consistency': float(state_consistency),
            'learning_progress': float(learning_progress),
            'self_awareness_trend': float(np.mean([
                exp['self_awareness'] for exp in recent_experiences
            ])),
            'attention_stability': float(np.mean([
                exp['attention_focus'] for exp in recent_experiences
            ]))
        }
        
        self.reflection_history.append(reflection)
    
    def get_metrics(self) -> Dict[str, float]:
        """Return current metacognitive metrics with historical context."""
        recent_metrics = {
            'self_awareness': self.self_awareness_score,
            'learning_efficiency': self.learning_efficiency,
            'attention_focus': self.attention_focus
        }
        
        # Add historical trends if available
        if self.reflection_history:
            latest_reflection = self.reflection_history[-1]
            recent_metrics.update({
                'state_consistency': latest_reflection['state_consistency'],
                'learning_progress': latest_reflection['learning_progress'],
                'attention_stability': latest_reflection['attention_stability']
            })
        
        return recent_metrics
    
    def regulate_learning(self, 
                         current_performance: float,
                         target_performance: float) -> Dict[str, float]:
        """Adjust learning parameters based on metacognitive assessment."""
        performance_gap = target_performance - current_performance
        
        # Calculate adjustment factors
        learning_rate_factor = np.clip(1.0 + performance_gap, 0.5, 2.0)
        exploration_factor = np.clip(1.0 - abs(performance_gap), 0.1, 1.0)
        
        # Consider metacognitive state for adjustments
        metacog_state = self.get_metrics()
        
        return {
            'learning_rate_adjustment': learning_rate_factor * metacog_state['learning_efficiency'],
            'exploration_rate': exploration_factor * metacog_state['attention_focus'],
            'attention_threshold': metacog_state['attention_focus'],
            'self_awareness_level': metacog_state['self_awareness']
        }
