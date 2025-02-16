# emotional_regulation.py

import torch
import torch.nn as nn
from collections import deque
from typing import Dict, Any, Optional
import numpy as np

class EmotionalState:
    def __init__(self, device='cuda'):
        self.device = device
        # Use only 4 primary emotions: joy, trust, fear, surprise.
        self.primary_emotions = nn.ParameterDict({
            'joy': nn.Parameter(torch.tensor(0.0, device=device)),
            'trust': nn.Parameter(torch.tensor(0.0, device=device)),
            'fear': nn.Parameter(torch.tensor(0.0, device=device)),
            'surprise': nn.Parameter(torch.tensor(0.0, device=device))
        })
        
        self.complex_emotions = {
            'love': {'joy': 0.6, 'trust': 0.4},
            'guilt': {'fear': 0.5, 'surprise': 0.5},
            'pride': {'joy': 0.7, 'fear': 0.3},
            'shame': {'trust': 0.6, 'surprise': 0.4},
            'anxiety': {'fear': 0.7, 'surprise': 0.3},
            'contentment': {'joy': 0.5, 'trust': 0.5},
            'rejection': {'fear': 0.4, 'surprise': 0.6},
            'excitement': {'joy': 0.5, 'surprise': 0.5}
        }
        
        self.stability_window = deque(maxlen=100)
        self.baseline = {k: 0.5 for k in self.primary_emotions.keys()}
        
    def update(self, emotional_input: dict, learning_rate: float = 0.1) -> None:
        for emotion, value in emotional_input.items():
            if emotion in self.primary_emotions:
                current = self.primary_emotions[emotion].item()
                delta = (value - current) * learning_rate
                noise = torch.randn(1, device=self.device).item() * 0.05
                new_value = torch.clamp(current + delta + noise, 0.0, 1.0)
                self.primary_emotions[emotion].data = torch.tensor(new_value, device=self.device)
                
        total_change = sum(abs(self.primary_emotions[k].item() - self.baseline[k]) for k in self.primary_emotions.keys())
        self.stability_window.append(total_change)
        
    def get_complex_emotion(self, emotion_name: str) -> float:
        if (emotion_name not in self.complex_emotions):
            return 0.0
        composition = self.complex_emotions[emotion_name]
        intensity = sum(
            self.primary_emotions[primary].item() * weight 
            for primary, weight in composition.items()
        )
        return float(torch.clamp(torch.tensor(intensity), 0.0, 1.0))
    
    def get_dominant_emotion(self):
        primary_intensities = {name: self.primary_emotions[name].item() for name in self.primary_emotions.keys()}
        complex_intensities = {name: self.get_complex_emotion(name) for name in self.complex_emotions.keys()}
        all_emotions = {**primary_intensities, **complex_intensities}
        dominant = max(all_emotions.items(), key=lambda x: x[1])
        return dominant
    
    def get_emotional_stability(self) -> float:
        if not self.stability_window:
            return 1.0
        recent_volatility = sum(self.stability_window) / len(self.stability_window)
        stability = 1.0 - min(recent_volatility, 1.0)
        return float(stability)
    
    def to_tensor(self) -> torch.Tensor:
        return torch.tensor([self.primary_emotions[emotion].item() for emotion in sorted(self.primary_emotions.keys())], device=self.device)
    
    def from_tensor(self, tensor: torch.Tensor) -> None:
        sorted_emotions = sorted(self.primary_emotions.keys())
        for i, emotion in enumerate(sorted_emotions):
            self.primary_emotions[emotion].data = tensor[i]

class EmotionalRegulation(nn.Module):
    def __init__(self, 
                 emotion_dim: int = 4,
                 hidden_dim: int = 32,
                 device: Optional[torch.device] = None):
        super().__init__()
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.emotion_dim = emotion_dim
        self.hidden_dim = hidden_dim
        
        # Emotion processing network
        self.emotion_processor = nn.Sequential(
            nn.Linear(emotion_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, emotion_dim),
            nn.Sigmoid()
        )
        
        # Context integration network
        self.context_network = nn.Sequential(
            nn.Linear(hidden_dim + emotion_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, emotion_dim),
            nn.Sigmoid()
        )
        
        # Regulation network
        self.regulation_network = nn.Sequential(
            nn.Linear(emotion_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, emotion_dim),
            nn.Sigmoid()
        )
        
        # Adaptive parameters
        self.regulation_strength = nn.Parameter(torch.tensor(0.5))
        self.emotional_memory = nn.Parameter(torch.zeros(emotion_dim))
        self.baseline_emotions = nn.Parameter(torch.ones(emotion_dim) * 0.5)
        
        # Move to device
        self.to(self.device)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Process and regulate emotional state"""
        # Ensure input is on correct device
        x = x.to(self.device)
        
        # Extract emotional features
        emotional_state = self.emotion_processor(x)
        
        # Integrate context
        context_features = torch.mean(x.view(-1, self.hidden_dim), dim=0)
        context = self.context_network(
            torch.cat([context_features, emotional_state.squeeze(0)], dim=0)
        )
        
        # Update emotional memory
        self.emotional_memory.data = (
            0.9 * self.emotional_memory.data +
            0.1 * emotional_state.squeeze(0)
        )
        
        # Regulate emotions
        target_state = self._compute_target_state(emotional_state, context)
        regulated_state = self.regulation_network(
            torch.cat([emotional_state, target_state], dim=-1)
        )
        
        # Apply regulation strength
        final_state = (
            self.regulation_strength * regulated_state +
            (1 - self.regulation_strength) * emotional_state
        )
        
        return final_state
    
    def _compute_target_state(self, 
                            current_state: torch.Tensor,
                            context: torch.Tensor) -> torch.Tensor:
        """Compute target emotional state based on context and baseline"""
        # Weighted combination of baseline and context
        target = (
            0.7 * self.baseline_emotions +
            0.3 * context
        )
        
        # Ensure target state is achievable
        max_change = 0.2  # Maximum allowed change per step
        delta = target - current_state
        clamped_delta = torch.clamp(delta, -max_change, max_change)
        target_state = current_state + clamped_delta
        
        return target_state
    
    def update_baseline(self, emotion_sequence: torch.Tensor):
        """Update baseline emotions based on observed sequence"""
        with torch.no_grad():
            # Calculate moving average
            sequence_mean = torch.mean(emotion_sequence, dim=0)
            
            # Update baseline with momentum
            self.baseline_emotions.data = (
                0.95 * self.baseline_emotions.data +
                0.05 * sequence_mean
            )
    
    def adjust_regulation_strength(self, 
                                 performance: float,
                                 target_performance: float):
        """Adjust regulation strength based on performance"""
        with torch.no_grad():
            # Calculate performance gap
            performance_gap = target_performance - performance
            
            # Adjust regulation strength
            if performance_gap > 0.2:  # Underperforming
                self.regulation_strength.data *= 1.1  # Increase regulation
            elif performance_gap < -0.2:  # Overperforming
                self.regulation_strength.data *= 0.9  # Decrease regulation
            
            # Clamp to reasonable range
            self.regulation_strength.data = torch.clamp(
                self.regulation_strength.data,
                0.1,  # Minimum regulation
                0.9   # Maximum regulation
            )
    
    def get_emotional_state(self) -> Dict[str, float]:
        """Get current emotional state metrics"""
        return {
            'current_memory': self.emotional_memory.tolist(),
            'baseline': self.baseline_emotions.tolist(),
            'regulation_strength': self.regulation_strength.item()
        }
