# emotional_regulation.py

import torch
import torch.nn as nn
from collections import deque
from typing import Dict, Any, Optional, List, Tuple
import numpy as np
from dataclasses import dataclass

@dataclass
class EmotionalMemory:
    """Represents a memory with associated emotional context."""
    content: str
    emotions: Dict[str, float]
    timestamp: float
    intensity: float
    valence: float  # -1 to 1, negative to positive
    arousal: float  # 0 to 1, low to high activation

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
    """Advanced emotional regulation system with memory and context awareness."""
    
    def __init__(self, 
                 emotion_dim: int = 8,  # Increased for more emotional dimensions
                 hidden_dim: int = 64,  # Increased for more complex processing
                 memory_size: int = 100,
                 device: Optional[torch.device] = None):
        super().__init__()
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.emotion_dim = emotion_dim
        self.hidden_dim = hidden_dim
        
        # Emotional memory system
        self.memory_buffer = deque(maxlen=memory_size)
        self.emotional_memory = nn.Parameter(torch.zeros(emotion_dim))
        self.memory_attention = nn.MultiheadAttention(hidden_dim, num_heads=4)
        
        # Enhanced emotion processing network
        self.emotion_processor = nn.Sequential(
            nn.Linear(emotion_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, emotion_dim),
            nn.Sigmoid()
        )
        
        # Context-aware processing
        self.context_encoder = nn.Sequential(
            nn.Linear(hidden_dim + emotion_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, emotion_dim),
            nn.Sigmoid()
        )
        
        # Advanced regulation network with residual connections
        self.regulation_network = nn.ModuleList([
            nn.Sequential(
                nn.Linear(emotion_dim * 2, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ),
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ),
            nn.Sequential(
                nn.Linear(hidden_dim, emotion_dim),
                nn.Sigmoid()
            )
        ])
        
        # Adaptive parameters with learned weights
        self.regulation_strength = nn.Parameter(torch.tensor(0.5))
        self.emotional_stability = nn.Parameter(torch.tensor(0.7))
        self.baseline_emotions = nn.Parameter(torch.ones(emotion_dim) * 0.5)
        
        # Emotion mixing weights
        self.emotion_mixing_weights = nn.Parameter(torch.ones(emotion_dim, emotion_dim) / emotion_dim)
        
        # Move to device
        self.to(self.device)
    
    def forward(self, x: torch.Tensor, context: Optional[Dict[str, Any]] = None) -> torch.Tensor:
        """Process and regulate emotional state with context awareness."""
        x = x.to(self.device)
        
        # Process basic emotions
        emotional_state = self.emotion_processor(x)
        
        # Apply context if available
        if context:
            context_tensor = self._encode_context(context)
            emotional_state = self._apply_context(emotional_state, context_tensor)
        
        # Update emotional memory
        self._update_memory(emotional_state, context)
        
        # Get memory-influenced target state
        target_state = self._compute_target_state(emotional_state)
        
        # Apply regulation through residual network
        regulated_state = emotional_state
        for layer in self.regulation_network:
            residual = regulated_state
            regulated_state = layer(torch.cat([regulated_state, target_state], dim=-1))
            if regulated_state.shape == residual.shape:
                regulated_state = regulated_state + residual
        
        # Apply adaptive regulation strength
        final_state = (
            self.regulation_strength * regulated_state +
            (1 - self.regulation_strength) * emotional_state
        )
        
        # Apply emotion mixing for complex emotional states
        mixed_emotions = torch.matmul(final_state, self.emotion_mixing_weights)
        final_state = torch.sigmoid(mixed_emotions)
        
        return final_state
    
    def _encode_context(self, context: Dict[str, Any]) -> torch.Tensor:
        """Encode contextual information into a tensor."""
        # Convert context features to tensor
        context_features = []
        if 'valence' in context:
            context_features.append(float(context['valence']))
        if 'arousal' in context:
            context_features.append(float(context['arousal']))
        if 'intensity' in context:
            context_features.append(float(context['intensity']))
            
        # Pad or truncate to match hidden_dim
        while len(context_features) < self.hidden_dim:
            context_features.append(0.0)
        context_features = context_features[:self.hidden_dim]
        
        return torch.tensor(context_features, device=self.device)
    
    def _apply_context(self, emotional_state: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """Apply contextual modulation to emotional state."""
        context_influence = self.context_encoder(
            torch.cat([context, emotional_state], dim=-1)
        )
        return (emotional_state + context_influence) / 2
    
    def _update_memory(self, emotional_state: torch.Tensor, context: Optional[Dict[str, Any]] = None):
        """Update emotional memory with current state and context."""
        # Create memory entry
        memory = EmotionalMemory(
            content=context.get('content', '') if context else '',
            emotions={
                'primary': emotional_state.detach().cpu().numpy(),
                'context': context or {}
            },
            timestamp=float(torch.rand(1)),  # Simplified timestamp
            intensity=float(emotional_state.mean()),
            valence=float(emotional_state[0] - emotional_state[1]),  # happiness - sadness
            arousal=float(emotional_state.std())
        )
        
        self.memory_buffer.append(memory)
        
        # Update emotional memory with exponential decay
        decay = 0.95
        self.emotional_memory.data = (
            decay * self.emotional_memory.data +
            (1 - decay) * emotional_state.detach()
        )
    
    def _compute_target_state(self, current_state: torch.Tensor) -> torch.Tensor:
        """Compute target emotional state using memory and baseline."""
        # Get recent memory influence
        memory_influence = torch.zeros_like(current_state)
        if self.memory_buffer:
            recent_states = torch.stack([
                torch.tensor(m.emotions['primary'], device=self.device)
                for m in list(self.memory_buffer)[-5:]
            ])
            memory_influence = recent_states.mean(dim=0)
        
        # Combine baseline, memory, and stability
        target = (
            0.4 * self.baseline_emotions +
            0.3 * memory_influence +
            0.3 * self.emotional_memory
        )
        
        # Apply emotional stability as a smoothing factor
        max_change = 1.0 - self.emotional_stability
        delta = target - current_state
        clamped_delta = torch.clamp(delta, -max_change, max_change)
        target_state = current_state + clamped_delta
        
        return target_state
    
    def get_emotional_state(self) -> Dict[str, Any]:
        """Get detailed emotional state metrics."""
        return {
            'current_memory': self.emotional_memory.tolist(),
            'baseline': self.baseline_emotions.tolist(),
            'regulation_strength': float(self.regulation_strength),
            'emotional_stability': float(self.emotional_stability),
            'memory_size': len(self.memory_buffer),
            'recent_memories': [
                {
                    'timestamp': m.timestamp,
                    'intensity': m.intensity,
                    'valence': m.valence,
                    'arousal': m.arousal
                }
                for m in list(self.memory_buffer)[-5:]
            ]
        }
