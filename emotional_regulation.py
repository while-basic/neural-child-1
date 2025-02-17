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

@dataclass
class EmotionalState:
    """Represents a complex emotional state with multiple dimensions and derived emotions."""
    # Primary emotions (based on Plutchik's wheel)
    happiness: float  # joy/ecstasy
    sadness: float
    anger: float
    fear: float
    surprise: float = 0.0
    disgust: float = 0.0
    trust: float = 0.5
    anticipation: float = 0.5

    def to_vector(self) -> List[float]:
        """Convert emotional state to a vector representation."""
        return [
            self.happiness, self.sadness, self.anger, self.fear,
            self.surprise, self.disgust, self.trust, self.anticipation
        ]

    @classmethod
    def from_vector(cls, vector: List[float]) -> 'EmotionalState':
        """Create an EmotionalState instance from a vector."""
        return cls(
            happiness=vector[0],
            sadness=vector[1],
            anger=vector[2],
            fear=vector[3],
            surprise=vector[4] if len(vector) > 4 else 0.0,
            disgust=vector[5] if len(vector) > 5 else 0.0,
            trust=vector[6] if len(vector) > 6 else 0.5,
            anticipation=vector[7] if len(vector) > 7 else 0.5
        )

    def get_complex_emotions(self) -> Dict[str, float]:
        """Calculate complex emotions based on combinations of primary emotions."""
        return {
            'love': min(1.0, (self.happiness + self.trust) / 2),
            'submission': min(1.0, (self.trust + self.fear) / 2),
            'awe': min(1.0, (self.fear + self.surprise) / 2),
            'disappointment': min(1.0, (self.surprise + self.sadness) / 2),
            'remorse': min(1.0, (self.sadness + self.disgust) / 2),
            'contempt': min(1.0, (self.disgust + self.anger) / 2),
            'aggressiveness': min(1.0, (self.anger + self.anticipation) / 2),
            'optimism': min(1.0, (self.anticipation + self.happiness) / 2),
            'guilt': min(1.0, (self.fear + self.sadness) / 2),
            'curiosity': min(1.0, (self.anticipation + self.trust) / 2),
            'pride': min(1.0, (self.happiness + self.anticipation) / 2),
            'shame': min(1.0, (self.sadness + self.fear) / 2),
            'anxiety': min(1.0, (self.fear + self.anticipation) / 2),
            'contentment': min(1.0, (self.happiness + self.trust) / 2)
        }

    def get_dominant_emotions(self, threshold: float = 0.5) -> List[Tuple[str, float]]:
        """Get the dominant emotions above a certain threshold."""
        primary = {
            'happiness': self.happiness,
            'sadness': self.sadness,
            'anger': self.anger,
            'fear': self.fear,
            'surprise': self.surprise,
            'disgust': self.disgust,
            'trust': self.trust,
            'anticipation': self.anticipation
        }
        
        complex = self.get_complex_emotions()
        all_emotions = {**primary, **complex}
        
        dominant = [
            (emotion, intensity) 
            for emotion, intensity in all_emotions.items() 
            if intensity >= threshold
        ]
        return sorted(dominant, key=lambda x: x[1], reverse=True)

    def get_emotional_description(self) -> str:
        """Generate a natural language description of the emotional state."""
        dominant = self.get_dominant_emotions(threshold=0.6)
        if not dominant:
            return "feeling neutral"
            
        if len(dominant) == 1:
            emotion, intensity = dominant[0]
            intensity_word = "slightly " if intensity < 0.7 else "very " if intensity > 0.8 else ""
            return f"feeling {intensity_word}{emotion}"
            
        emotions = [emotion for emotion, _ in dominant[:2]]
        return f"feeling a mix of {emotions[0]} and {emotions[1]}"

class EmotionalRegulation(nn.Module):
    """Advanced emotional regulation system with memory and context awareness."""
    
    def __init__(self, 
                 emotion_dim: int = 8,  # Fixed to 8 dimensions for all primary emotions
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
                nn.Linear(hidden_dim, emotion_dim),
                nn.LayerNorm(emotion_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ),
            nn.Sequential(
                nn.Linear(emotion_dim, emotion_dim),
                nn.Sigmoid()
            )
        ])
        
        # Initialize baseline emotions for all 8 dimensions
        self.baseline_emotions = nn.Parameter(torch.tensor([
            0.5,  # happiness
            0.5,  # sadness
            0.5,  # anger
            0.5,  # fear
            0.0,  # surprise
            0.0,  # disgust
            0.5,  # trust
            0.5   # anticipation
        ], device=self.device))
        
        # Adaptive parameters with learned weights
        self.regulation_strength = nn.Parameter(torch.tensor(0.5))
        self.emotional_stability = nn.Parameter(torch.tensor(0.7))
        
        # Emotion mixing weights for all 8 dimensions
        self.emotion_mixing_weights = nn.Parameter(torch.eye(emotion_dim))
        
        # Move to device
        self.to(self.device)
    
    def forward(self, x: torch.Tensor, context: Optional[Dict[str, Any]] = None) -> torch.Tensor:
        """Process and regulate emotional state with context awareness."""
        # Ensure input has correct dimensions
        if x.dim() == 1:
            x = x.unsqueeze(0)
        if x.size(-1) != self.emotion_dim:
            # Pad with default values if necessary
            padded = torch.zeros(x.size(0), self.emotion_dim, device=self.device)
            padded[:, :x.size(-1)] = x
            padded[:, 4:] = torch.tensor([0.0, 0.0, 0.5, 0.5], device=self.device)  # Default values for additional emotions
            x = padded
        
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
        for i, layer in enumerate(self.regulation_network):
            if i == 0:
                # First layer concatenates current and target states
                regulated_state = layer(torch.cat([regulated_state, target_state], dim=-1))
            else:
                # Subsequent layers maintain emotion_dim dimensions
                residual = regulated_state
                regulated_state = layer(regulated_state)
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
        # Ensure both tensors have the same number of dimensions
        if emotional_state.dim() == 1:
            emotional_state = emotional_state.unsqueeze(0)
        if context.dim() == 1:
            context = context.unsqueeze(0)
            
        context_influence = self.context_encoder(
            torch.cat([context, emotional_state], dim=-1)
        )
        return (emotional_state + context_influence) / 2
    
    def _update_memory(self, emotional_state: torch.Tensor, context: Optional[Dict[str, Any]] = None):
        """Update emotional memory with current state and context."""
        # Ensure emotional_state is 1D
        if emotional_state.dim() == 2:
            emotional_state = emotional_state.squeeze(0)
            
        # Create memory entry
        memory = EmotionalMemory(
            content=context.get('content', '') if context else '',
            emotions={
                'primary': emotional_state.detach().cpu().numpy(),
                'context': context or {}
            },
            timestamp=float(torch.rand(1)),  # Simplified timestamp
            intensity=float(emotional_state.mean()),
            valence=float(emotional_state[0] - emotional_state[1]) if len(emotional_state) > 1 else 0.0,  # happiness - sadness
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
