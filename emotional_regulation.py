# emotional_regulation.py

import torch
import torch.nn as nn
from collections import deque

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
    def __init__(self, emotion_dim=4, hidden_dim=64, device='cpu'):
        super().__init__()
        self.device = device
        self.baseline = torch.zeros(emotion_dim, device=device)
        self.emotional_memory = torch.zeros((100, emotion_dim), device=device)
        self.memory_pointer = 0
        
        self.regulation_network = nn.Sequential(
            nn.Linear(emotion_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, emotion_dim)
        ).to(device)
        
        self.context_processor = nn.LSTM(
            input_size=emotion_dim,
            hidden_size=emotion_dim * 2,
            num_layers=2,
            batch_first=True,
            dropout=0.1
        ).to(device)
        
        self.stability_net = nn.Sequential(
            nn.Linear(emotion_dim * 2 + hidden_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, emotion_dim)
        ).to(device)
        
        self.memory_gate = nn.Sequential(
            nn.Linear(hidden_dim + emotion_dim, 64),
            nn.GELU(),
            nn.Linear(64, emotion_dim),
            nn.Sigmoid()
        ).to(device)
        
        self.emotional_history = deque(maxlen=5)
        self.to(self.device)
        
    def update_baseline(self):
        if self.emotional_history:
            recent_emotions = torch.stack(list(self.emotional_history))
            alpha = 0.1
            self.baseline = alpha * recent_emotions.mean(dim=0) + (1 - alpha) * self.baseline
            
    def detect_trauma(self, emotional_state):
        intensity = torch.norm(emotional_state - self.baseline)
        duration = len([e for e in self.emotional_history if torch.norm(e - self.baseline) > 0.7])
        return {
            'is_traumatic': intensity > 1.0,
            'duration': duration,
            'intensity': intensity.item()
        }
        
    def compute_regulation_strength(self, emotional_state):
        deviation = torch.abs(emotional_state - self.baseline)
        return torch.sigmoid(deviation * 1.0)
        
    def regulate(self, emotional_state, stimulus, memory_context=None):
        if len(self.emotional_history) >= 2:
            context_tensor = torch.stack(list(self.emotional_history))
            context_output, _ = self.context_processor(context_tensor.unsqueeze(0))
            context_embedding = context_output[0, -1]
        else:
            context_embedding = torch.zeros(self.emotion_dim * 2, device=self.device)
        if memory_context is not None:
            memory_influence = self.memory_gate(torch.cat([memory_context, emotional_state], dim=-1))
            context_embedding = context_embedding * memory_influence
        else:
            memory_influence = None
        combined_input = torch.cat([
            context_embedding,
            memory_context if memory_context is not None else torch.zeros(64, device=self.device)
        ], dim=-1)
        regulated_response = self.stability_net(combined_input)
        new_state = torch.clamp(regulated_response, 0, 1)
        self.emotional_history.append(emotional_state.detach())
        return {
            'emotional_state': new_state,
            'context_influence': context_embedding,
            'memory_influence': memory_influence
        }
