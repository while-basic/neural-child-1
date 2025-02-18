import torch
import torch.nn as nn
from torch.nn.utils import parametrize
import time
from emotional_regulation import EmotionalRegulation
from memory_module import DifferentiableMemory
from psychological_components import TheoryOfMind, AttachmentSystem, DefenseMechanisms
from typing import Dict

class SensoryExperience:
    def __init__(self, device='cuda'):
        # Initialize sensory channels with learnable parameters
        self.visual = nn.Parameter(torch.randn(512, device=device))
        self.auditory = nn.Parameter(torch.randn(256, device=device))
        self.tactile = nn.Parameter(torch.randn(128, device=device))
        self.proprioceptive = nn.Parameter(torch.randn(64, device=device))
        
        # Sensory integration network - outputs 256 dimensions
        self.integration_net = nn.Sequential(
            nn.Linear(960, 512),  # Combined sensory inputs
            nn.GELU(),
            nn.Linear(512, 256)
        ).to(device)
        
        # Attention weights for each sense
        self.attention = nn.Parameter(torch.ones(4, device=device))
        
    def process_input(self, stimulus):
        # Get batch size from input
        batch_size = stimulus.size(0) if stimulus.dim() > 1 else 1
        stimulus = stimulus.view(batch_size, -1)  # Ensure proper shape
        
        # Expand sensory parameters to match batch size
        visual = self.visual.unsqueeze(0).expand(batch_size, -1)
        auditory = self.auditory.unsqueeze(0).expand(batch_size, -1)
        tactile = self.tactile.unsqueeze(0).expand(batch_size, -1)
        proprioceptive = self.proprioceptive.unsqueeze(0).expand(batch_size, -1)
        attention = self.attention.unsqueeze(0).expand(batch_size, -1)
        
        # Combine all sensory inputs with attention weights
        combined = torch.cat([
            visual * attention[:, 0:1],
            auditory * attention[:, 1:2],
            tactile * attention[:, 2:3],
            proprioceptive * attention[:, 3:4]
        ], dim=1)
        
        # Process through integration network
        return self.integration_net(combined)
        
    def update_sensitivity(self, feedback):
        self.attention.data += torch.tanh(feedback) * 0.1
        self.attention.data = torch.clamp(self.attention.data, 0.1, 2.0)

class CoreDrives:
    def __init__(self, device='cuda'):
        # Basic survival and developmental drives
        self.drives = {
            'curiosity': nn.Parameter(torch.tensor(0.8, device=device)),
            'social_need': nn.Parameter(torch.tensor(0.6, device=device)),
            'safety_need': nn.Parameter(torch.tensor(0.7, device=device)),
            'autonomy': nn.Parameter(torch.tensor(0.3, device=device)),
            'mastery': nn.Parameter(torch.tensor(0.5, device=device))
        }
        
        self.personality_traits = {
            'openness': nn.Parameter(torch.tensor(0.5, device=device)),
            'conscientiousness': nn.Parameter(torch.tensor(0.5, device=device)),
            'extraversion': nn.Parameter(torch.tensor(0.5, device=device)),
            'agreeableness': nn.Parameter(torch.tensor(0.5, device=device)),
            'neuroticism': nn.Parameter(torch.tensor(0.5, device=device))
        }
        
        # Update regulation network dimensions
        total_dims = len(self.drives) + len(self.personality_traits)  # Should be 10
        self.regulation = nn.Sequential(
            nn.Linear(total_dims, 64),
            nn.GELU(),
            nn.Linear(64, total_dims)
        ).to(device)
    
    def get_motivation_vector(self):
        drive_values = torch.stack(list(self.drives.values()))
        personality_values = torch.stack(list(self.personality_traits.values()))
        combined = torch.cat([drive_values, personality_values])
        regulated = self.regulation(combined)
        return torch.sigmoid(regulated)  # This should output a 10-dimensional vector
    
    def update_drives(self, experience_feedback, satisfaction_level):
        for drive in self.drives.values():
            delta = (satisfaction_level - drive.data) * 0.1
            drive.data += delta
            drive.data = torch.clamp(drive.data, 0.1, 1.0)

class CognitiveBiases:
    def __init__(self):
        self.confirmation_bias_strength = 0.3
        
    def apply_confirmation_bias(self, beliefs, new_evidence):
        # Amplify alignment with existing beliefs
        biased_evidence = new_evidence * (1 + self.confirmation_bias_strength * beliefs)
        return biased_evidence

class DynamicNeuralChild(nn.Module):
    def __init__(self, device='cuda', hidden_size: int = 256):
        super().__init__()
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.hidden_size = hidden_size  # Add hidden size parameter
        self.input_size = 768
        self.output_size = 4  # joy, trust, fear, surprise
        
        # Initialize age (starting at early elementary school age)
        self.age = 7.0  # Starting age in years
        self.birth_time = time.time()  # Record birth time for age calculation
        self.aging_rate = 1.0  # Years per real second - can be adjusted
        
        # Core neural layers
        self.layers = nn.ModuleList([
            nn.Linear(self.input_size, self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, self.output_size),
            nn.Tanh()
        ]).to(self.device)
        
        # Add emotion projection layer
        self.emotion_projection_layer = nn.Sequential(
            nn.Linear(4, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.input_size)
        ).to(self.device)
        
        # Initialize emotional state
        self.emotional_state = torch.zeros(4, device=self.device)  # [joy, trust, fear, surprise]
        
    def forward(self, x):
        """Forward pass for neural network"""
        for layer in self.layers:
            x = layer(x)
        return x
        
    def update_emotions(self, mother_vector: torch.Tensor) -> Dict[str, float]:
        """Update emotional state based on mother's input"""
        # Ensure mother_vector is on the correct device
        mother_vector = mother_vector.to(self.device)
        
        # Calculate emotional update
        delta = mother_vector - self.emotional_state
        self.emotional_state += 0.3 * delta + 0.1 * torch.randn_like(delta)
        self.emotional_state = torch.clamp(self.emotional_state, 0, 1)
        
        # Return current emotional state as dict
        return {
            'joy': self.emotional_state[0].item(),
            'trust': self.emotional_state[1].item(),
            'fear': self.emotional_state[2].item(),
            'surprise': self.emotional_state[3].item(),
            'trust_level': self.emotional_state[1].item()  # Using trust as trust_level
        }
        
    def express_feeling(self) -> str:
        """Express current emotional state as text"""
        emotions = self.emotional_state.tolist()
        dominant_emotion = max(enumerate(emotions), key=lambda x: x[1])[0]
        
        if dominant_emotion == 0:
            return "HAPPY"
        elif dominant_emotion == 1:
            return "TRUSTING"
        elif dominant_emotion == 2:
            return "FEARFUL"
        else:
            return "SURPRISED"
        
    def process_interaction(self, message: str) -> str:
        """Process an interaction and generate a response"""
        # Convert message to embeddings (simplified)
        stimulus = torch.randn(1, 512, device=self.device)  # Placeholder for actual embedding
        
        # Process through sensory system
        sensory_output = self.sensory.process_input(stimulus)
        
        # Update emotional state
        emotional_state = self.emotional_system.update(sensory_output)
        
        # Store in memory
        self.memory.store(message, emotional_state)
        
        # Generate response using theory of mind
        response = self.theory_of_mind.generate_response(message, emotional_state)
        
        return response
        
    def get_emotional_state(self) -> dict:
        """Get the current emotional state"""
        return self.emotional_system.get_state()
        
    def get_development_stage(self) -> str:
        """Get the current developmental stage based on age"""
        if self.age >= 11:
            return "EARLY_ADOLESCENCE"
        elif self.age >= 9:
            return "LATE_ELEMENTARY"
        elif self.age >= 8:
            return "MIDDLE_ELEMENTARY"
        else:
            return "EARLY_ELEMENTARY"
            
    def get_age(self) -> float:
        """Get the current age in years, calculated from birth time"""
        elapsed_time = time.time() - self.birth_time
        current_age = self.age + (elapsed_time * self.aging_rate / (365 * 24 * 60 * 60))  # Convert seconds to years
        return current_age

    def add_layer(self) -> None:
        """Add a new layer to the neural network"""
        # Find position to insert new layer (before last linear layer)
        insert_pos = len(self.layers) - 2
        
        # Create new layer with same hidden size
        new_layer = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.GELU()
        ).to(self.device)
        
        # Insert the new layer
        self.layers.insert(insert_pos, new_layer)
        
    def modify_layer(self, layer_idx: int) -> None:
        """Modify an existing layer's architecture"""
        if isinstance(self.layers[layer_idx], nn.Linear):
            # Randomly adjust the layer size while maintaining input/output dimensions
            current_layer = self.layers[layer_idx]
            in_features = current_layer.in_features
            out_features = current_layer.out_features
            
            # Create new layer with same dimensions but new initialization
            new_layer = nn.Linear(in_features, out_features).to(self.device)
            self.layers[layer_idx] = new_layer
