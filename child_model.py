import torch
import torch.nn as nn
from torch.nn.utils import parametrize
import time
from emotional_regulation import EmotionalRegulation
from memory_module import DifferentiableMemory
from psychological_components import TheoryOfMind, AttachmentSystem, DefenseMechanisms

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
    def __init__(self, base_dim=128):
        super().__init__()
        self.base_dim = base_dim
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize core components
        self.sensory = SensoryExperience(self.device)
        self.drives = CoreDrives(self.device)
        
        # Initialize psychological components
        self.theory_of_mind = TheoryOfMind(self.device)
        self.attachment = AttachmentSystem(self.device)
        self.defense_mechanisms = DefenseMechanisms(self.device)
        
        # Enhanced emotional system
        self.emotional_regulation = EmotionalRegulation(
            emotion_dim=4,
            context_window=5,
            memory_dim=32
        )
        
        # Memory systems
        self.memory = DifferentiableMemory()
        self.emotional_state = torch.zeros(4, device=self.device)
        
        # Projection and embedding layers
        self.input_projection = nn.Linear(768, base_dim).to(self.device)
        self.drive_projection = nn.Linear(10, len(self.drives.drives)).to(self.device)
        self.psychological_projection = nn.Linear(
            base_dim + 256 + len(self.drives.drives) + 4, 
            512
        ).to(self.device)
        
        # Core processing layers with psychological integration
        # FIX: Changed input dimension from (base_dim+256+5+4=393) to base_dim (128)
        self.core_layers = nn.ModuleList([
            nn.Linear(base_dim, base_dim),
            nn.LayerNorm(base_dim),
            nn.GELU(),
            nn.Dropout(0.1)
        ])
        
        # Update the decision network to match our actual input size (398)
        self.decision_network = nn.Sequential(
            nn.Linear(398, 512),  # Changed from 393 to 398 to match our input
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, base_dim)
        ).to(self.device)
        
        # Cognitive systems
        self.cognitive_biases = CognitiveBiases()
        self.current_beliefs = torch.zeros(base_dim, device=self.device)
        
        # Growth and plasticity parameters
        self.growth_rate = 1.2
        self.current_dim = base_dim
        
        # Attribute to store last attachment trust for loss computation
        self.last_attachment_trust = torch.tensor(0.5, device=self.device)
        
        # Move everything to device
        self.to(self.device)

    def update_emotions(self, mother_vector):
        attachment_state = self.attachment(mother_vector)
        attachment_influence = attachment_state['trust_level']
        regulation_result = self.emotional_regulation.regulate(
            self.emotional_state,
            mother_vector
        )
        regulated_emotion = regulation_result['emotional_state'] * attachment_influence
        anxiety_level = regulated_emotion[2]  # Fear component
        defense_response = self.defense_mechanisms(mother_vector, anxiety_level)
        if defense_response['active_defense'] is not None:
            regulated_emotion = regulated_emotion * (1 - defense_response['defense_strength'])
        self.emotional_state = regulated_emotion
        self.memory.record_experience(
            mother_vector,
            self.emotional_state.unsqueeze(0),
            regulation_result['context_influence'].mean().item(),
            time.time(),
            self.emotional_state
        )
        trauma_info = regulation_result.get('trauma_info', {})
        if trauma_info.get('is_traumatic', False):
            print(f"⚠️ Trauma detected! Intensity: {trauma_info.get('intensity', 0):.2f}")
            self._process_trauma(trauma_info)
        return {
            'regulation_result': regulation_result,
            'attachment_state': attachment_state,
            'defense_response': defense_response
        }
        
    def _process_trauma(self, trauma_info):
        self.defense_mechanisms.anxiety_threshold.data *= 0.95
        self.attachment.update_attachment(0.3)
        self.memory.replay_consolidation(
            batch_size=64,
            emotional_state=self.emotional_state  
        )

    def express_feeling(self):
        baseline = torch.zeros_like(self.emotional_state)
        deviation = self.emotional_state - baseline
        if torch.norm(deviation) < 0.2:
            return "[CALM]"
        joy, trust, fear, surprise = self.emotional_state.tolist()
        feelings = []
        attachment_state = self.attachment.attachment_styles.tolist()
        defense_active = self.defense_mechanisms(
            self.emotional_state.unsqueeze(0), 
            torch.tensor(fear, device=self.device)
        )['active_defense']
        if joy >= 0.8 and attachment_state[0] > 0.5:
            feelings.append("HAPPY" * min(int(joy * 3), 3))
        elif joy >= 0.6:
            feelings.append("content")
        if trust >= 0.8:
            feelings.append("affectionate")
        elif trust >= 0.6:
            feelings.append("warm")
        if fear >= 0.7 and not defense_active:
            feelings.append("terrified")
        elif fear >= 0.5:
            feelings.append("anxious")
        if surprise >= 0.7:
            feelings.append("startled")
        elif surprise >= 0.5:
            feelings.append("curious")
        if not feelings:
            if torch.all(self.emotional_state < 0.3):
                feelings.append("tired")
            else:
                feelings.append("neutral")
        return "[" + " & ".join(feelings).upper() + "]"

    def grow_layer(self):
        new_dim = int(self.current_dim * self.growth_rate)
        # FIX: Change the input dimension to self.current_dim rather than (self.current_dim + 256 + len(self.drives.drives))
        self.core_layers.extend([
            nn.Linear(self.current_dim, new_dim),
            nn.LayerNorm(new_dim),
            nn.GELU(),
            nn.Linear(new_dim, new_dim)
        ])
        self.current_dim = new_dim
        for param in self.parameters():
            try:
                parametrize.register_parametrization(
                    param, 'plasticity', 
                    nn.Identity()
                )
            except Exception:
                pass

    def _reparametrize_weights(self):
        for param in self.parameters():
            parametrize.register_parametrization(
                param, 'plasticity', 
                nn.utils.parametrization.L0Parametrization(param.size(0))
            )

    def forward(self, x):
        # Ensure input has batch dimension
        if x.dim() == 2:
            batch_size = x.size(0)
        else:
            x = x.unsqueeze(0)  # Add batch dimension if it's missing
            batch_size = 1
        
        # Project input
        x = self.input_projection(x)
        
        # Process sensory input and ensure batch dimension
        sensory_output = self.sensory.process_input(x)
        
        # Get drive vector and ensure batch dimension
        drive_vector = self.drives.get_motivation_vector().unsqueeze(0).expand(batch_size, -1)
        
        # Prepare emotional state with batch dimension
        emotional_state = self.emotional_state.unsqueeze(0).expand(batch_size, -1)
        
        # # For debugging
        # print(f"x shape: {x.shape}")
        # print(f"sensory_output shape: {sensory_output.shape}")
        # print(f"drive_vector shape: {drive_vector.shape}")
        # print(f"emotional_state shape: {emotional_state.shape}")
        
        # Verify all tensors have correct batch dimension
        assert x.size(0) == batch_size
        assert sensory_output.size(0) == batch_size
        assert drive_vector.size(0) == batch_size
        assert emotional_state.size(0) == batch_size
        
        # Now all tensors should have matching dimensions for concatenation
        combined_input = torch.cat([x, sensory_output, drive_vector, emotional_state], dim=-1)
        # print(f"combined_input shape: {combined_input.shape}")
        
        # Process through theory of mind and attachment
        theory_of_mind_output = self.theory_of_mind(combined_input)
        attachment_output = self.attachment(self.emotional_state)
        self.last_attachment_trust = attachment_output['trust_level']
        
        # Process through decision network
        output = self.decision_network(combined_input)
        output = self.cognitive_biases.apply_confirmation_bias(self.current_beliefs, output)
        
        # Process through core layers
        for layer in self.core_layers:
            output = layer(output)
            if isinstance(layer, nn.Linear) and self.training:
                curiosity_level = self.drives.drives['curiosity'].item() * attachment_output['trust_level'].item()
                mask = torch.rand_like(output) > (0.1 / curiosity_level)
                output = output * mask
        
        if self.training:
            anxiety_level = emotional_state[0, 2]
            defense_response = self.defense_mechanisms(combined_input, anxiety_level)
            if defense_response['active_defense'] is not None:
                output = output * (1 - defense_response['defense_strength'])
        
        return output

    def update_drives_and_senses(self, feedback, satisfaction):
        self.drives.update_drives(feedback, satisfaction)
        self.sensory.update_sensitivity(feedback)
        self.theory_of_mind.update_relationship_model(feedback, satisfaction)
        self.attachment.update_attachment(satisfaction)
        self.defense_mechanisms.update_threshold(1 - satisfaction)
