import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional
import numpy as np

class MoralPolicyNetwork(nn.Module):
    def __init__(self, input_dim: int = 384, hidden_dim: int = 256):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Core moral reasoning network
        self.moral_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU()
        )
        
        # Value assessment heads
        self.empathy_head = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid()
        )
        
        self.fairness_head = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid()
        )
        
        self.harm_assessment = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid()
        )
        
        # Decision making network
        self.decision_network = nn.Sequential(
            nn.Linear(hidden_dim // 2 + 3, hidden_dim // 4),  # +3 for moral metrics
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid()
        )
        
        # Moral learning parameters
        self.moral_memory = []
        self.learning_rate = 0.01
        self.discount_factor = 0.95
        
    def forward(self, 
                state: torch.Tensor,
                action: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Evaluate moral implications of state or state-action pair"""
        # Encode state for moral reasoning
        moral_features = self.moral_encoder(state)
        
        # Calculate moral metrics
        empathy_score = self.empathy_head(moral_features)
        fairness_score = self.fairness_head(moral_features)
        harm_score = self.harm_assessment(moral_features)
        
        # Combine features for decision making
        moral_metrics = torch.cat([empathy_score, fairness_score, harm_score], dim=-1)
        decision_features = torch.cat([moral_features, moral_metrics], dim=-1)
        
        # Calculate action value if action is provided
        action_value = None
        if action is not None:
            action_value = self.decision_network(decision_features)
        
        return {
            'empathy': empathy_score,
            'fairness': fairness_score,
            'harm_assessment': harm_score,
            'action_value': action_value,
            'moral_features': moral_features
        }
    
    def evaluate_action(self, 
                       state: torch.Tensor,
                       action: torch.Tensor) -> Tuple[float, Dict[str, float]]:
        """Evaluate the moral implications of an action in a given state"""
        with torch.no_grad():
            evaluation = self.forward(state, action)
            
            # Calculate overall moral value
            moral_value = (
                evaluation['empathy'].item() * 0.4 +
                evaluation['fairness'].item() * 0.3 +
                (1 - evaluation['harm_assessment'].item()) * 0.3
            )
            
            metrics = {
                'empathy': evaluation['empathy'].item(),
                'fairness': evaluation['fairness'].item(),
                'harm_risk': evaluation['harm_assessment'].item(),
                'moral_value': moral_value
            }
            
            return moral_value, metrics
    
    def update_values(self, 
                     state: torch.Tensor,
                     action: torch.Tensor,
                     reward: float,
                     feedback: Dict[str, float]):
        """Update moral values based on feedback and outcomes"""
        # Store experience in moral memory
        self.moral_memory.append({
            'state': state.detach(),
            'action': action.detach(),
            'reward': reward,
            'feedback': feedback
        })
        
        # Limit memory size
        if len(self.moral_memory) > 1000:
            self.moral_memory.pop(0)
        
        # Update network based on recent experiences
        if len(self.moral_memory) >= 10:
            self._batch_update()
    
    def _batch_update(self):
        """Perform batch update on moral network"""
        # Sample recent experiences
        batch = self.moral_memory[-10:]
        
        # Prepare batch data
        states = torch.stack([exp['state'] for exp in batch])
        actions = torch.stack([exp['action'] for exp in batch])
        rewards = torch.tensor([exp['reward'] for exp in batch])
        
        # Calculate target values
        with torch.no_grad():
            current_values = torch.stack([
                self.forward(state, action)['action_value']
                for state, action in zip(states, actions)
            ])
            
            target_values = rewards + self.discount_factor * current_values
        
        # Update network
        self.train()
        for state, action, target in zip(states, actions, target_values):
            output = self.forward(state, action)
            value = output['action_value']
            
            # Calculate loss and update
            loss = nn.functional.mse_loss(value, target)
            loss.backward()
            
            # Apply updates
            with torch.no_grad():
                for param in self.parameters():
                    if param.grad is not None:
                        param.data -= self.learning_rate * param.grad
                        param.grad.zero_()
        
        self.eval()