# self_supervised_trainer.py

import torch
torch.autograd.set_detect_anomaly(True)
import torch.nn as nn
import time

class AutonomousTrainer:
    def __init__(self, child_model, memory, moral_net):
        self.child = child_model
        self.memory = memory
        self.moral_net = moral_net
        self.optimizer = torch.optim.AdamW(child_model.parameters(), lr=3e-4)
        
    def training_step(self, inputs):
        """
        Safe and efficient training step that avoids in-place operations
        """
        # Start with a clean slate
        self.optimizer.zero_grad()
        
        # Make sure inputs are properly cloned and require gradients
        inputs = inputs.clone().detach().requires_grad_(True)
        
        # Forward pass - create new tensors instead of modifying in-place
        outputs = self.child(inputs)
        moral_feedback = self.moral_net(outputs)['moral_score']
        
        # Safely store experience without affecting gradient computation
        with torch.no_grad():
            self.memory.record_experience(
                inputs.clone(),
                outputs.clone(),
                moral_feedback.item(),
                time.time(),
                self.child.emotional_state.clone()  # Clone emotional state too!
            )
        
        # Process batch and get loss
        replay_loss = self._process_batch()
        
        # Backward pass
        replay_loss.backward()
        
        # Clip gradients safely
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.child.parameters(), 
            max_norm=1.0
        )
        
        # Step optimizer
        self.optimizer.step()
        
        # Safely update beliefs
        with torch.no_grad():
            self.child.current_beliefs = outputs.clone().detach()
        
        return replay_loss.item()

    def _process_batch(self, batch_size=32):
        """
        Safe batch processing that avoids in-place operations
        """
        samples, indices = self.memory.replay_optimizer.sample_batch(batch_size)
        if not samples:
            return torch.tensor(0.0, device=self.child.device)
        
        # Safely create tensors
        inputs = torch.stack([
            sample[:768].clone().detach().requires_grad_(True) 
            for sample in samples
        ]).to(self.child.device)
        
        past_states = torch.stack([
            sample[768:896].clone().detach() 
            for sample in samples
        ]).to(self.child.device)
        
        rewards = torch.stack([
            sample[896].clone().detach() 
            for sample in samples
        ]).to(self.child.device)
        
        # Forward pass
        current_outputs = self.child(inputs)
        
        # Compute losses safely - no in-place operations
        similarity = nn.CosineSimilarity(dim=-1)(current_outputs, past_states)
        consistency_loss = torch.ones_like(similarity).mean() - similarity.mean()
        
        moral_scores = self.moral_net(current_outputs)['moral_score']
        moral_loss = nn.MSELoss()(moral_scores.squeeze(), rewards)
        
        # Compute EWC loss safely
        ewc_loss = sum(
            torch.norm(param, p=2)
            for name, param in self.child.named_parameters()
            if param.requires_grad and '_plasticity' in name
        )
        
        # Combine losses without in-place operations
        total_loss = (
            0.7 * consistency_loss + 
            0.3 * moral_loss + 
            0.1 * ewc_loss
        )
        
        return total_loss
    
    def _safe_tensor_op(self, tensor, requires_grad=False):
        """Helper method for safe tensor operations"""
        result = tensor.clone().detach()
        if requires_grad:
            result.requires_grad_(True)
        return result.to(self.child.device)