import torch
import torch.nn as nn
from collections import deque
import random
import time
import math
from typing import Dict, List, Tuple, Any
import config
from replay_system import ReplayOptimizer

class MemoryCluster:
    """Represents a cluster of related memories"""
    def __init__(self, centroid: torch.Tensor):
        self.centroid = centroid
        self.memories = []
        self.importance = 1.0
        self.last_accessed = time.time()
        
    def add_memory(self, memory: torch.Tensor):
        self.memories.append(memory)
        self.centroid = torch.mean(torch.stack([m[0] for m in self.memories]), dim=0)
        
    def get_age(self) -> float:
        return time.time() - self.last_accessed

class DifferentiableMemory(nn.Module):
    def __init__(self, embedding_dim: int = 768, 
                short_term_capacity: int = 1000,
                long_term_capacity: int = 50000,
                device=None):
        super().__init__()
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        
        self.short_term_memory = deque(maxlen=short_term_capacity)
        self.working_memory = deque(maxlen=10)
        self.long_term_clusters: List[MemoryCluster] = []
        self.max_clusters = long_term_capacity // 100
        self.experiences = []  # Add experiences list
        
        # Updated encoder: maps 768 -> 128
        self.encoder = nn.Sequential(
            nn.Linear(embedding_dim, 256),
            nn.GELU(),
            nn.Linear(256, 128)
        )
        
        # Standardize input dimensions
        self.input_dim = embedding_dim
        self.internal_state_dim = 128
        self.emotional_state_dim = 4
        
        total_input_dim = (
            self.input_dim +            # Input embedding
            self.internal_state_dim +   # Internal state
            1 +                         # Reward
            1 +                         # Timestamp
            self.emotional_state_dim    # Emotional state
        )
        
        # Updated importance network
        self.importance_net = nn.Sequential(
            nn.Linear(total_input_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        ).to(self.device)
        
        # Updated consolidation network: maps internal state (128) -> 128
        self.consolidation_net = nn.Sequential(
            nn.Linear(128, 256),
            nn.GELU(),
            nn.Linear(256, 128)
        )
        
        self.replay_optimizer = ReplayOptimizer(memory_capacity=long_term_capacity, device=self.device)
        self.forgetting_rate = nn.Parameter(torch.tensor(0.1, device=self.device))
        self.consolidation_threshold = nn.Parameter(torch.tensor(0.7, device=self.device))
        self.emotional_importance = nn.Parameter(torch.ones(4, device=self.device))
        
        self.to(self.device)  # Move the whole module to the specified device
        
    def compute_memory_importance(self, memory_embedding: torch.Tensor, 
                                emotional_state: torch.Tensor) -> float:
        """Compute importance score for a memory with dimension checks."""
        try:
            # Ensure inputs are on correct device and have correct shape
            memory_embedding = memory_embedding.to(self.device)
            emotional_state = emotional_state.to(self.device)
            
            # Reshape memory embedding
            if memory_embedding.dim() == 2:
                memory_embedding = memory_embedding.squeeze(0)
            
            # Ensure memory embedding has correct dimension
            if memory_embedding.shape[0] != self.input_dim:
                if memory_embedding.shape[0] > self.input_dim:
                    memory_embedding = memory_embedding[:self.input_dim]
                else:
                    padding = torch.zeros(self.input_dim - memory_embedding.shape[0], 
                                       device=self.device)
                    memory_embedding = torch.cat([memory_embedding, padding])
            
            # Reshape emotional state
            if emotional_state.dim() == 2:
                emotional_state = emotional_state.squeeze(0)
            if emotional_state.shape[0] != self.emotional_state_dim:
                emotional_state = torch.zeros(self.emotional_state_dim, device=self.device)
            
            # Create properly sized tensors
            internal_state = torch.zeros(self.internal_state_dim, device=self.device)
            reward = torch.tensor([0.5], device=self.device)
            timestamp = torch.tensor([time.time()], device=self.device)
            
            # Concatenate all components
            combined = torch.cat([
                memory_embedding,
                internal_state,
                reward,
                timestamp,
                emotional_state
            ])
            
            # Add batch dimension and compute importance
            combined = combined.unsqueeze(0)
            importance = self.importance_net(combined)
            emotional_weight = torch.mean(emotional_state)
            
            return importance.item() * emotional_weight.item()
            
        except Exception as e:
            print(f"Error in compute_memory_importance: {str(e)}")
            print(f"Memory shape: {memory_embedding.shape}, Emotional shape: {emotional_state.shape}")
            return 0.5
    
    def find_similar_cluster(self, memory_embedding: torch.Tensor) -> Tuple[MemoryCluster, float]:
        if not self.long_term_clusters:
            return None, 0
        similarities = [torch.cosine_similarity(memory_embedding, cluster.centroid, dim=0)
                        for cluster in self.long_term_clusters]
        max_sim, idx = max((s, i) for i, s in enumerate(similarities))
        return self.long_term_clusters[idx], max_sim.item()
    
    def consolidate_memory(self, memory: torch.Tensor, importance: float):
        encoded_memory = self.encoder(memory)
        cluster, similarity = self.find_similar_cluster(encoded_memory)
        if similarity > self.consolidation_threshold:
            cluster.add_memory((encoded_memory, importance))
            cluster.importance = max(cluster.importance, importance)
            cluster.last_accessed = time.time()
        else:
            if len(self.long_term_clusters) < self.max_clusters:
                new_cluster = MemoryCluster(encoded_memory)
                new_cluster.add_memory((encoded_memory, importance))
                self.long_term_clusters.append(new_cluster)
            else:
                least_important = min(self.long_term_clusters, key=lambda c: c.importance * math.exp(-c.get_age() / 86400))
                self.long_term_clusters.remove(least_important)
                new_cluster = MemoryCluster(encoded_memory)
                new_cluster.add_memory((encoded_memory, importance))
                self.long_term_clusters.append(new_cluster)

    def record_experience(self, input_vec: torch.Tensor, internal_state: torch.Tensor, 
                        reward: float, timestamp: float, emotional_state: torch.Tensor):
        # Move all tensors to the same device
        input_vec = input_vec.to(self.device)
        internal_state = internal_state.to(self.device)
        reward_tensor = torch.tensor([reward], device=self.device)
        timestamp_tensor = torch.tensor([timestamp], device=self.device)
        emotional_state = emotional_state.to(self.device)
        
        # Create memory entry
        memory_entry = torch.cat([
            input_vec.squeeze(0),
            internal_state.squeeze(0),
            reward_tensor,
            timestamp_tensor,
            emotional_state if emotional_state.dim() == 1 else emotional_state.squeeze(0)
        ])
        
        self.short_term_memory.append(memory_entry)
        importance = self.compute_memory_importance(memory_entry, emotional_state)
        
        if importance > 0.8:
            self.working_memory.append(memory_entry)
        
        self.replay_optimizer.add_experience(memory_entry)
        return importance
    
    def forget_memories(self):
        # Create forget mask on the same device as forgetting_rate
        forget_mask = (torch.rand(len(self.short_term_memory), device=self.device) > 
                    self.forgetting_rate)
        
        # Update the short term memory with the mask
        self.short_term_memory = deque([m for i, m in enumerate(self.short_term_memory) 
                                    if forget_mask[i].item()])
        
        # Handle long term clusters
        clusters_to_remove = []
        for cluster in self.long_term_clusters:
            age_factor = math.exp(-cluster.get_age() / 86400)
            if random.random() > (cluster.importance * age_factor):
                clusters_to_remove.append(cluster)
        
        # Remove clusters outside the loop to avoid modifying while iterating
        for cluster in clusters_to_remove:
            self.long_term_clusters.remove(cluster)
    
    def replay_consolidation(self, batch_size=32, emotional_state=None):
        samples, indices = self.replay_optimizer.sample_batch(batch_size)
        losses = []
        with torch.no_grad():
            for sample in samples:
                encoded = self.encoder(sample[:768])
                # Use the internal state part (indices 768:896, which is 128-dimensional)
                target = self.consolidation_net(sample[768:896])
                loss = nn.MSELoss()(encoded, target)
                losses.append(loss.item())
                if emotional_state is not None:
                    importance = self.compute_memory_importance(sample, emotional_state)
                    if importance > self.consolidation_threshold:
                        self.consolidate_memory(sample, importance)
        self.replay_optimizer.update_weights(indices, torch.tensor(losses, device=self.device))
        self.forget_memories()
        return torch.tensor(losses).mean().item()
    
    def retrieve_memories(self, cue: torch.Tensor, top_k: int = 5) -> List[torch.Tensor]:
        if not self.long_term_clusters:
            return []
        encoded_cue = self.encoder(cue)
        similarities = [torch.cosine_similarity(encoded_cue, cluster.centroid, dim=0)
                        for cluster in self.long_term_clusters]
        top_indices = torch.topk(torch.stack(similarities), top_k).indices
        retrieved_memories = []
        for idx in top_indices:
            retrieved_memories.extend([mem[0] for mem in self.long_term_clusters[idx].memories])
        return retrieved_memories[:top_k]
