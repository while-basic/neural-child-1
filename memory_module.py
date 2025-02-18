import torch
from torch import nn
from collections import deque
import random
import time
import math
from typing import Dict, List, Tuple
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
                max_experiences=10000):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.short_term_memory = deque(maxlen=short_term_capacity)
        self.working_memory = deque(maxlen=10)
        self.long_term_clusters: List[MemoryCluster] = []
        self.max_clusters = long_term_capacity // 100
        
        # Updated encoder: maps 768 -> 128
        self.encoder = nn.Sequential(
            nn.Linear(embedding_dim, 256),
            nn.GELU(),
            nn.Linear(256, 128)
        )
        
        # Importance network remains the same (expects concatenated size 906)
        self.importance_net = nn.Sequential(
            nn.Linear(906, 64),  # 902 (memory entry) + 4 (emotional_state) = 906
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        # Updated consolidation network: maps internal state (128) -> 128
        self.consolidation_net = nn.Sequential(
            nn.Linear(128, 256),
            nn.GELU(),
            nn.Linear(256, 128)
        )
        
        self.replay_optimizer = ReplayOptimizer(memory_capacity=long_term_capacity)
        self.forgetting_rate = nn.Parameter(torch.tensor(0.1, device=self.device))
        self.consolidation_threshold = nn.Parameter(torch.tensor(0.7, device=self.device))
        self.emotional_importance = nn.Parameter(torch.ones(4, device=self.device))
        
        self.experiences = deque(maxlen=max_experiences)
        self.max_experiences = max_experiences
        
        self.to(self.device)  # Move the whole module to the specified device
        
    def compute_memory_importance(self, memory_embedding: torch.Tensor, 
                                emotional_state: torch.Tensor) -> float:
        # Ensure both tensors are on the same device
        memory_embedding = memory_embedding.to(self.device)
        emotional_state = emotional_state.to(self.device)
        
        combined = torch.cat([memory_embedding, emotional_state])
        importance = self.importance_net(combined)
        emotional_weight = torch.sum(emotional_state * self.emotional_importance)
        return importance.item() * emotional_weight.item()
    
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

    def record_experience(self, input_data, internal_state, reward, timestamp, emotional_state):
        """Record an experience in memory"""
        experience = {
            'input': input_data,
            'internal_state': internal_state,
            'reward': reward,
            'timestamp': timestamp,
            'emotional_state': emotional_state
        }
        self.experiences.append(experience)
        
    def get_important_experiences(self, k=10):
        """Retrieve k most important experiences based on reward"""
        if not self.experiences:
            return []
            
        # Sort experiences by reward and recency
        sorted_experiences = sorted(
            self.experiences,
            key=lambda x: (x['reward'], -abs(time.time() - x['timestamp'])),
            reverse=True
        )
        
        return sorted_experiences[:k]
        
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
    
    def replay_consolidation(self, batch_size=32):
        """Consolidate memories through replay"""
        if len(self.experiences) < batch_size:
            return []
            
        # Sample random batch of experiences
        indices = torch.randperm(len(self.experiences))[:batch_size]
        batch = [self.experiences[i] for i in indices]
        
        return batch
    
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
