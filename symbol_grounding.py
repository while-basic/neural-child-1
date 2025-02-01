import torch
from text_embed import get_embeddings

class SymbolGrounding:
    def __init__(self):
        self.concept_map = {}
        self.reverse_map = {}
        self.embedding_matrix = torch.empty((0, 768), device='cuda')
        
    def add_symbol(self, concept: str, token: str):
        embedding = torch.tensor(get_embeddings(concept)[0]['embedding'], device='cuda')
        self.concept_map[token] = embedding
        self.reverse_map[tuple(embedding.cpu().numpy())] = token
        self.embedding_matrix = torch.cat([self.embedding_matrix, embedding.unsqueeze(0)], dim=0)
        
    def get_token(self, embedding: torch.Tensor) -> str:
        similarities = torch.matmul(self.embedding_matrix, embedding)
        closest_idx = torch.argmax(similarities)
        return self.reverse_map[tuple(self.embedding_matrix[closest_idx].cpu().numpy())]
    
    def batch_ground(self, concepts: list) -> dict:
        embeddings = get_embeddings(concepts)
        return {
            concept: {
                'token': f"[{concept.upper()}]",
                'embedding': torch.tensor(emb['embedding'], device='cuda')
            } for concept, emb in zip(concepts, embeddings)
        }
