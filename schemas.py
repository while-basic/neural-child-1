from pydantic import BaseModel
from typing import Optional

class MotherResponse(BaseModel):
    """Structured response schema for mother-child interactions"""
    content: str
    emotional_context: Optional[dict] = {
        'joy': 0.5,
        'trust': 0.5, 
        'fear': 0.1,
        'surprise': 0.3
    }
    reward_score: float = 0.7
    success_metric: float = 0.0
    complexity_rating: float = 0.0
    self_critique_score: float = 0.0
    cognitive_labels: Optional[list] = []
    
    model_config = {
        "json_schema_extra": {
            "examples": [{
                "content": "That's a good attempt! [HUG]",
                "emotional_context": {
                    "joy": 0.8,
                    "trust": 0.6,
                    "fear": 0.05,
                    "surprise": 0.1
                },
                "reward_score": 0.85,
                "success_metric": 0.7,
                "complexity_rating": 0.4,
                "self_critique_score": 0.3,
                "cognitive_labels": ["encouragement", "basic_concept"]
            }]
        }
    }
