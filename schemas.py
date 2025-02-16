from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict
from enum import Enum

class EmotionalContext(BaseModel):
    joy: float = Field(default=0.5, ge=0.0, le=1.0)
    trust: float = Field(default=0.5, ge=0.0, le=1.0)
    fear: float = Field(default=0.1, ge=0.0, le=1.0)
    surprise: float = Field(default=0.3, ge=0.0, le=1.0)
    
    @validator('*')
    def validate_emotion_range(cls, v):
        if not 0 <= v <= 1:
            raise ValueError('Emotional values must be between 0 and 1')
        return v

class ActionType(str, Enum):
    FEED = "FEED"
    SLEEP = "SLEEP"
    COMFORT = "COMFORT"
    PLAY = "PLAY"
    TEACH = "TEACH"
    ENCOURAGE = "ENCOURAGE"
    PRAISE = "PRAISE"
    GUIDE = "GUIDE"
    EXPLORE = "EXPLORE"
    REFLECT = "REFLECT"

class MotherResponse(BaseModel):
    """Structured response schema for mother-child interactions"""
    content: str = Field(..., min_length=1, max_length=1000)
    emotional_context: EmotionalContext = Field(default_factory=EmotionalContext)
    action: Optional[ActionType] = None
    reward_score: float = Field(default=0.7, ge=0.0, le=1.0)
    success_metric: float = Field(default=0.0, ge=0.0, le=1.0)
    complexity_rating: float = Field(default=0.0, ge=0.0, le=1.0)
    self_critique_score: float = Field(default=0.0, ge=0.0, le=1.0)
    cognitive_labels: List[str] = Field(default_factory=list)
    effectiveness: float = Field(default=0.5, ge=0.0, le=1.0)
    developmental_focus: Optional[Dict[str, float]] = None
    
    @validator('content')
    def validate_content_markers(cls, v):
        if '[' not in v or ']' not in v:
            raise ValueError('Content must contain action markers in [BRACKETS]')
        return v
    
    @validator('cognitive_labels')
    def validate_labels(cls, v):
        if not all(isinstance(label, str) for label in v):
            raise ValueError('All cognitive labels must be strings')
        return v
    
    @validator('developmental_focus')
    def validate_focus(cls, v):
        if v is not None:
            if not all(isinstance(k, str) and isinstance(v, float) for k, v in v.items()):
                raise ValueError('Developmental focus must be a dict of string keys and float values')
            if not all(0 <= value <= 1 for value in v.values()):
                raise ValueError('Developmental focus values must be between 0 and 1')
        return v
    
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
                "action": "COMFORT",
                "reward_score": 0.85,
                "success_metric": 0.7,
                "complexity_rating": 0.4,
                "self_critique_score": 0.3,
                "cognitive_labels": ["encouragement", "basic_concept"],
                "effectiveness": 0.75,
                "developmental_focus": {
                    "emotional_regulation": 0.8,
                    "social_skills": 0.6,
                    "cognitive_development": 0.4
                }
            }]
        }
    }
