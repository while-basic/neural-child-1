from pydantic import BaseModel
from typing import List, Dict

class EmotionalState(BaseModel):
    """Represents the emotional state with various dimensions."""
    happiness: float
    sadness: float
    anger: float
    fear: float
    surprise: float
    disgust: float
    trust: float
    anticipation: float

class WarningMetrics(BaseModel):
    """Metrics used for monitoring the child's development."""
    emotional_stability: float
    learning_efficiency: float
    attention_level: float
    overstimulation_risk: float

class WarningIndicators(BaseModel):
    """Warning indicators for the child's development state."""
    warning_state: str
    metrics: WarningMetrics
    recent_warnings: List[Dict[str, str]]

class InteractionRequest(BaseModel):
    """Request model for interactions with the child."""
    category: str
    interaction: str

class DevelopmentState(BaseModel):
    """Overall development state of the child."""
    emotionalState: EmotionalState
    warnings: WarningIndicators
    developmentSpeed: float
    currentStage: str
    ageMonths: float
