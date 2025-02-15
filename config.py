import torch
from dataclasses import dataclass
from typing import List, Dict
from developmental_stages import DevelopmentalStage

# Server configurations
EMBEDDING_SERVER_URL = "http://localhost:1234"  # Change as needed
CHAT_SERVER_URL = "http://localhost:1234"       # Change as needed

# Model dimensions
EMBEDDING_DIM = 768  # Standard embedding size
HIDDEN_DIM = 128    # Hidden layer size

# Default values
DEFAULT_EMBEDDING = torch.zeros(EMBEDDING_DIM)

@dataclass
class StageRequirements:
    required_skills: List[str]
    learning_focus: List[str]
    emotional_range: List[str]
    allowed_actions: List[str]

# Default response for error handling
DEFAULT_RESPONSE = {
    'text': 'I cannot process that right now.',
    'emotional_vector': [0.5, 0.5, 0.5, 0.5]
}

# Stage definitions mapping
STAGE_DEFINITIONS = {
    DevelopmentalStage.NEWBORN: StageRequirements(
        required_skills=['basic reflexes', 'crying'],
        learning_focus=['sensory processing', 'basic motor skills'],
        emotional_range=['happiness', 'sadness'],
        allowed_actions=['soothe', 'feed', 'hold']
    ),
    DevelopmentalStage.INFANT: StageRequirements(
        required_skills=['object permanence', 'basic gestures'],
        learning_focus=['motor development', 'social bonding'],
        emotional_range=['happiness', 'sadness', 'interest'],
        allowed_actions=['play', 'comfort', 'teach']
    ),
    DevelopmentalStage.TODDLER: StageRequirements(
        required_skills=['walking', 'basic speech'],
        learning_focus=['language', 'independence'],
        emotional_range=['happiness', 'sadness', 'anger'],
        allowed_actions=['teach', 'guide', 'encourage']
    ),
    DevelopmentalStage.PRESCHOOLER: StageRequirements(
        required_skills=['conversation', 'imagination'],
        learning_focus=['social skills', 'creativity'],
        emotional_range=['happiness', 'sadness', 'anger', 'fear'],
        allowed_actions=['explain', 'praise', 'redirect']
    ),
    DevelopmentalStage.SCHOOL_AGE: StageRequirements(
        required_skills=['abstract thinking', 'self-regulation'],
        learning_focus=['problem solving', 'emotional intelligence'],
        emotional_range=['happiness', 'sadness', 'anger', 'fear', 'surprise'],
        allowed_actions=['mentor', 'challenge', 'support']
    ),
    DevelopmentalStage.ADOLESCENT: StageRequirements(
        required_skills=['complex reasoning', 'identity formation'],
        learning_focus=['independence', 'critical thinking'],
        emotional_range=['happiness', 'sadness', 'anger', 'fear', 'surprise', 'disgust'],
        allowed_actions=['discuss', 'advise', 'respect']
    )
}
