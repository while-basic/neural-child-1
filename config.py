import torch
from dataclasses import dataclass
from typing import List, Dict
from developmental_stages import DevelopmentalStage

class Config:
    # Add your configuration parameters here
    def __init__(self):
        self.embedding_dim = 768
        self.max_sequence_length = 512
        self.batch_size = 32
        self.learning_rate = 1e-4
        # Add other configuration parameters as needed

# Create a global config instance
config = Config()

# Server configurations
EMBEDDING_SERVER_URL = "http://localhost:1234"  # Change as needed
CHAT_SERVER_URL = "http://localhost:1234"       # Change as needed

# Model Configuration
EMBEDDING_DIM = 768
HIDDEN_DIM = 512
LEARNING_RATE = 1e-4

# Training Configuration
MAX_ITERATIONS = 100
SAVE_INTERVAL = 10
CHECKPOINT_DIR = "checkpoints"
LOG_DIR = "logs"

# Development Settings
INTERACTIONS_PER_STAGE = 20  # Reduced from 100 to make progression faster
MIN_STAGE_SUCCESS_RATE = 0.5  # Lowered from 0.7 to make progression easier
STAGE_PROGRESSION_THRESHOLD = 0.6  # Lowered from 0.8 to make progression easier

# Memory Configuration
MEMORY_CAPACITY = 10000
REPLAY_BATCH_SIZE = 32
MEMORY_CONSOLIDATION_INTERVAL = 100

# Default Response
DEFAULT_RESPONSE = {
    'text': 'I need a moment to think.',
    'emotional_vector': [0.5, 0.5, 0.5, 0.5]
}

# Default Embedding (zeros vector of EMBEDDING_DIM size)
DEFAULT_EMBEDDING = [0.0] * EMBEDDING_DIM

# Device Configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

@dataclass
class StageRequirements:
    required_skills: List[str]
    learning_focus: List[str]
    emotional_range: List[str]
    allowed_actions: List[str]

# Stage definitions mapping
STAGE_DEFINITIONS = {
    DevelopmentalStage.NEWBORN: StageRequirements(
        required_skills=['basic reflexes', 'crying'],
        learning_focus=['sensory processing', 'basic motor skills'],
        emotional_range=['happiness', 'sadness'],
        allowed_actions=['soothe', 'feed', 'hold']
    ),
    DevelopmentalStage.EARLY_INFANCY: StageRequirements(
        required_skills=['object permanence', 'basic gestures'],
        learning_focus=['motor development', 'social bonding'],
        emotional_range=['happiness', 'sadness', 'interest'],
        allowed_actions=['play', 'comfort', 'teach']
    ),
    DevelopmentalStage.LATE_INFANCY: StageRequirements(
        required_skills=['crawling', 'babbling'],
        learning_focus=['motor skills', 'vocal development'],
        emotional_range=['happiness', 'sadness', 'anger'],
        allowed_actions=['encourage', 'guide', 'protect']
    ),
    DevelopmentalStage.EARLY_TODDLER: StageRequirements(
        required_skills=['walking', 'basic words'],
        learning_focus=['language', 'motor coordination'],
        emotional_range=['happiness', 'sadness', 'anger', 'fear'],
        allowed_actions=['teach', 'guide', 'encourage']
    ),
    DevelopmentalStage.LATE_TODDLER: StageRequirements(
        required_skills=['sentences', 'running'],
        learning_focus=['communication', 'independence'],
        emotional_range=['happiness', 'sadness', 'anger', 'fear'],
        allowed_actions=['explain', 'praise', 'redirect']
    ),
    DevelopmentalStage.EARLY_PRESCHOOL: StageRequirements(
        required_skills=['conversation', 'imagination'],
        learning_focus=['social skills', 'creativity'],
        emotional_range=['happiness', 'sadness', 'anger', 'fear', 'surprise'],
        allowed_actions=['teach', 'explore', 'create']
    ),
    DevelopmentalStage.LATE_PRESCHOOL: StageRequirements(
        required_skills=['complex sentences', 'basic reasoning'],
        learning_focus=['abstract thinking', 'problem solving'],
        emotional_range=['happiness', 'sadness', 'anger', 'fear', 'surprise'],
        allowed_actions=['challenge', 'guide', 'discuss']
    ),
    DevelopmentalStage.EARLY_CHILDHOOD: StageRequirements(
        required_skills=['reading basics', 'logical thinking'],
        learning_focus=['academics', 'social skills'],
        emotional_range=['happiness', 'sadness', 'anger', 'fear', 'surprise', 'disgust'],
        allowed_actions=['teach', 'mentor', 'support']
    ),
    DevelopmentalStage.MIDDLE_CHILDHOOD: StageRequirements(
        required_skills=['abstract thinking', 'self-regulation'],
        learning_focus=['academic skills', 'emotional intelligence'],
        emotional_range=['happiness', 'sadness', 'anger', 'fear', 'surprise', 'disgust'],
        allowed_actions=['guide', 'challenge', 'support']
    ),
    DevelopmentalStage.LATE_CHILDHOOD: StageRequirements(
        required_skills=['complex reasoning', 'social awareness'],
        learning_focus=['critical thinking', 'social development'],
        emotional_range=['happiness', 'sadness', 'anger', 'fear', 'surprise', 'disgust'],
        allowed_actions=['discuss', 'mentor', 'support']
    ),
    DevelopmentalStage.EARLY_ELEMENTARY: StageRequirements(
        required_skills=['research skills', 'project planning'],
        learning_focus=['academic independence', 'collaboration'],
        emotional_range=['all'],
        allowed_actions=['mentor', 'challenge', 'guide']
    ),
    DevelopmentalStage.MIDDLE_ELEMENTARY: StageRequirements(
        required_skills=['analysis', 'teamwork'],
        learning_focus=['complex problem solving', 'leadership'],
        emotional_range=['all'],
        allowed_actions=['coach', 'challenge', 'support']
    ),
    DevelopmentalStage.LATE_ELEMENTARY: StageRequirements(
        required_skills=['advanced reasoning', 'metacognition'],
        learning_focus=['independent research', 'critical analysis'],
        emotional_range=['all'],
        allowed_actions=['mentor', 'discuss', 'guide']
    ),
    DevelopmentalStage.EARLY_ADOLESCENCE: StageRequirements(
        required_skills=['abstract reasoning', 'identity formation'],
        learning_focus=['self-discovery', 'independence'],
        emotional_range=['all'],
        allowed_actions=['support', 'guide', 'respect']
    ),
    DevelopmentalStage.MIDDLE_ADOLESCENCE: StageRequirements(
        required_skills=['complex analysis', 'moral reasoning'],
        learning_focus=['values', 'life skills'],
        emotional_range=['all'],
        allowed_actions=['mentor', 'respect', 'guide']
    ),
    DevelopmentalStage.LATE_ADOLESCENCE: StageRequirements(
        required_skills=['life planning', 'advanced reasoning'],
        learning_focus=['future preparation', 'responsibility'],
        emotional_range=['all'],
        allowed_actions=['advise', 'support', 'respect']
    ),
    DevelopmentalStage.YOUNG_ADULT: StageRequirements(
        required_skills=['independence', 'life management'],
        learning_focus=['adulting', 'responsibility'],
        emotional_range=['all'],
        allowed_actions=['support', 'respect', 'advise']
    ),
    DevelopmentalStage.MATURE_ADULT: StageRequirements(
        required_skills=['wisdom', 'self-actualization'],
        learning_focus=['mentorship', 'legacy'],
        emotional_range=['all'],
        allowed_actions=['respect', 'support', 'learn']
    )
}
