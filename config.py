import torch
from dataclasses import dataclass
from typing import List, Dict, Any
from developmental_stages import DevelopmentalStage  # Only import the enum

# Device configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# LLM configuration
CHAT_SERVER_URL = "http://localhost:1234/v1"
DEFAULT_MODEL = "mistral"

# Model dimensions
embedding_dim = 384  # Standard dimension for text embeddings
hidden_dim = 256
emotion_dim = 4

# Training parameters
learning_rate = 3e-4
weight_decay = 0.01
gradient_clip_norm = 1.0
warmup_steps = 1000
checkpoint_interval = 100
save_interval = 300  # Save every 5 minutes
memory_consolidation_interval = 50
moving_average_window = 50  # Added moving average window size

# Interaction parameters
interactions_per_stage = 100
max_response_tokens = 1000
temperature = 0.7

# Default response when LLM fails
DEFAULT_RESPONSE: Dict[str, Any] = {
    'text': 'I need a moment to think.',
    'emotional_context': {
        'joy': 0.5,
        'trust': 0.5,
        'fear': 0.1,
        'surprise': 0.3
    },
    'reward_score': 0.5,
    'success_metric': 0.5,
    'complexity_rating': 0.3,
    'self_critique_score': 0.5,
    'cognitive_labels': ['basic_interaction'],
    'effectiveness': 0.5
}

# Stage-specific configurations
STAGE_DEFINITIONS = {
    'NEWBORN': {
        'age_range': (0, 3),  # months
        'complexity_range': (0.1, 0.2),
        'emotional_range': (0.6, 0.8),
        'trust_emphasis': 0.8,
        'current_milestones': [
            'Basic emotional responses',
            'Simple reflexes',
            'Recognition of primary caregiver'
        ],
        'upcoming_milestones': [
            'Social smiling',
            'Head control',
            'Cooing sounds'
        ]
    },
    'EARLY_INFANCY': {
        'age_range': (3, 6),
        'complexity_range': (0.2, 0.3),
        'emotional_range': (0.5, 0.7),
        'trust_emphasis': 0.7,
        'current_milestones': [
            'Social smiling',
            'Laughing',
            'Object tracking'
        ],
        'upcoming_milestones': [
            'Sitting without support',
            'Object manipulation',
            'Babbling'
        ]
    }
    # Add more stages as needed
}

# Create a config object that can be imported
class Config:
    def __init__(self):
        for key, value in globals().items():
            if not key.startswith('_'):
                setattr(self, key, value)

config = Config()

# Server configurations
EMBEDDING_SERVER_URL = "http://localhost:1234"  # Change as needed

# Model Configuration
LEARNING_RATE = 1e-4

# Training Configuration
MAX_ITERATIONS = 1000
CHECKPOINT_DIR = "checkpoints"
LOG_DIR = "logs"

# Development Settings
MIN_STAGE_SUCCESS_RATE = 0.4
STAGE_PROGRESSION_THRESHOLD = 0.5

# Memory Configuration
MEMORY_CAPACITY = 50000
REPLAY_BATCH_SIZE = 64

# Default Embedding (zeros vector of embedding_dim size)
DEFAULT_EMBEDDING = [0.0] * embedding_dim

@dataclass
class StageRequirements:
    required_skills: List[str]
    learning_focus: List[str]
    emotional_range: List[str]
    allowed_actions: List[str]
    upcoming_milestones: List[str]
    current_milestones: List[str]

# Add stage definitions with proper enum handling
for stage in DevelopmentalStage:
    if stage not in STAGE_DEFINITIONS:
        if stage == DevelopmentalStage.NEWBORN:
            STAGE_DEFINITIONS[stage] = StageRequirements(
                required_skills=['basic reflexes', 'crying'],
                learning_focus=['sensory processing', 'basic motor skills'],
                emotional_range=['happiness', 'sadness'],
                allowed_actions=['soothe', 'feed', 'hold'],
                upcoming_milestones=[
                    'Recognize mother\'s voice',
                    'Follow moving objects with eyes',
                    'Respond to loud sounds',
                    'Make cooing sounds'
                ],
                current_milestones=[
                    'Basic reflexes',
                    'Can cry to express needs'
                ]
            )
        elif stage == DevelopmentalStage.EARLY_INFANCY:
            STAGE_DEFINITIONS[stage] = StageRequirements(
                required_skills=['object permanence', 'basic gestures'],
                learning_focus=['motor development', 'social bonding'],
                emotional_range=['happiness', 'sadness', 'interest'],
                allowed_actions=['play', 'comfort', 'teach'],
                upcoming_milestones=[
                    'Smile at familiar faces',
                    'Reach for objects',
                    'Hold head steady',
                    'Make babbling sounds'
                ],
                current_milestones=[
                    'Recognize mother\'s voice',
                    'Track moving objects',
                    'Respond to sounds'
                ]
            )
        elif stage == DevelopmentalStage.LATE_INFANCY:
            STAGE_DEFINITIONS[stage] = StageRequirements(
                required_skills=['crawling', 'babbling'],
                learning_focus=['motor skills', 'vocal development'],
                emotional_range=['happiness', 'sadness', 'anger'],
                allowed_actions=['encourage', 'guide', 'protect'],
                upcoming_milestones=[
                    'First words',
                    'Pull to stand',
                    'Wave bye-bye',
                    'Play simple games'
                ],
                current_milestones=[
                    'Babbling with intent',
                    'Crawling',
                    'Object permanence'
                ]
            )
        elif stage == DevelopmentalStage.EARLY_TODDLER:
            STAGE_DEFINITIONS[stage] = StageRequirements(
                required_skills=['walking', 'basic words'],
                learning_focus=['language', 'motor coordination'],
                emotional_range=['happiness', 'sadness', 'anger', 'fear'],
                allowed_actions=['teach', 'guide', 'encourage'],
                upcoming_milestones=[
                    'Two-word phrases',
                    'Run steadily',
                    'Follow simple commands',
                    'Use spoon/fork'
                ],
                current_milestones=[
                    'Walking independently',
                    'Several clear words',
                    'Point to objects'
                ]
            )
        elif stage == DevelopmentalStage.LATE_TODDLER:
            STAGE_DEFINITIONS[stage] = StageRequirements(
                required_skills=['sentences', 'running'],
                learning_focus=['communication', 'independence'],
                emotional_range=['happiness', 'sadness', 'anger', 'fear', 'excitement'],
                allowed_actions=['teach', 'guide', 'discipline'],
                upcoming_milestones=[
                    'Complete sentences',
                    'Toilet training',
                    'Share toys',
                    'Draw shapes'
                ],
                current_milestones=[
                    'Two-word combinations',
                    'Running and climbing',
                    'Basic self-help skills'
                ]
            )
        elif stage == DevelopmentalStage.EARLY_PRESCHOOL:
            STAGE_DEFINITIONS[stage] = StageRequirements(
                required_skills=['conversation', 'imagination'],
                learning_focus=['social skills', 'creativity'],
                emotional_range=['happiness', 'sadness', 'anger', 'fear', 'excitement', 'curiosity'],
                allowed_actions=['teach', 'guide', 'encourage', 'explain'],
                upcoming_milestones=[
                    'Complex sentences',
                    'Imaginative play',
                    'Count to 10',
                    'Name colors'
                ],
                current_milestones=[
                    'Basic conversations',
                    'Follow two-step instructions',
                    'Take turns in games'
                ]
            )
        elif stage == DevelopmentalStage.LATE_PRESCHOOL:
            STAGE_DEFINITIONS[stage] = StageRequirements(
                required_skills=['complex sentences', 'basic reasoning'],
                learning_focus=['abstract thinking', 'problem solving'],
                emotional_range=['happiness', 'sadness', 'anger', 'fear', 'surprise'],
                allowed_actions=['challenge', 'guide', 'discuss'],
                upcoming_milestones=[
                    'Read simple words',
                    'Write letters',
                    'Basic addition',
                    'Complex problem-solving'
                ],
                current_milestones=[
                    'Complex sentences',
                    'Imaginative play',
                    'Basic counting',
                    'Color recognition'
                ]
            )
        elif stage == DevelopmentalStage.EARLY_CHILDHOOD:
            STAGE_DEFINITIONS[stage] = StageRequirements(
                required_skills=['reading basics', 'logical thinking'],
                learning_focus=['academics', 'social skills'],
                emotional_range=['happiness', 'sadness', 'anger', 'fear', 'surprise', 'disgust'],
                allowed_actions=['teach', 'mentor', 'support'],
                upcoming_milestones=[
                    'Read simple books',
                    'Basic writing',
                    'Simple math',
                    'Group play'
                ],
                current_milestones=[
                    'Letter recognition',
                    'Number concepts',
                    'Basic reasoning'
                ]
            )
        elif stage == DevelopmentalStage.MIDDLE_CHILDHOOD:
            STAGE_DEFINITIONS[stage] = StageRequirements(
                required_skills=['abstract thinking', 'self-regulation'],
                learning_focus=['academic skills', 'emotional intelligence'],
                emotional_range=['happiness', 'sadness', 'anger', 'fear', 'surprise', 'disgust'],
                allowed_actions=['guide', 'challenge', 'support'],
                upcoming_milestones=[
                    'Complex reading',
                    'Written expression',
                    'Mathematical operations',
                    'Social awareness'
                ],
                current_milestones=[
                    'Reading comprehension',
                    'Basic writing',
                    'Addition/Subtraction'
                ]
            )
        elif stage == DevelopmentalStage.LATE_CHILDHOOD:
            STAGE_DEFINITIONS[stage] = StageRequirements(
                required_skills=['complex reasoning', 'social awareness'],
                learning_focus=['critical thinking', 'social development'],
                emotional_range=['happiness', 'sadness', 'anger', 'fear', 'surprise', 'disgust'],
                allowed_actions=['discuss', 'mentor', 'support'],
                upcoming_milestones=[
                    'Abstract reasoning',
                    'Complex writing',
                    'Advanced math',
                    'Emotional intelligence'
                ],
                current_milestones=[
                    'Critical thinking',
                    'Social skills',
                    'Problem-solving'
                ]
            )
        elif stage == DevelopmentalStage.EARLY_ELEMENTARY:
            STAGE_DEFINITIONS[stage] = StageRequirements(
                required_skills=['research skills', 'project planning'],
                learning_focus=['academic independence', 'collaboration'],
                emotional_range=['all'],
                allowed_actions=['mentor', 'challenge', 'guide'],
                upcoming_milestones=[
                    'Independent research',
                    'Project management',
                    'Team leadership',
                    'Scientific method'
                ],
                current_milestones=[
                    'Basic research',
                    'Collaboration',
                    'Time management'
                ]
            )
        elif stage == DevelopmentalStage.MIDDLE_ELEMENTARY:
            STAGE_DEFINITIONS[stage] = StageRequirements(
                required_skills=['analysis', 'teamwork'],
                learning_focus=['complex problem solving', 'leadership'],
                emotional_range=['all'],
                allowed_actions=['coach', 'challenge', 'support'],
                upcoming_milestones=[
                    'Advanced analysis',
                    'Leadership skills',
                    'Complex projects',
                    'Abstract concepts'
                ],
                current_milestones=[
                    'Problem analysis',
                    'Team coordination',
                    'Project completion'
                ]
            )
        elif stage == DevelopmentalStage.LATE_ELEMENTARY:
            STAGE_DEFINITIONS[stage] = StageRequirements(
                required_skills=['advanced reasoning', 'metacognition'],
                learning_focus=['independent research', 'critical analysis'],
                emotional_range=['all'],
                allowed_actions=['mentor', 'discuss', 'guide'],
                upcoming_milestones=[
                    'Research methodology',
                    'Critical analysis',
                    'Advanced reasoning',
                    'Complex problem-solving'
                ],
                current_milestones=[
                    'Independent study',
                    'Analytical thinking',
                    'Research skills'
                ]
            )
        elif stage == DevelopmentalStage.EARLY_ADOLESCENCE:
            STAGE_DEFINITIONS[stage] = StageRequirements(
                required_skills=['abstract reasoning', 'identity formation'],
                learning_focus=['self-discovery', 'independence'],
                emotional_range=['all'],
                allowed_actions=['support', 'guide', 'respect'],
                upcoming_milestones=[
                    'Personal identity',
                    'Emotional regulation',
                    'Social navigation',
                    'Abstract thinking'
                ],
                current_milestones=[
                    'Self-awareness',
                    'Basic identity',
                    'Emotional understanding'
                ]
            )
        elif stage == DevelopmentalStage.MIDDLE_ADOLESCENCE:
            STAGE_DEFINITIONS[stage] = StageRequirements(
                required_skills=['complex analysis', 'moral reasoning'],
                learning_focus=['values', 'life skills'],
                emotional_range=['all'],
                allowed_actions=['mentor', 'respect', 'guide'],
                upcoming_milestones=[
                    'Value system',
                    'Career planning',
                    'Life goals',
                    'Social responsibility'
                ],
                current_milestones=[
                    'Moral reasoning',
                    'Personal values',
                    'Future planning'
                ]
            )
        elif stage == DevelopmentalStage.LATE_ADOLESCENCE:
            STAGE_DEFINITIONS[stage] = StageRequirements(
                required_skills=['life planning', 'advanced reasoning'],
                learning_focus=['future preparation', 'responsibility'],
                emotional_range=['all'],
                allowed_actions=['advise', 'support', 'respect'],
                upcoming_milestones=[
                    'Career path',
                    'Life independence',
                    'Relationship building',
                    'Financial planning'
                ],
                current_milestones=[
                    'Goal setting',
                    'Decision making',
                    'Responsibility'
                ]
            )
        elif stage == DevelopmentalStage.YOUNG_ADULT:
            STAGE_DEFINITIONS[stage] = StageRequirements(
                required_skills=['independence', 'life management'],
                learning_focus=['adulting', 'responsibility'],
                emotional_range=['all'],
                allowed_actions=['support', 'respect', 'advise'],
                upcoming_milestones=[
                    'Career establishment',
                    'Long-term planning',
                    'Relationship mastery',
                    'Financial independence'
                ],
                current_milestones=[
                    'Basic independence',
                    'Career start',
                    'Life management'
                ]
            )
        elif stage == DevelopmentalStage.MATURE_ADULT:
            STAGE_DEFINITIONS[stage] = StageRequirements(
                required_skills=['wisdom', 'self-actualization'],
                learning_focus=['mentorship', 'legacy'],
                emotional_range=['all'],
                allowed_actions=['respect', 'support', 'learn'],
                upcoming_milestones=[
                    'Wisdom sharing',
                    'Legacy building',
                    'Community impact',
                    'Personal fulfillment'
                ],
                current_milestones=[
                    'Life wisdom',
                    'Self-actualization',
                    'Mentorship ability'
                ]
            )
