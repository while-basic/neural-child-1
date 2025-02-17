"""Developmental stages module.

This module defines the various developmental stages and related functionality
for the Neural Child project.
"""

from enum import Enum, auto
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

# Constants for stage progression
MIN_STAGE_SUCCESS_RATE = 0.4
STAGE_PROGRESSION_THRESHOLD = 0.5

@dataclass
class StageCharacteristics:
    """Characteristics of a developmental stage."""
    age_range: Tuple[int, int]
    complexity_range: Tuple[float, float]
    emotional_range: Tuple[float, float]
    required_skills: List[str]
    learning_objectives: List[str]
    success_criteria: Dict[str, float]

class DevelopmentalStage(Enum):
    """Enumeration of developmental stages from newborn to mature adult."""
    # Early Development (0-2 years)
    NEWBORN = auto()  # 0-3 months
    EARLY_INFANCY = auto()  # 3-6 months
    LATE_INFANCY = auto()  # 6-12 months
    EARLY_TODDLER = auto()  # 12-18 months
    LATE_TODDLER = auto()  # 18-24 months
    
    # Preschool Period (2-4 years)
    EARLY_PRESCHOOL = auto()  # 2-3 years
    LATE_PRESCHOOL = auto()  # 3-4 years
    
    # Early Childhood (4-7 years)
    EARLY_CHILDHOOD = auto()  # 4-5 years
    MIDDLE_CHILDHOOD = auto()  # 5-6 years
    LATE_CHILDHOOD = auto()  # 6-7 years
    
    # Elementary Period (7-11 years)
    EARLY_ELEMENTARY = auto()  # 7-8 years
    MIDDLE_ELEMENTARY = auto()  # 8-9 years
    LATE_ELEMENTARY = auto()  # 9-11 years
    
    # Adolescence (11-18 years)
    EARLY_ADOLESCENCE = auto()  # 11-13 years
    MIDDLE_ADOLESCENCE = auto()  # 13-15 years
    LATE_ADOLESCENCE = auto()  # 15-18 years
    
    # Adulthood (18+ years)
    YOUNG_ADULT = auto()  # 18-21 years
    MATURE_ADULT = auto()  # 21+ years

# Stage definitions with comprehensive developmental characteristics
STAGE_DEFINITIONS = {
    DevelopmentalStage.NEWBORN: {
        'age_range': (0, 3),  # months
        'complexity_range': (0.1, 0.2),
        'emotional_range': (0.6, 0.8),
        'required_skills': ['basic reflexes', 'crying', 'sucking'],
        'learning_focus': ['sensory processing', 'basic motor skills'],
        'current_milestones': ['Basic reflexes', 'Can cry to express needs'],
        'upcoming_milestones': [
            'Recognize mother\'s voice',
            'Follow moving objects with eyes',
            'Respond to loud sounds',
            'Make cooing sounds'
        ],
        'trust_emphasis': 0.8,
        'behaviors': [
            'Crying to express needs',
            'Basic reflexes',
            'Sleep-wake cycles',
            'Feeding responses',
            'Startle response',
            'Rooting reflex',
            'Grasping reflex',
            'Visual tracking'
        ]
    },
    DevelopmentalStage.EARLY_INFANCY: {
        'age_range': (3, 6),
        'complexity_range': (0.2, 0.3),
        'emotional_range': (0.5, 0.7),
        'required_skills': ['object permanence', 'basic gestures', 'cooing'],
        'learning_focus': ['motor development', 'social bonding'],
        'current_milestones': ['Social smiling', 'Track moving objects', 'Respond to sounds'],
        'upcoming_milestones': [
            'Sitting without support',
            'Object manipulation',
            'Babbling',
            'Recognize familiar faces'
        ],
        'trust_emphasis': 0.7,
        'behaviors': [
            'Social smiling',
            'Cooing and babbling',
            'Following objects with eyes',
            'Responding to sounds',
            'Head control',
            'Reaching for objects',
            'Recognizing familiar faces',
            'Laughing'
        ]
    },
    DevelopmentalStage.LATE_INFANCY: {
        'age_range': (6, 12),
        'complexity_range': (0.3, 0.4),
        'emotional_range': (0.4, 0.6),
        'required_skills': ['crawling', 'babbling', 'object manipulation'],
        'learning_focus': ['mobility', 'communication'],
        'current_milestones': ['Crawling', 'Object permanence', 'Babbling'],
        'upcoming_milestones': [
            'First words',
            'Standing with support',
            'Pincer grasp',
            'Wave bye-bye'
        ],
        'trust_emphasis': 0.6,
        'behaviors': [
            'Crawling',
            'Pulling to stand',
            'Object manipulation',
            'Babbling with intent',
            'Stranger anxiety',
            'Separation anxiety',
            'Imitation of actions',
            'Simple gestures'
        ]
    },
    DevelopmentalStage.EARLY_TODDLER: {
        'age_range': (12, 18),
        'complexity_range': (0.4, 0.5),
        'emotional_range': (0.3, 0.5),
        'required_skills': ['walking', 'first words', 'feeding self'],
        'learning_focus': ['language', 'independence'],
        'current_milestones': ['Walking independently', 'Several clear words', 'Self-feeding'],
        'upcoming_milestones': [
            'Two-word phrases',
            'Run steadily',
            'Stack blocks',
            'Use spoon'
        ],
        'trust_emphasis': 0.5,
        'behaviors': [
            'Walking independently',
            'Using single words',
            'Following simple commands',
            'Pointing to objects',
            'Simple problem solving',
            'Exploring environment',
            'Showing preferences',
            'Asserting independence'
        ]
    },
    DevelopmentalStage.LATE_TODDLER: {
        'age_range': (18, 36),
        'complexity_range': (0.5, 0.6),
        'emotional_range': (0.2, 0.4),
        'required_skills': ['running', 'short sentences', 'symbolic play'],
        'learning_focus': ['social skills', 'imagination'],
        'current_milestones': ['Running', 'Short sentences', 'Imaginative play'],
        'upcoming_milestones': [
            'Full sentences',
            'Toilet training',
            'Share toys',
            'Draw shapes'
        ],
        'trust_emphasis': 0.5,
        'behaviors': [
            'Running and climbing',
            'Using short phrases',
            'Imaginative play',
            'Toilet training',
            'Emotional expression',
            'Simple turn-taking',
            'Basic self-help skills',
            'Following routines'
        ]
    },
    DevelopmentalStage.EARLY_PRESCHOOL: {
        'age_range': (36, 48),
        'complexity_range': (0.6, 0.7),
        'emotional_range': (0.2, 0.4),
        'required_skills': ['conversation', 'drawing', 'group play'],
        'learning_focus': ['creativity', 'social interaction'],
        'current_milestones': ['Basic conversations', 'Draw simple shapes', 'Group play'],
        'upcoming_milestones': [
            'Complex sentences',
            'Count to 20',
            'Name colors',
            'Ride tricycle'
        ],
        'trust_emphasis': 0.4,
        'behaviors': [
            'Speaking in sentences',
            'Following complex instructions',
            'Engaging in pretend play',
            'Basic problem solving',
            'Sharing with others',
            'Drawing with purpose',
            'Showing empathy',
            'Group participation'
        ]
    },
    DevelopmentalStage.LATE_PRESCHOOL: {
        'age_range': (48, 60),
        'complexity_range': (0.7, 0.8),
        'emotional_range': (0.2, 0.4),
        'required_skills': ['storytelling', 'basic math', 'rule following'],
        'learning_focus': ['pre-academic skills', 'rule understanding'],
        'current_milestones': ['Tell stories', 'Count to 20', 'Follow rules'],
        'upcoming_milestones': [
            'Write letters',
            'Simple addition',
            'Read sight words',
            'Complex problem-solving'
        ],
        'trust_emphasis': 0.4,
        'behaviors': [
            'Complex conversations',
            'Early reading skills',
            'Basic math concepts',
            'Advanced social play',
            'Rule following',
            'Creative expression',
            'Emotional regulation',
            'Cooperative play'
        ]
    },
    DevelopmentalStage.EARLY_CHILDHOOD: {
        'age_range': (60, 72),
        'complexity_range': (0.8, 0.85),
        'emotional_range': (0.2, 0.4),
        'required_skills': ['reading basics', 'writing', 'team play'],
        'learning_focus': ['academic foundations', 'teamwork'],
        'current_milestones': ['Read simple words', 'Write name', 'Team games'],
        'upcoming_milestones': [
            'Read sentences',
            'Basic math operations',
            'Scientific thinking',
            'Complex social play'
        ],
        'trust_emphasis': 0.4,
        'behaviors': [
            'Reading simple words',
            'Writing letters and numbers',
            'Basic addition and subtraction',
            'Team participation',
            'Following multi-step instructions',
            'Independent problem solving',
            'Peer collaboration',
            'Self-regulation'
        ]
    },
    DevelopmentalStage.MIDDLE_CHILDHOOD: {
        'age_range': (72, 96),
        'complexity_range': (0.85, 0.9),
        'emotional_range': (0.2, 0.4),
        'required_skills': ['reading comprehension', 'mathematical operations', 'social navigation'],
        'learning_focus': ['academic skills', 'social competence'],
        'current_milestones': ['Read fluently', 'Basic math', 'Complex friendships'],
        'upcoming_milestones': [
            'Abstract thinking',
            'Advanced math',
            'Independent research',
            'Leadership skills'
        ],
        'trust_emphasis': 0.3,
        'behaviors': [
            'Reading comprehension',
            'Mathematical reasoning',
            'Scientific inquiry',
            'Complex social interactions',
            'Team leadership',
            'Project planning',
            'Critical thinking',
            'Emotional understanding'
        ]
    },
    DevelopmentalStage.LATE_CHILDHOOD: {
        'age_range': (96, 120),
        'complexity_range': (0.9, 0.95),
        'emotional_range': (0.2, 0.4),
        'required_skills': ['abstract thinking', 'complex problem solving', 'emotional regulation'],
        'learning_focus': ['critical thinking', 'emotional intelligence'],
        'current_milestones': ['Abstract reasoning', 'Complex math', 'Emotional awareness'],
        'upcoming_milestones': [
            'Scientific method',
            'Advanced writing',
            'Complex analysis',
            'Peer leadership'
        ],
        'trust_emphasis': 0.3,
        'behaviors': [
            'Abstract reasoning',
            'Complex problem solving',
            'Advanced social skills',
            'Independent research',
            'Leadership roles',
            'Emotional intelligence',
            'Moral reasoning',
            'Self-reflection'
        ]
    },
    DevelopmentalStage.EARLY_ELEMENTARY: {
        'age_range': (72, 96),
        'complexity_range': (0.8, 0.85),
        'emotional_range': (0.2, 0.4),
        'required_skills': ['basic literacy', 'numeracy', 'classroom behavior'],
        'learning_focus': ['foundational academics', 'classroom skills'],
        'current_milestones': ['Read and write', 'Basic math', 'Follow instructions'],
        'upcoming_milestones': [
            'Reading comprehension',
            'Math operations',
            'Scientific inquiry',
            'Writing composition'
        ],
        'trust_emphasis': 0.3,
        'behaviors': [
            'Reading and writing',
            'Basic mathematics',
            'Following classroom rules',
            'Group participation',
            'Task completion',
            'Organization skills',
            'Peer interaction',
            'Basic research'
        ]
    },
    DevelopmentalStage.MIDDLE_ELEMENTARY: {
        'age_range': (96, 120),
        'complexity_range': (0.85, 0.9),
        'emotional_range': (0.2, 0.4),
        'required_skills': ['reading comprehension', 'mathematical thinking', 'research skills'],
        'learning_focus': ['academic depth', 'independent learning'],
        'current_milestones': ['Comprehend text', 'Problem solving', 'Basic research'],
        'upcoming_milestones': [
            'Critical analysis',
            'Advanced math concepts',
            'Independent projects',
            'Presentation skills'
        ],
        'trust_emphasis': 0.3,
        'behaviors': [
            'Text comprehension',
            'Mathematical problem solving',
            'Scientific method',
            'Project management',
            'Presentation skills',
            'Team collaboration',
            'Independent study',
            'Critical analysis'
        ]
    },
    DevelopmentalStage.LATE_ELEMENTARY: {
        'age_range': (120, 144),
        'complexity_range': (0.9, 0.95),
        'emotional_range': (0.2, 0.4),
        'required_skills': ['critical thinking', 'advanced math', 'project management'],
        'learning_focus': ['analytical skills', 'project-based learning'],
        'current_milestones': ['Critical thinking', 'Complex math', 'Project planning'],
        'upcoming_milestones': [
            'Research methods',
            'Advanced analysis',
            'Leadership roles',
            'Complex projects'
        ],
        'trust_emphasis': 0.3,
        'behaviors': [
            'Advanced research',
            'Complex analysis',
            'Project leadership',
            'Abstract reasoning',
            'Advanced writing',
            'Peer mentoring',
            'Self-directed learning',
            'Time management'
        ]
    },
    DevelopmentalStage.EARLY_ADOLESCENCE: {
        'age_range': (144, 168),
        'complexity_range': (0.95, 0.97),
        'emotional_range': (0.2, 0.4),
        'required_skills': ['abstract reasoning', 'emotional awareness', 'social navigation'],
        'learning_focus': ['identity development', 'social understanding'],
        'current_milestones': ['Abstract thinking', 'Emotional insight', 'Social skills'],
        'upcoming_milestones': [
            'Personal identity',
            'Career exploration',
            'Moral reasoning',
            'Complex relationships'
        ],
        'trust_emphasis': 0.3,
        'behaviors': [
            'Identity exploration',
            'Abstract thinking',
            'Emotional development',
            'Peer relationships',
            'Self-awareness',
            'Social navigation',
            'Value formation',
            'Independence seeking'
        ]
    },
    DevelopmentalStage.MIDDLE_ADOLESCENCE: {
        'age_range': (168, 192),
        'complexity_range': (0.97, 0.98),
        'emotional_range': (0.2, 0.4),
        'required_skills': ['self-reflection', 'goal setting', 'interpersonal skills'],
        'learning_focus': ['future planning', 'personal values'],
        'current_milestones': ['Self-awareness', 'Goal planning', 'Relationship building'],
        'upcoming_milestones': [
            'Career direction',
            'Value system',
            'Independence',
            'Leadership roles'
        ],
        'trust_emphasis': 0.3,
        'behaviors': [
            'Future planning',
            'Career exploration',
            'Complex relationships',
            'Value development',
            'Self-identity',
            'Social leadership',
            'Emotional maturity',
            'Goal setting'
        ]
    },
    DevelopmentalStage.LATE_ADOLESCENCE: {
        'age_range': (192, 216),
        'complexity_range': (0.98, 0.99),
        'emotional_range': (0.2, 0.4),
        'required_skills': ['independence', 'decision making', 'life planning'],
        'learning_focus': ['autonomy', 'life skills'],
        'current_milestones': ['Independent decisions', 'Life planning', 'Complex relationships'],
        'upcoming_milestones': [
            'Career preparation',
            'Independent living',
            'Adult relationships',
            'Financial planning'
        ],
        'trust_emphasis': 0.3,
        'behaviors': [
            'Independent decision making',
            'Life planning',
            'Career preparation',
            'Relationship maturity',
            'Financial responsibility',
            'Personal autonomy',
            'Social responsibility',
            'Future orientation'
        ]
    },
    DevelopmentalStage.YOUNG_ADULT: {
        'age_range': (216, 360),
        'complexity_range': (0.99, 1.0),
        'emotional_range': (0.2, 0.4),
        'required_skills': ['career development', 'relationship building', 'financial management'],
        'learning_focus': ['professional growth', 'life management'],
        'current_milestones': ['Career progress', 'Stable relationships', 'Financial independence'],
        'upcoming_milestones': [
            'Career advancement',
            'Long-term planning',
            'Complex problem solving',
            'Leadership development'
        ],
        'trust_emphasis': 0.3,
        'behaviors': [
            'Career development',
            'Relationship building',
            'Financial planning',
            'Life management',
            'Professional networking',
            'Personal growth',
            'Community involvement',
            'Leadership development'
        ]
    },
    DevelopmentalStage.MATURE_ADULT: {
        'age_range': (360, 600),
        'complexity_range': (1.0, 1.0),
        'emotional_range': (0.2, 0.4),
        'required_skills': ['wisdom', 'mentorship', 'life mastery'],
        'learning_focus': ['wisdom development', 'legacy building'],
        'current_milestones': ['Life wisdom', 'Mentoring others', 'Complex understanding'],
        'upcoming_milestones': [
            'Legacy creation',
            'Knowledge transfer',
            'Societal contribution',
            'Continued growth'
        ],
        'trust_emphasis': 0.3,
        'behaviors': [
            'Wisdom sharing',
            'Mentoring others',
            'Legacy building',
            'Societal contribution',
            'Lifelong learning',
            'Complex problem solving',
            'Emotional mastery',
            'Value transmission'
        ]
    }
}

class DevelopmentalSystem:
    """System for managing developmental progression and milestones."""
    
    def __init__(self) -> None:
        """Initialize the developmental system."""
        self.current_stage = DevelopmentalStage.NEWBORN
        self.milestones: Dict[DevelopmentalStage, Dict[str, Any]] = {}
        self._initialize_milestones()
    
    def _initialize_milestones(self) -> None:
        """Initialize the developmental milestones for each stage."""
        logger.info("Initializing developmental milestones")
        for stage in DevelopmentalStage:
            if stage in STAGE_DEFINITIONS:
                self.milestones[stage] = {
                    'current': STAGE_DEFINITIONS[stage]['current_milestones'],
                    'upcoming': STAGE_DEFINITIONS[stage]['upcoming_milestones'],
                    'completed': []
                }
    
    def get_stage_requirements(self) -> Dict[str, Any]:
        """Get detailed requirements and behaviors for the current stage.
        
        Returns:
            Dict containing stage requirements including:
            - stage: Current developmental stage
            - age_range: Tuple of (min_age, max_age)
            - required_skills: List of required skills
            - learning_focus: List of learning objectives
            - emotional_range: Tuple of (min_emotional, max_emotional)
            - behaviors: List of expected behaviors
            - current_milestones: List of current milestones
            - upcoming_milestones: List of upcoming milestones
            - complexity_range: Tuple of (min_complexity, max_complexity)
            - trust_emphasis: Float indicating trust importance
        
        Raises:
            ValueError: If no definition is found for the current stage
        """
        stage_def = STAGE_DEFINITIONS.get(self.current_stage)
        if not stage_def:
            raise ValueError(f"No definition found for stage {self.current_stage}")
            
        return {
            'stage': self.current_stage,
            'age_range': stage_def['age_range'],
            'required_skills': stage_def['required_skills'],
            'learning_focus': stage_def.get('learning_focus', []),
            'emotional_range': stage_def.get('emotional_range', (0.0, 1.0)),
            'behaviors': stage_def.get('behaviors', []),
            'current_milestones': stage_def.get('current_milestones', []),
            'upcoming_milestones': stage_def.get('upcoming_milestones', []),
            'complexity_range': stage_def.get('complexity_range', (0.1, 0.5)),
            'trust_emphasis': stage_def.get('trust_emphasis', 0.5)
        }
    
    def check_stage_progression(self, metrics: Dict[str, float]) -> bool:
        """Check if the current metrics indicate stage progression.
        
        Args:
            metrics: Dictionary of developmental metrics and their values
            
        Returns:
            bool: True if ready to progress to next stage, False otherwise
        """
        stage_def = STAGE_DEFINITIONS.get(self.current_stage)
        if not stage_def:
            return False
            
        # Check if metrics meet minimum requirements
        min_success_rate = MIN_STAGE_SUCCESS_RATE
        progression_threshold = STAGE_PROGRESSION_THRESHOLD
        
        success_rate = metrics.get('success_rate', 0.0)
        emotional_regulation = metrics.get('emotional_regulation', 0.0)
        social_skills = metrics.get('social_skills', 0.0)
        cognitive_development = metrics.get('cognitive_development', 0.0)
        
        # Calculate overall development score
        development_score = (
            success_rate * 0.3 +
            emotional_regulation * 0.2 +
            social_skills * 0.2 +
            cognitive_development * 0.3
        )
        
        return (
            success_rate >= min_success_rate and
            development_score >= progression_threshold
        )
    
    def progress_stage(self) -> None:
        """Progress to the next developmental stage if possible."""
        current_index = list(DevelopmentalStage).index(self.current_stage)
        if current_index < len(DevelopmentalStage) - 1:
            self.current_stage = list(DevelopmentalStage)[current_index + 1]
            logger.info(f"Progressed to stage: {self.current_stage.name}")
        else:
            logger.info("Already at maximum developmental stage") 