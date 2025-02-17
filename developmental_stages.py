from enum import Enum, auto
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

# Constants for stage progression
MIN_STAGE_SUCCESS_RATE = 0.4
STAGE_PROGRESSION_THRESHOLD = 0.5

class DevelopmentalStage(Enum):
    NEWBORN = 0
    EARLY_INFANCY = auto()
    LATE_INFANCY = auto()
    EARLY_TODDLER = auto()
    LATE_TODDLER = auto()
    EARLY_PRESCHOOL = auto()
    LATE_PRESCHOOL = auto()
    EARLY_CHILDHOOD = auto()
    MIDDLE_CHILDHOOD = auto()
    LATE_CHILDHOOD = auto()
    EARLY_ELEMENTARY = auto()
    MIDDLE_ELEMENTARY = auto()
    LATE_ELEMENTARY = auto()
    EARLY_ADOLESCENCE = auto()
    MIDDLE_ADOLESCENCE = auto()
    LATE_ADOLESCENCE = auto()
    YOUNG_ADULT = auto()
    MATURE_ADULT = auto()

@dataclass
class StageCharacteristics:
    age_range: tuple
    complexity_range: tuple
    emotional_range: tuple
    required_skills: List[str]
    learning_objectives: List[str]
    success_criteria: Dict[str, float]

# Stage definitions moved from config.py
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
        'trust_emphasis': 0.8
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
        'trust_emphasis': 0.7
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
        'trust_emphasis': 0.6
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
        'trust_emphasis': 0.5
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
        'trust_emphasis': 0.5
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
        'trust_emphasis': 0.4
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
        'trust_emphasis': 0.4
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
        'trust_emphasis': 0.4
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
        'trust_emphasis': 0.3
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
        'trust_emphasis': 0.3
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
        'trust_emphasis': 0.3
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
        'trust_emphasis': 0.3
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
        'trust_emphasis': 0.3
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
        'trust_emphasis': 0.3
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
        'trust_emphasis': 0.3
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
        'trust_emphasis': 0.3
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
        'trust_emphasis': 0.3
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
        'trust_emphasis': 0.3
    }
        }

class DevelopmentalSystem:
    def __init__(self):
        self.current_stage = DevelopmentalStage.NEWBORN
        self.stage_metrics = {
            'success_rate': 0.0,
            'abstraction': 0.0,
            'self_awareness': 0.0,
            'emotional_regulation': 0.0,
            'social_skills': 0.0,
            'cognitive_development': 0.0
        }
        self.consecutive_successes = 0
        self.stage_duration = 0
        self.stage_history = []
        
        # Add detailed skill tracking
        self.skill_progress = {
            'voice_recognition': 0.0,
            'visual_tracking': 0.0,
            'sound_response': 0.0,
            'vocalization': 0.0,
            'object_permanence': 0.0,
            'social_bonding': 0.0,
            'motor_skills': 0.0,
            'emotional_expression': 0.0
        }
        
        # Add interaction tracking
        self.interaction_counts = {
            'mother_voice': 0,
            'visual_stimuli': 0,
            'sound_stimuli': 0,
            'social_engagement': 0,
            'motor_practice': 0
        }
    
    def get_stage_characteristics(self) -> StageCharacteristics:
        """Get the characteristics of the current developmental stage"""
        stage_def = STAGE_DEFINITIONS.get(self.current_stage)
        if not stage_def:
            raise ValueError(f"No definition found for stage {self.current_stage}")
            
        return StageCharacteristics(
            age_range=stage_def['age_range'],
            complexity_range=stage_def['complexity_range'],
            emotional_range=stage_def.get('emotional_range', (0.0, 1.0)),
            required_skills=stage_def['required_skills'],
            learning_objectives=stage_def.get('learning_focus', []),
            success_criteria={
                'min_success_rate': MIN_STAGE_SUCCESS_RATE,
                'progression_threshold': STAGE_PROGRESSION_THRESHOLD
            }
        )
    
    def get_stage_requirements(self) -> Dict[str, Any]:
        """Get detailed requirements and behaviors for the current stage"""
        stage_def = STAGE_DEFINITIONS.get(self.current_stage)
        if not stage_def:
            raise ValueError(f"No definition found for stage {self.current_stage}")
            
        # Get stage-specific behaviors based on the stage
        behaviors = {
            DevelopmentalStage.NEWBORN: [
                'Crying to express needs',
                'Basic reflexes',
                'Sleep-wake cycles',
                'Feeding responses'
            ],
            DevelopmentalStage.EARLY_INFANCY: [
                'Social smiling',
                'Cooing and babbling',
                'Following objects with eyes',
                'Responding to sounds'
            ],
            DevelopmentalStage.LATE_INFANCY: [
                'Reaching for objects',
                'Making distinct sounds',
                'Showing emotions',
                'Recognizing familiar faces'
            ],
            DevelopmentalStage.EARLY_TODDLER: [
                'Walking with support',
                'Using single words',
                'Following simple commands',
                'Exploring objects'
            ],
            DevelopmentalStage.LATE_TODDLER: [
                'Running and climbing',
                'Using short phrases',
                'Beginning imaginative play',
                'Showing independence'
            ],
            DevelopmentalStage.EARLY_PRESCHOOL: [
                'Speaking in sentences',
                'Following complex instructions',
                'Engaging in pretend play',
                'Basic problem solving'
            ],
            DevelopmentalStage.LATE_PRESCHOOL: [
                'Complex conversations',
                'Early reading skills',
                'Basic math concepts',
                'Advanced social play'
            ],
            DevelopmentalStage.EARLY_CHILDHOOD: [
                'Reading simple words',
                'Writing letters',
                'Basic addition',
                'Group activities'
            ],
            DevelopmentalStage.MIDDLE_CHILDHOOD: [
                'Reading comprehension',
                'Mathematical operations',
                'Scientific thinking',
                'Team collaboration'
            ],
            DevelopmentalStage.LATE_CHILDHOOD: [
                'Abstract reasoning',
                'Complex problem solving',
                'Critical thinking',
                'Social awareness'
            ],
            DevelopmentalStage.EARLY_ELEMENTARY: [
                'Basic academic skills',
                'Classroom participation',
                'Following schedules',
                'Peer relationships'
            ],
            DevelopmentalStage.MIDDLE_ELEMENTARY: [
                'Independent learning',
                'Research skills',
                'Project planning',
                'Leadership emergence'
            ],
            DevelopmentalStage.LATE_ELEMENTARY: [
                'Advanced academics',
                'Critical analysis',
                'Complex projects',
                'Team leadership'
            ],
            DevelopmentalStage.EARLY_ADOLESCENCE: [
                'Identity exploration',
                'Abstract thinking',
                'Emotional development',
                'Peer relationships'
            ],
            DevelopmentalStage.MIDDLE_ADOLESCENCE: [
                'Career exploration',
                'Value development',
                'Complex reasoning',
                'Social dynamics'
            ],
            DevelopmentalStage.LATE_ADOLESCENCE: [
                'Life planning',
                'Decision making',
                'Independence',
                'Relationship building'
            ],
            DevelopmentalStage.YOUNG_ADULT: [
                'Career development',
                'Financial planning',
                'Relationship management',
                'Life skills mastery'
            ],
            DevelopmentalStage.MATURE_ADULT: [
                'Wisdom application',
                'Mentoring others',
                'Legacy building',
                'Continued growth'
            ]
        }.get(self.current_stage, ['Basic development'])
        
        return {
            'stage': self.current_stage,
            'age_range': stage_def['age_range'],
            'required_skills': stage_def['required_skills'],
            'learning_focus': stage_def.get('learning_focus', []),
            'emotional_range': stage_def.get('emotional_range', (0.0, 1.0)),
            'behaviors': behaviors,
            'current_milestones': stage_def.get('current_milestones', []),
            'upcoming_milestones': stage_def.get('upcoming_milestones', []),
            'complexity_range': stage_def.get('complexity_range', (0.1, 0.5)),
            'trust_emphasis': stage_def.get('trust_emphasis', 0.5)
        }
    
    def get_development_status(self) -> Dict[str, Any]:
        """Get detailed development status for current stage"""
        try:
            stage_progress = self.get_stage_progress()
            stage_reqs = self.get_stage_requirements()
            
            return {
                'current_stage': self.current_stage.name,
                'stage_duration': self.stage_duration,
                'skill_progress': self.skill_progress,
                'interaction_counts': self.interaction_counts,
                'metrics': self.stage_metrics,
                'ready_for_progression': self.check_stage_progression(),
                'stage_progress': stage_progress,
                'current_milestones': stage_reqs.get('current_milestones', []),
                'upcoming_milestones': stage_reqs.get('upcoming_milestones', [])
            }
        except Exception as e:
            print(f"Error getting development status: {str(e)}")
            # Return a minimal status if there's an error
            return {
                'current_stage': self.current_stage.name,
                'stage_duration': self.stage_duration,
                'skill_progress': self.skill_progress,
                'interaction_counts': self.interaction_counts,
                'metrics': self.stage_metrics,
                'ready_for_progression': False
            }
    
    def update_skill_progress(self, interaction_type: str, success_rate: float):
        """Update skill progress based on interaction type and success"""
        # Map interactions to skills
        skill_mapping = {
            'TALK': ['voice_recognition', 'sound_response'],
            'LOOK': ['visual_tracking'],
            'SMILE': ['social_bonding', 'emotional_expression'],
            'PLAY': ['object_permanence', 'motor_skills'],
            'COMFORT': ['social_bonding', 'emotional_expression'],
            'FEED': ['motor_skills'],
            'SLEEP': ['emotional_regulation'],
            'SING': ['sound_response', 'emotional_expression']
        }
        
        # Update relevant skills
        if interaction_type.upper() in skill_mapping:
            for skill in skill_mapping[interaction_type.upper()]:
                if skill in self.skill_progress:
                    current = self.skill_progress[skill]
                    # Weighted update with more emphasis on recent performance
                    self.skill_progress[skill] = 0.7 * current + 0.3 * success_rate
                
        # Track interaction counts
        interaction_category = {
            'TALK': 'mother_voice',
            'LOOK': 'visual_stimuli',
            'SOUND': 'sound_stimuli',
            'SMILE': 'social_engagement',
            'PLAY': 'motor_practice',
            'SING': 'mother_voice',
            'COMFORT': 'social_engagement',
            'FEED': 'motor_practice'
        }.get(interaction_type.upper())
        
        if interaction_category and interaction_category in self.interaction_counts:
            self.interaction_counts[interaction_category] += 1
    
    def update_stage(self, metrics: Dict[str, float], interactions_per_stage: int = 100):
        """Update stage metrics and check for progression"""
        # Update internal metrics
        for key, value in metrics.items():
            if key in self.stage_metrics:
                self.stage_metrics[key] = value
                
        self.stage_duration += 1
        
        # Check if ready to progress with enhanced criteria
        if self.check_stage_progression():
            self.consecutive_successes += 1
            if self.consecutive_successes >= 3:  # Need 3 consecutive successful evaluations
                self._advance_stage()
                self.consecutive_successes = 0
                # Reset interaction counts for new stage
                self.interaction_counts = {key: 0 for key in self.interaction_counts}
        else:
            self.consecutive_successes = 0
            
    def can_progress(self, interactions_per_stage: int = 100) -> bool:
        """Determine if the child can progress to the next stage"""
        characteristics = self.get_stage_characteristics()
        
        # Calculate overall development score with time acceleration
        development_score = (
            self.stage_metrics['success_rate'] * 0.3 +
            self.stage_metrics['emotional_regulation'] * 0.2 +
            self.stage_metrics['social_skills'] * 0.2 +
            self.stage_metrics['cognitive_development'] * 0.2 +
            (self.stage_metrics['abstraction'] + 
             self.stage_metrics['self_awareness']) * 0.1
        )
        
        # Reduce minimum duration requirement with time acceleration
        min_duration = max(5, interactions_per_stage // 4)  # Minimum 5 interactions
        if self.stage_duration < min_duration:
            return False
            
        # Lower thresholds slightly to account for accelerated development
        min_success_rate = characteristics.success_criteria['min_success_rate'] * 0.8
        progression_threshold = characteristics.success_criteria['progression_threshold'] * 0.8
        
        return (
            development_score > progression_threshold and
            self.stage_metrics['success_rate'] > min_success_rate
        )
        
    def _advance_stage(self):
        """Advance to the next developmental stage"""
        # Record current stage progress
        self.stage_history.append({
            'stage': self.current_stage,
            'duration': self.stage_duration,
            'final_metrics': self.stage_metrics.copy()
        })
        
        current_value = self.current_stage.value
        if current_value < len(DevelopmentalStage) - 1:
            self.current_stage = DevelopmentalStage(current_value + 1)
            # Reset metrics for new stage
            for key in self.stage_metrics:
                self.stage_metrics[key] = 0.0
            self.stage_duration = 0
            print(f"\nAdvancing to stage: {self.current_stage.name}")
            print(f"Stage requirements: {self.get_stage_characteristics()}")
    
    def get_stage_progress(self, interactions_per_stage: int = 100) -> Dict[str, float]:
        """Get current progress in the stage"""
        characteristics = self.get_stage_characteristics()
        
        return {
            'stage_name': self.current_stage.name,
            'duration': self.stage_duration,
            'progress': min(self.stage_duration / interactions_per_stage, 1.0),
            'success_rate': self.stage_metrics['success_rate'],
            'development_score': (
                self.stage_metrics['emotional_regulation'] * 0.3 +
                self.stage_metrics['social_skills'] * 0.3 +
                self.stage_metrics['cognitive_development'] * 0.4
            ),
            'ready_for_progression': self.can_progress(interactions_per_stage)
        }
    
    def check_stage_progression(self) -> bool:
        """Check if ready to progress to next stage with enhanced criteria"""
        if self.current_stage == DevelopmentalStage.NEWBORN:
            # Check core NEWBORN skills
            basic_skills_mastered = (
                self.skill_progress['voice_recognition'] > 0.7 and
                self.skill_progress['visual_tracking'] > 0.7 and
                self.skill_progress['sound_response'] > 0.6 and
                self.skill_progress['vocalization'] > 0.5
            )
            
            # Check interaction diversity
            sufficient_interactions = all(
                count >= 10 for count in self.interaction_counts.values()
            )
            
            # Check emotional development
            emotional_readiness = (
                self.stage_metrics['emotional_regulation'] > 0.6 and
                self.stage_metrics['social_skills'] > 0.5
            )
            
            # Calculate overall development score
            development_score = (
                self.skill_progress['voice_recognition'] * 0.2 +
                self.skill_progress['visual_tracking'] * 0.2 +
                self.skill_progress['sound_response'] * 0.2 +
                self.skill_progress['social_bonding'] * 0.2 +
                self.stage_metrics['emotional_regulation'] * 0.1 +
                self.stage_metrics['social_skills'] * 0.1
            )
            
            return (
                basic_skills_mastered and
                sufficient_interactions and
                emotional_readiness and
                development_score > 0.7 and
                self.stage_duration >= 50  # Minimum interactions required
            )
        
        return self.can_progress()
