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
        'age_range': (0, 3),
        'complexity_range': (0.1, 0.2),
        'emotional_range': (0.6, 0.8),
        'required_skills': ['basic reflexes', 'crying'],
        'learning_focus': ['sensory processing', 'basic motor skills'],
        'current_milestones': ['Basic reflexes', 'Can cry to express needs'],
        'upcoming_milestones': [
            'Recognize mother\'s voice',
            'Follow moving objects with eyes',
            'Respond to loud sounds'
        ]
    },
    DevelopmentalStage.EARLY_INFANCY: {
        'age_range': (3, 6),
        'complexity_range': (0.2, 0.3),
        'emotional_range': (0.5, 0.7),
        'required_skills': ['object permanence', 'basic gestures'],
        'learning_focus': ['motor development', 'social bonding'],
        'current_milestones': ['Social smiling', 'Track moving objects'],
        'upcoming_milestones': [
            'Sitting without support',
            'Object manipulation',
            'Babbling'
        ]
    }
}

# Add default values for any undefined stages
for stage in DevelopmentalStage:
    if stage not in STAGE_DEFINITIONS:
        STAGE_DEFINITIONS[stage] = {
            'age_range': (0, 0),
            'complexity_range': (0.1, 1.0),
            'emotional_range': (0.0, 1.0),
            'required_skills': ['basic development'],
            'learning_focus': ['general development'],
            'current_milestones': ['Stage not fully defined'],
            'upcoming_milestones': ['Stage not fully defined']
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
