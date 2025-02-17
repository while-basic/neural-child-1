"""Curriculum module for managing developmental learning progression.

This module provides the curriculum management system for the Neural Child project,
handling stage-appropriate learning objectives and progression tracking.
"""

from typing import Dict, Any, List
import logging
from dataclasses import dataclass
from developmental_stages import DevelopmentalStage

logger = logging.getLogger(__name__)

@dataclass
class LearningObjective:
    """Represents a specific learning objective within the curriculum."""
    name: str
    description: str
    stage: DevelopmentalStage
    required_skills: List[str]
    success_criteria: Dict[str, float]
    completion_status: float = 0.0

class DevelopmentalCurriculum:
    """Manages the developmental curriculum and learning progression."""
    
    def __init__(self) -> None:
        """Initialize the developmental curriculum system."""
        self.objectives: Dict[DevelopmentalStage, List[LearningObjective]] = {}
        self.current_objectives: List[LearningObjective] = []
        self.current_stage: DevelopmentalStage = DevelopmentalStage.NEWBORN
        self._initialize_objectives()
        self._set_current_objectives(self.current_stage)
        self.interaction_history: List[Dict[str, Any]] = []
    
    def _set_current_objectives(self, stage: DevelopmentalStage) -> None:
        """Set the current objectives based on the developmental stage.
        
        Args:
            stage: The developmental stage to set objectives for
        """
        self.current_stage = stage
        self.current_objectives = self.get_objectives(stage)
        logger.info(f"Set current objectives for stage: {stage.name}")
    
    def _initialize_objectives(self) -> None:
        """Initialize the learning objectives for each developmental stage."""
        # TODO: NEURAL-126 - Implement comprehensive learning objectives
        self.objectives = {
            DevelopmentalStage.NEWBORN: [
                LearningObjective(
                    name="Basic Needs Recognition",
                    description="Learn to recognize and express basic needs",
                    stage=DevelopmentalStage.NEWBORN,
                    required_skills=["emotional_expression"],
                    success_criteria={"need_recognition": 0.7}
                ),
                LearningObjective(
                    name="Emotional Bonding",
                    description="Develop initial emotional bonds",
                    stage=DevelopmentalStage.NEWBORN,
                    required_skills=["emotional_response"],
                    success_criteria={"trust_level": 0.6}
                )
            ],
            DevelopmentalStage.EARLY_INFANCY: [
                LearningObjective(
                    name="Social Smiling",
                    description="Develop social smiling response",
                    stage=DevelopmentalStage.EARLY_INFANCY,
                    required_skills=["facial_recognition", "emotional_expression"],
                    success_criteria={"social_response": 0.6}
                )
            ],
            DevelopmentalStage.LATE_INFANCY: [
                LearningObjective(
                    name="Object Permanence",
                    description="Understand object permanence",
                    stage=DevelopmentalStage.LATE_INFANCY,
                    required_skills=["memory", "attention"],
                    success_criteria={"object_tracking": 0.7}
                )
            ],
            DevelopmentalStage.EARLY_TODDLER: [
                LearningObjective(
                    name="First Words",
                    description="Begin using simple words",
                    stage=DevelopmentalStage.EARLY_TODDLER,
                    required_skills=["vocalization", "memory"],
                    success_criteria={"word_usage": 0.6}
                )
            ],
            DevelopmentalStage.LATE_TODDLER: [
                LearningObjective(
                    name="Simple Sentences",
                    description="Form basic sentences",
                    stage=DevelopmentalStage.LATE_TODDLER,
                    required_skills=["language", "memory"],
                    success_criteria={"sentence_formation": 0.6}
                )
            ],
            DevelopmentalStage.EARLY_PRESCHOOL: [
                LearningObjective(
                    name="Imaginative Play",
                    description="Engage in pretend play",
                    stage=DevelopmentalStage.EARLY_PRESCHOOL,
                    required_skills=["imagination", "social_interaction"],
                    success_criteria={"creative_play": 0.7}
                )
            ],
            DevelopmentalStage.LATE_PRESCHOOL: [
                LearningObjective(
                    name="Basic Problem Solving",
                    description="Solve simple problems",
                    stage=DevelopmentalStage.LATE_PRESCHOOL,
                    required_skills=["reasoning", "memory"],
                    success_criteria={"problem_solving": 0.7}
                )
            ],
            DevelopmentalStage.EARLY_CHILDHOOD: [
                LearningObjective(
                    name="Early Reading",
                    description="Begin reading simple words",
                    stage=DevelopmentalStage.EARLY_CHILDHOOD,
                    required_skills=["letter_recognition", "phonics"],
                    success_criteria={"reading_basics": 0.6}
                )
            ],
            DevelopmentalStage.MIDDLE_CHILDHOOD: [
                LearningObjective(
                    name="Abstract Thinking",
                    description="Develop abstract thinking skills",
                    stage=DevelopmentalStage.MIDDLE_CHILDHOOD,
                    required_skills=["reasoning", "conceptualization"],
                    success_criteria={"abstract_thought": 0.7}
                )
            ],
            DevelopmentalStage.LATE_CHILDHOOD: [
                LearningObjective(
                    name="Critical Thinking",
                    description="Develop critical thinking skills",
                    stage=DevelopmentalStage.LATE_CHILDHOOD,
                    required_skills=["analysis", "evaluation"],
                    success_criteria={"critical_analysis": 0.7}
                )
            ],
            DevelopmentalStage.EARLY_ELEMENTARY: [
                LearningObjective(
                    name="Research Skills",
                    description="Develop basic research skills",
                    stage=DevelopmentalStage.EARLY_ELEMENTARY,
                    required_skills=["information_gathering", "organization"],
                    success_criteria={"research_ability": 0.7}
                )
            ],
            DevelopmentalStage.MIDDLE_ELEMENTARY: [
                LearningObjective(
                    name="Project Planning",
                    description="Learn to plan and execute projects",
                    stage=DevelopmentalStage.MIDDLE_ELEMENTARY,
                    required_skills=["planning", "execution"],
                    success_criteria={"project_completion": 0.7}
                )
            ],
            DevelopmentalStage.LATE_ELEMENTARY: [
                LearningObjective(
                    name="Advanced Analysis",
                    description="Develop advanced analytical skills",
                    stage=DevelopmentalStage.LATE_ELEMENTARY,
                    required_skills=["complex_analysis", "synthesis"],
                    success_criteria={"analytical_thinking": 0.8}
                )
            ],
            DevelopmentalStage.EARLY_ADOLESCENCE: [
                LearningObjective(
                    name="Identity Formation",
                    description="Develop sense of self",
                    stage=DevelopmentalStage.EARLY_ADOLESCENCE,
                    required_skills=["self_reflection", "emotional_awareness"],
                    success_criteria={"identity_development": 0.7}
                )
            ],
            DevelopmentalStage.MIDDLE_ADOLESCENCE: [
                LearningObjective(
                    name="Moral Reasoning",
                    description="Develop complex moral reasoning",
                    stage=DevelopmentalStage.MIDDLE_ADOLESCENCE,
                    required_skills=["ethical_thinking", "empathy"],
                    success_criteria={"moral_development": 0.8}
                )
            ],
            DevelopmentalStage.LATE_ADOLESCENCE: [
                LearningObjective(
                    name="Future Planning",
                    description="Develop long-term planning skills",
                    stage=DevelopmentalStage.LATE_ADOLESCENCE,
                    required_skills=["strategic_thinking", "goal_setting"],
                    success_criteria={"planning_ability": 0.8}
                )
            ],
            DevelopmentalStage.YOUNG_ADULT: [
                LearningObjective(
                    name="Life Management",
                    description="Develop independent life skills",
                    stage=DevelopmentalStage.YOUNG_ADULT,
                    required_skills=["self_management", "decision_making"],
                    success_criteria={"independence": 0.8}
                )
            ],
            DevelopmentalStage.MATURE_ADULT: [
                LearningObjective(
                    name="Wisdom Development",
                    description="Develop wisdom and mentorship abilities",
                    stage=DevelopmentalStage.MATURE_ADULT,
                    required_skills=["reflection", "guidance"],
                    success_criteria={"wisdom": 0.9}
                )
            ]
        }
        logger.info("Initialized developmental curriculum")
    
    def get_objectives(self, stage: DevelopmentalStage) -> List[LearningObjective]:
        """Get learning objectives for a specific developmental stage.
        
        Args:
            stage: The developmental stage to get objectives for
            
        Returns:
            List of learning objectives for the specified stage
        """
        return self.objectives.get(stage, [])
    
    def update_progress(self, objective_name: str, progress: float) -> None:
        """Update progress for a specific learning objective.
        
        Args:
            objective_name: Name of the objective to update
            progress: Progress value between 0 and 1
        """
        for objective in self.current_objectives:
            if objective.name == objective_name:
                objective.completion_status = min(1.0, max(0.0, progress))
                logger.info(f"Updated progress for {objective_name}: {progress:.2f}")
                # Update the main objectives dictionary as well
                stage_objectives = self.objectives.get(self.current_stage, [])
                for obj in stage_objectives:
                    if obj.name == objective_name:
                        obj.completion_status = objective.completion_status
                break
    
    def check_stage_completion(self, stage: DevelopmentalStage) -> bool:
        """Check if all objectives for a stage are completed.
        
        Args:
            stage: The developmental stage to check
            
        Returns:
            True if all objectives for the stage are completed
        """
        objectives = self.get_objectives(stage)
        if not objectives:
            return True
        return all(obj.completion_status >= 0.9 for obj in objectives)
    
    def get_state(self) -> Dict[str, Any]:
        """Get the current state of the curriculum.
        
        Returns:
            Dictionary containing the curriculum state
        """
        return {
            'current_stage': self.current_stage.name,
            'objectives': {
                stage.name: [
                    {
                        'name': obj.name,
                        'completion_status': obj.completion_status
                    }
                    for obj in objectives
                ]
                for stage, objectives in self.objectives.items()
            }
        }
    
    def load_state(self, state: Dict[str, Any]) -> None:
        """Load curriculum state from a state dictionary.
        
        Args:
            state: Dictionary containing the curriculum state
        """
        try:
            if 'current_stage' in state:
                self.current_stage = DevelopmentalStage[state['current_stage']]
            
            if 'objectives' in state:
                for stage_name, objectives in state['objectives'].items():
                    stage = DevelopmentalStage[stage_name]
                    if stage in self.objectives:
                        for saved_obj in objectives:
                            for obj in self.objectives[stage]:
                                if obj.name == saved_obj['name']:
                                    obj.completion_status = saved_obj['completion_status']
            
            # Update current objectives after loading state
            self._set_current_objectives(self.current_stage)
            logger.info("Successfully loaded curriculum state")
        except Exception as error:
            logger.error(f"Error loading curriculum state: {error}")
            self.reset()
    
    def reset(self) -> None:
        """Reset the curriculum to its initial state."""
        self._initialize_objectives()
        self.current_stage = DevelopmentalStage.NEWBORN
        self._set_current_objectives(self.current_stage)
        for objectives in self.objectives.values():
            for objective in objectives:
                objective.completion_status = 0.0
        logger.info("Reset curriculum to initial state")
    
    def update_stage(self, new_stage: DevelopmentalStage) -> None:
        """Update the current stage and its objectives.
        
        Args:
            new_stage: The new developmental stage
        """
        if new_stage != self.current_stage:
            self._set_current_objectives(new_stage)
            logger.info(f"Updated curriculum stage to: {new_stage.name}")
    
    def get_stage_requirements(self) -> Dict[str, Any]:
        """Get the requirements and behaviors for the current developmental stage.
        
        Returns:
            Dictionary containing stage requirements and behaviors
        """
        stage_requirements = {
            DevelopmentalStage.NEWBORN: {
                'behaviors': ['crying', 'feeding', 'sleeping'],
                'skills': ['emotional_expression', 'emotional_response'],
                'milestones': ['Basic reflexes', 'Can cry to express needs'],
                'current_milestones': ['Basic reflexes', 'Can cry to express needs'],
                'upcoming_milestones': ['Social smiling', 'Track moving objects'],
                'complexity_range': (0.1, 0.2)
            },
            DevelopmentalStage.EARLY_INFANCY: {
                'behaviors': ['smiling', 'cooing', 'reaching'],
                'skills': ['facial_recognition', 'emotional_expression'],
                'milestones': ['Social smiling', 'Track moving objects'],
                'current_milestones': ['Social smiling', 'Track moving objects'],
                'upcoming_milestones': ['Object manipulation', 'Babbling'],
                'complexity_range': (0.2, 0.3)
            }
        }
        
        # Get requirements for current stage or return default values
        requirements = stage_requirements.get(self.current_stage, {
            'behaviors': [],
            'skills': [],
            'milestones': [],
            'current_milestones': [],
            'upcoming_milestones': [],
            'complexity_range': (0.1, 0.2)  # Default complexity range for unknown stages
        })
        
        # Add learning objectives
        current_objectives = self.get_objectives(self.current_stage)
        requirements['learning_objectives'] = [
            {
                'name': obj.name,
                'description': obj.description,
                'required_skills': obj.required_skills,
                'completion_status': obj.completion_status
            }
            for obj in current_objectives
        ]
        
        return requirements
    
    def get_development_status(self) -> Dict[str, Any]:
        """Get comprehensive development status information.
        
        Returns:
            Dictionary containing current development status, including:
            - Current stage and duration
            - Skill progress
            - Interaction counts
            - Readiness for progression
            - Detailed metrics
        """
        # Calculate stage duration
        stage_duration = len([
            interaction for interaction in self.interaction_history 
            if interaction.get('stage') == self.current_stage
        ])
        
        # Calculate skill progress
        skill_progress = {}
        for objective in self.current_objectives:
            for skill in objective.required_skills:
                if skill not in skill_progress:
                    skill_progress[skill] = 0.0
                skill_progress[skill] = max(
                    skill_progress[skill],
                    objective.completion_status
                )
        
        # Count different types of interactions
        interaction_counts = {
            'learning': len([i for i in self.interaction_history if i.get('type') == 'learning']),
            'emotional': len([i for i in self.interaction_history if i.get('type') == 'emotional']),
            'social': len([i for i in self.interaction_history if i.get('type') == 'social']),
            'physical': len([i for i in self.interaction_history if i.get('type') == 'physical'])
        }
        
        # Calculate overall progress and readiness
        avg_completion = sum(obj.completion_status for obj in self.current_objectives) / len(self.current_objectives) if self.current_objectives else 0.0
        ready_for_progression = (
            avg_completion >= 0.7 and  # Basic skills mastered
            stage_duration >= 10 and    # Sufficient interactions
            all(count >= 5 for count in interaction_counts.values())  # Diverse interaction types
        )
        
        # Detailed metrics
        metrics = {
            'average_completion': avg_completion,
            'objectives_completed': sum(1 for obj in self.current_objectives if obj.completion_status >= 0.9),
            'total_objectives': len(self.current_objectives),
            'stage_progress': min(1.0, stage_duration / 50),  # Cap at 50 interactions
            'skill_mastery': sum(skill_progress.values()) / len(skill_progress) if skill_progress else 0.0
        }
        
        return {
            'current_stage': self.current_stage.name,
            'stage_duration': stage_duration,
            'skill_progress': skill_progress,
            'interaction_counts': interaction_counts,
            'ready_for_progression': ready_for_progression,
            'metrics': metrics
        } 