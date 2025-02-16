# curriculum_manager.py
from enum import Enum
from dataclasses import dataclass
from typing import Dict, List, Optional, Set
import torch
import numpy as np
from datetime import datetime, timedelta

@dataclass
class StageMetrics:
    """Metrics that define success and progress in a developmental stage"""
    curiosity_threshold: float
    attention_span: timedelta
    emotional_stability: float
    social_awareness: float
    abstraction_level: float
    language_complexity: float
    memory_retention: float
    problem_solving: float
    motor_skills: float  # For embodied learning
    self_awareness: float
    
    def to_tensor(self) -> torch.Tensor:
        return torch.tensor([
            self.curiosity_threshold,
            self.attention_span.total_seconds() / 3600,  # hours
            self.emotional_stability,
            self.social_awareness,
            self.abstraction_level,
            self.language_complexity,
            self.memory_retention,
            self.problem_solving,
            self.motor_skills,
            self.self_awareness
        ], dtype=torch.float32)

@dataclass
class StageBehaviors:
    """Expected behaviors and capabilities for each stage"""
    allowed_actions: Set[str]
    required_skills: Set[str]
    learning_focus: List[str]
    emotional_range: List[str]
    social_patterns: List[str]
    cognitive_abilities: List[str]

class DevelopmentalStage(Enum):
    NEWBORN = 0
    EARLY_INFANCY = 1
    LATE_INFANCY = 2
    EARLY_TODDLER = 3
    LATE_TODDLER = 4
    EARLY_PRESCHOOL = 5
    LATE_PRESCHOOL = 6
    EARLY_CHILDHOOD = 7
    MIDDLE_CHILDHOOD = 8
    LATE_CHILDHOOD = 9
    EARLY_ELEMENTARY = 10
    MIDDLE_ELEMENTARY = 11
    LATE_ELEMENTARY = 12
    EARLY_ADOLESCENCE = 13
    MIDDLE_ADOLESCENCE = 14
    LATE_ADOLESCENCE = 15
    YOUNG_ADULT = 16
    MATURE_ADULT = 17

class DevelopmentalSystem:
    def __init__(self):
        self.stage_definitions = self._initialize_stages()
        self.current_stage = DevelopmentalStage.NEWBORN
        self.stage_history = []
        self.regression_count = 0
        self.last_transition = datetime.now()
        
    def _initialize_stages(self) -> Dict[DevelopmentalStage, tuple[StageMetrics, StageBehaviors]]:
        """Initialize comprehensive stage definitions with metrics and behaviors"""
        stages = {}
        
        # Newborn Stage (0-3 months) 👶
        stages[DevelopmentalStage.NEWBORN] = (
            StageMetrics(
                curiosity_threshold=0.2,
                attention_span=timedelta(minutes=5),
                emotional_stability=0.3,
                social_awareness=0.1,
                abstraction_level=0.1,
                language_complexity=0.1,
                memory_retention=0.2,
                problem_solving=0.1,
                motor_skills=0.1,
                self_awareness=0.1
            ),
            StageBehaviors(
                allowed_actions={'cry', 'sleep', 'feed'},
                required_skills={'basic_reflexes'},
                learning_focus=['sensory_processing', 'basic_motor_control'],
                emotional_range=['distress', 'contentment'],
                social_patterns=['eye_contact', 'crying_communication'],
                cognitive_abilities=['reflex_response', 'basic_recognition']
            )
        )

        # Early Infancy (3-6 months) 🍼
        stages[DevelopmentalStage.EARLY_INFANCY] = (
            StageMetrics(
                curiosity_threshold=0.3,
                attention_span=timedelta(minutes=10),
                emotional_stability=0.4,
                social_awareness=0.2,
                abstraction_level=0.2,
                language_complexity=0.2,
                memory_retention=0.3,
                problem_solving=0.2,
                motor_skills=0.2,
                self_awareness=0.2
            ),
            StageBehaviors(
                allowed_actions={'cry', 'sleep', 'feed', 'smile', 'babble'},
                required_skills={'basic_reflexes', 'head_control'},
                learning_focus=['object_permanence', 'vocal_experimentation'],
                emotional_range=['joy', 'distress', 'interest'],
                social_patterns=['social_smile', 'responsive_cooing'],
                cognitive_abilities=['pattern_recognition', 'cause_effect']
            )
        )

        # Late Infancy (6-12 months) 🎈
        stages[DevelopmentalStage.LATE_INFANCY] = (
            StageMetrics(
                curiosity_threshold=0.4,
                attention_span=timedelta(minutes=15),
                emotional_stability=0.5,
                social_awareness=0.3,
                abstraction_level=0.3,
                language_complexity=0.3,
                memory_retention=0.4,
                problem_solving=0.3,
                motor_skills=0.4,
                self_awareness=0.3
            ),
            StageBehaviors(
                allowed_actions={'cry', 'sleep', 'feed', 'smile', 'babble', 'crawl', 'grasp'},
                required_skills={'object_permanence', 'crawling', 'grasping'},
                learning_focus=['mobility', 'object_manipulation', 'sound_mimicry'],
                emotional_range=['joy', 'distress', 'interest', 'fear', 'anger'],
                social_patterns=['stranger_anxiety', 'joint_attention', 'social_games'],
                cognitive_abilities=['object_permanence', 'intentional_behavior']
            )
        )

        # Early Toddler (12-18 months) 🚶
        stages[DevelopmentalStage.EARLY_TODDLER] = (
            StageMetrics(
                curiosity_threshold=0.5,
                attention_span=timedelta(minutes=20),
                emotional_stability=0.5,
                social_awareness=0.4,
                abstraction_level=0.4,
                language_complexity=0.4,
                memory_retention=0.5,
                problem_solving=0.4,
                motor_skills=0.5,
                self_awareness=0.4
            ),
            StageBehaviors(
                allowed_actions={'walk', 'point', 'simple_words', 'explore', 'imitate'},
                required_skills={'walking', 'pointing', 'basic_words'},
                learning_focus=['language_acquisition', 'symbolic_play', 'tool_use'],
                emotional_range=['joy', 'distress', 'fear', 'anger', 'affection'],
                social_patterns=['parallel_play', 'simple_cooperation', 'empathy_beginning'],
                cognitive_abilities=['symbolic_thinking', 'simple_problem_solving']
            )
        )

        # Late Toddler (18-24 months) 🎨
        stages[DevelopmentalStage.LATE_TODDLER] = (
            StageMetrics(
                curiosity_threshold=0.6,
                attention_span=timedelta(minutes=25),
                emotional_stability=0.6,
                social_awareness=0.5,
                abstraction_level=0.5,
                language_complexity=0.5,
                memory_retention=0.6,
                problem_solving=0.5,
                motor_skills=0.6,
                self_awareness=0.5
            ),
            StageBehaviors(
                allowed_actions={'run', 'climb', 'simple_sentences', 'pretend_play'},
                required_skills={'running', 'self_awareness', 'two_word_combinations'},
                learning_focus=['self_awareness', 'language_expansion', 'imaginative_play'],
                emotional_range=['all_basic_emotions', 'beginning_complex_emotions'],
                social_patterns=['cooperative_play', 'turn_taking', 'helping_behaviors'],
                cognitive_abilities=['categorization', 'memory_recall', 'simple_planning']
            )
        )

        # Early Preschool (2-3 years) 📚
        stages[DevelopmentalStage.EARLY_PRESCHOOL] = (
            StageMetrics(
                curiosity_threshold=0.7,
                attention_span=timedelta(minutes=30),
                emotional_stability=0.6,
                social_awareness=0.6,
                abstraction_level=0.6,
                language_complexity=0.6,
                memory_retention=0.7,
                problem_solving=0.6,
                motor_skills=0.7,
                self_awareness=0.6
            ),
            StageBehaviors(
                allowed_actions={'complex_sentences', 'imaginative_play', 'drawing', 'sharing'},
                required_skills={'toilet_training', 'social_play', 'self_expression'},
                learning_focus=['emotional_regulation', 'social_rules', 'creative_expression'],
                emotional_range=['complex_emotions', 'early_empathy'],
                social_patterns=['group_play', 'simple_games', 'friendship_formation'],
                cognitive_abilities=['symbolic_play', 'basic_counting', 'color_recognition']
            )
        )
        
        # Late Preschool (3-4 years) 🎭
        stages[DevelopmentalStage.LATE_PRESCHOOL] = (
            StageMetrics(
                curiosity_threshold=0.75,
                attention_span=timedelta(minutes=45),
                emotional_stability=0.65,
                social_awareness=0.65,
                abstraction_level=0.65,
                language_complexity=0.7,
                memory_retention=0.7,
                problem_solving=0.65,
                motor_skills=0.75,
                self_awareness=0.7
            ),
            StageBehaviors(
                allowed_actions={'storytelling', 'role_play', 'basic_counting', 'question_asking'},
                required_skills={'turn_taking', 'basic_storytelling', 'emotional_recognition'},
                learning_focus=['narrative_skills', 'basic_math', 'emotional_understanding'],
                emotional_range=['complex_emotions', 'perspective_taking', 'guilt', 'pride'],
                social_patterns=['cooperative_play', 'imaginary_friends', 'social_rules'],
                cognitive_abilities=['sequencing', 'sorting', 'early_planning']
            )
        )

        # Early Childhood (4-5 years) 📖
        stages[DevelopmentalStage.EARLY_CHILDHOOD] = (
            StageMetrics(
                curiosity_threshold=0.8,
                attention_span=timedelta(hours=1),
                emotional_stability=0.7,
                social_awareness=0.7,
                abstraction_level=0.7,
                language_complexity=0.75,
                memory_retention=0.75,
                problem_solving=0.7,
                motor_skills=0.8,
                self_awareness=0.75
            ),
            StageBehaviors(
                allowed_actions={'complex_storytelling', 'game_rules', 'basic_writing', 'early_reading'},
                required_skills={'rule_following', 'letter_recognition', 'number_concepts'},
                learning_focus=['early_literacy', 'social_rules', 'abstract_thinking'],
                emotional_range=['emotional_regulation', 'empathy', 'moral_emotions'],
                social_patterns=['group_dynamics', 'friendship_maintenance', 'rule_negotiation'],
                cognitive_abilities=['pattern_recognition', 'hypothetical_thinking', 'categorization']
            )
        )

        # Middle Childhood (5-6 years) 🏫
        stages[DevelopmentalStage.MIDDLE_CHILDHOOD] = (
            StageMetrics(
                curiosity_threshold=0.85,
                attention_span=timedelta(hours=1.5),
                emotional_stability=0.75,
                social_awareness=0.75,
                abstraction_level=0.75,
                language_complexity=0.8,
                memory_retention=0.8,
                problem_solving=0.75,
                motor_skills=0.85,
                self_awareness=0.8
            ),
            StageBehaviors(
                allowed_actions={'reading', 'writing', 'arithmetic', 'complex_games'},
                required_skills={'basic_reading', 'simple_addition', 'self_regulation'},
                learning_focus=['academic_skills', 'peer_relationships', 'moral_reasoning'],
                emotional_range=['complex_emotional_understanding', 'self_conscious_emotions'],
                social_patterns=['peer_groups', 'team_play', 'social_comparison'],
                cognitive_abilities=['conservation', 'logical_reasoning', 'strategic_thinking']
            )
        )

        # Late Childhood (6-7 years) 🌟
        stages[DevelopmentalStage.LATE_CHILDHOOD] = (
            StageMetrics(
                curiosity_threshold=0.9,
                attention_span=timedelta(hours=2),
                emotional_stability=0.8,
                social_awareness=0.8,
                abstraction_level=0.8,
                language_complexity=0.85,
                memory_retention=0.85,
                problem_solving=0.8,
                motor_skills=0.9,
                self_awareness=0.85
            ),
            StageBehaviors(
                allowed_actions={'independent_reading', 'complex_math', 'social_planning'},
                required_skills={'fluent_reading', 'arithmetic', 'social_navigation'},
                learning_focus=['academic_mastery', 'social_skills', 'abstract_concepts'],
                emotional_range=['emotional_complexity', 'moral_reasoning'],
                social_patterns=['close_friendships', 'group_identity', 'social_hierarchy'],
                cognitive_abilities=['abstract_thinking', 'problem_solving', 'critical_thinking']
            )
        )

        # Early Elementary (7-8 years) 📝
        stages[DevelopmentalStage.EARLY_ELEMENTARY] = (
            StageMetrics(
                curiosity_threshold=0.9,
                attention_span=timedelta(hours=2.5),
                emotional_stability=0.85,
                social_awareness=0.85,
                abstraction_level=0.85,
                language_complexity=0.9,
                memory_retention=0.9,
                problem_solving=0.85,
                motor_skills=0.9,
                self_awareness=0.9
            ),
            StageBehaviors(
                allowed_actions={'research', 'project_planning', 'team_collaboration'},
                required_skills={'research_skills', 'time_management', 'teamwork'},
                learning_focus=['independent_learning', 'scientific_thinking', 'social_dynamics'],
                emotional_range=['emotional_sophistication', 'perspective_taking'],
                social_patterns=['peer_networks', 'social_roles', 'group_projects'],
                cognitive_abilities=['systematic_thinking', 'metacognition', 'hypothesis_testing']
            )
        )
        
        # Middle Elementary (8-9 years) 📚
        stages[DevelopmentalStage.MIDDLE_ELEMENTARY] = (
            StageMetrics(
                curiosity_threshold=0.92,
                attention_span=timedelta(hours=3),
                emotional_stability=0.87,
                social_awareness=0.87,
                abstraction_level=0.87,
                language_complexity=0.92,
                memory_retention=0.92,
                problem_solving=0.87,
                motor_skills=0.92,
                self_awareness=0.92
            ),
            StageBehaviors(
                allowed_actions={'complex_research', 'abstract_math', 'social_strategy', 'debate'},
                required_skills={'critical_analysis', 'advanced_arithmetic', 'conflict_resolution'},
                learning_focus=['analytical_thinking', 'social_complexity', 'advanced_planning'],
                emotional_range=['nuanced_emotions', 'social_insight', 'self_reflection'],
                social_patterns=['complex_friendships', 'social_negotiation', 'group_leadership'],
                cognitive_abilities=['abstract_reasoning', 'complex_planning', 'logical_deduction']
            )
        )

        # Late Elementary (9-11 years) 🎯
        stages[DevelopmentalStage.LATE_ELEMENTARY] = (
            StageMetrics(
                curiosity_threshold=0.94,
                attention_span=timedelta(hours=3.5),
                emotional_stability=0.89,
                social_awareness=0.89,
                abstraction_level=0.89,
                language_complexity=0.94,
                memory_retention=0.94,
                problem_solving=0.89,
                motor_skills=0.94,
                self_awareness=0.94
            ),
            StageBehaviors(
                allowed_actions={'advanced_research', 'complex_problem_solving', 'social_leadership'},
                required_skills={'research_methodology', 'abstract_reasoning', 'emotional_intelligence'},
                learning_focus=['independent_research', 'moral_complexity', 'social_systems'],
                emotional_range=['emotional_maturity', 'ethical_reasoning', 'identity_formation'],
                social_patterns=['social_hierarchy', 'peer_influence', 'group_dynamics'],
                cognitive_abilities=['systematic_analysis', 'abstract_concepts', 'complex_reasoning']
            )
        )

        # Early Adolescence (11-13 years) 🌱
        stages[DevelopmentalStage.EARLY_ADOLESCENCE] = (
            StageMetrics(
                curiosity_threshold=0.95,
                attention_span=timedelta(hours=4),
                emotional_stability=0.85,  # Slight dip due to hormonal changes
                social_awareness=0.92,
                abstraction_level=0.92,
                language_complexity=0.95,
                memory_retention=0.95,
                problem_solving=0.92,
                motor_skills=0.95,
                self_awareness=0.95
            ),
            StageBehaviors(
                allowed_actions={'philosophical_thinking', 'identity_exploration', 'social_experimentation'},
                required_skills={'abstract_thinking', 'emotional_regulation', 'social_navigation'},
                learning_focus=['identity_development', 'moral_philosophy', 'social_dynamics'],
                emotional_range=['complex_emotions', 'existential_thoughts', 'social_anxiety'],
                social_patterns=['peer_groups', 'social_identity', 'romantic_interest'],
                cognitive_abilities=['formal_operations', 'hypothetical_reasoning', 'self_reflection']
            )
        )

        # Middle Adolescence (13-15 years) 🤔
        stages[DevelopmentalStage.MIDDLE_ADOLESCENCE] = (
            StageMetrics(
                curiosity_threshold=0.96,
                attention_span=timedelta(hours=4.5),
                emotional_stability=0.83,  # Further emotional volatility
                social_awareness=0.94,
                abstraction_level=0.94,
                language_complexity=0.96,
                memory_retention=0.96,
                problem_solving=0.94,
                motor_skills=0.96,
                self_awareness=0.96
            ),
            StageBehaviors(
                allowed_actions={'abstract_reasoning', 'social_analysis', 'creative_expression'},
                required_skills={'emotional_awareness', 'social_competence', 'abstract_thinking'},
                learning_focus=['personal_identity', 'social_roles', 'future_planning'],
                emotional_range=['emotional_depth', 'identity_crisis', 'social_consciousness'],
                social_patterns=['peer_influence', 'romantic_relationships', 'group_identity'],
                cognitive_abilities=['complex_analysis', 'future_projection', 'moral_reasoning']
            )
        )

        # Late Adolescence (15-18 years) 🎓
        stages[DevelopmentalStage.LATE_ADOLESCENCE] = (
            StageMetrics(
                curiosity_threshold=0.97,
                attention_span=timedelta(hours=5),
                emotional_stability=0.87,  # Beginning to stabilize
                social_awareness=0.96,
                abstraction_level=0.96,
                language_complexity=0.97,
                memory_retention=0.97,
                problem_solving=0.96,
                motor_skills=0.97,
                self_awareness=0.97
            ),
            StageBehaviors(
                allowed_actions={'career_planning', 'relationship_building', 'independent_decision'},
                required_skills={'life_planning', 'emotional_maturity', 'social_responsibility'},
                learning_focus=['future_orientation', 'personal_values', 'social_responsibility'],
                emotional_range=['emotional_stability', 'future_anxiety', 'personal_identity'],
                social_patterns=['intimate_relationships', 'social_responsibility', 'independence'],
                cognitive_abilities=['advanced_planning', 'life_strategies', 'ethical_reasoning']
            )
        )    
            
        # Young Adult (18-21 years) 🎓
        stages[DevelopmentalStage.YOUNG_ADULT] = (
            StageMetrics(
                curiosity_threshold=0.98,
                attention_span=timedelta(hours=6),
                emotional_stability=0.92,  # Significant stabilization
                social_awareness=0.97,
                abstraction_level=0.97,
                language_complexity=0.98,
                memory_retention=0.98,
                problem_solving=0.97,
                motor_skills=0.98,
                self_awareness=0.98
            ),
            StageBehaviors(
                allowed_actions={
                    'complex_decision_making', 
                    'life_planning', 
                    'career_development',
                    'relationship_building', 
                    'independent_living'
                },
                required_skills={
                    'critical_thinking',
                    'emotional_intelligence',
                    'financial_planning',
                    'time_management',
                    'stress_management'
                },
                learning_focus=[
                    'professional_development',
                    'relationship_dynamics',
                    'personal_finance',
                    'life_skills',
                    'self_actualization'
                ],
                emotional_range=[
                    'emotional_maturity',
                    'professional_composure',
                    'relationship_depth',
                    'self_acceptance',
                    'future_orientation'
                ],
                social_patterns=[
                    'professional_networking',
                    'intimate_relationships',
                    'mentorship',
                    'community_involvement',
                    'cultural_awareness'
                ],
                cognitive_abilities=[
                    'strategic_planning',
                    'systems_thinking',
                    'ethical_reasoning',
                    'professional_judgment',
                    'complex_decision_making'
                ]
            )
        )

        # Mature Adult (21+ years) 🌟
        stages[DevelopmentalStage.MATURE_ADULT] = (
            StageMetrics(
                curiosity_threshold=0.99,
                attention_span=timedelta(hours=8),
                emotional_stability=0.95,  # Peak emotional regulation
                social_awareness=0.99,
                abstraction_level=0.99,
                language_complexity=0.99,
                memory_retention=0.99,
                problem_solving=0.99,
                motor_skills=0.99,
                self_awareness=0.99
            ),
            StageBehaviors(
                allowed_actions={
                    'wisdom_application',
                    'mentorship',
                    'complex_problem_solving',
                    'system_thinking',
                    'life_mastery',
                    'social_leadership',
                    'knowledge_synthesis'
                },
                required_skills={
                    'wisdom_integration',
                    'emotional_mastery',
                    'social_leadership',
                    'complex_reasoning',
                    'life_balance',
                    'adaptive_resilience',
                    'metacognitive_awareness'
                },
                learning_focus=[
                    'continuous_growth',
                    'wisdom_development',
                    'legacy_building',
                    'system_understanding',
                    'societal_contribution',
                    'knowledge_integration',
                    'personal_mastery'
                ],
                emotional_range=[
                    'emotional_wisdom',
                    'transcendent_awareness',
                    'balanced_perspective',
                    'nuanced_understanding',
                    'emotional_leadership',
                    'empathic_mastery',
                    'self_actualization'
                ],
                social_patterns=[
                    'mentorship_roles',
                    'community_leadership',
                    'wisdom_sharing',
                    'social_influence',
                    'legacy_building',
                    'cultural_synthesis',
                    'intergenerational_bonds'
                ],
                cognitive_abilities=[
                    'wisdom_synthesis',
                    'systems_mastery',
                    'ethical_leadership',
                    'complex_integration',
                    'transcendent_thinking',
                    'paradigm_innovation',
                    'metacognitive_mastery'
                ]
            )
        )
        
        return stages
    
    def evaluate_current_stage(self, metrics: Dict[str, float]) -> float:
        current_metrics = self.stage_definitions[self.current_stage][0]
        target_tensor = current_metrics.to_tensor()
        current_tensor = torch.tensor([
            metrics.get('curiosity', 0.0),
            metrics.get('attention_span', 0.0),
            metrics.get('emotional_stability', 0.0),
            metrics.get('social_awareness', 0.0),
            metrics.get('abstraction', 0.0),
            metrics.get('language', 0.0),
            metrics.get('memory', 0.0),
            metrics.get('problem_solving', 0.0),
            metrics.get('motor_skills', 0.0),
            metrics.get('self_awareness', 0.0)
        ])
        similarity = torch.nn.functional.cosine_similarity(
            current_tensor.unsqueeze(0),
            target_tensor.unsqueeze(0)
        )
        return similarity.item()
    
    def check_regression(self, performance: float) -> bool:
        if len(self.stage_history) >= 5:
            recent_performance = np.mean(self.stage_history[-5:])
            if performance < recent_performance * 0.7:
                self.regression_count += 1
                return True
        return False
    
    def update_stage(self, metrics: Dict[str, float]) -> Optional[str]:
        performance = self.evaluate_current_stage(metrics)
        self.stage_history.append(performance)
        if self.check_regression(performance):
            if self.regression_count >= 3:
                previous_stage = DevelopmentalStage(max(0, self.current_stage.value - 1))
                self.current_stage = previous_stage
                self.regression_count = 0
                return f"Regression detected: Moved back to {previous_stage.name}"
            return "Warning: Potential regression detected"
        if performance > 0.6:  # Lowered from 0.85
            if self.current_stage.value < len(DevelopmentalStage) - 1:
                self.current_stage = DevelopmentalStage(self.current_stage.value + 1)
                self.last_transition = datetime.now()
                self.regression_count = 0
                return f"Advanced to {self.current_stage.name}"
        return None
    
    def get_stage_requirements(self) -> Dict[str, any]:
        metrics, behaviors = self.stage_definitions[self.current_stage]
        return {
            'metrics': metrics.__dict__,
            'behaviors': {
                'allowed_actions': list(behaviors.allowed_actions),
                'required_skills': list(behaviors.required_skills),
                'learning_focus': behaviors.learning_focus,
                'emotional_range': behaviors.emotional_range,
                'social_patterns': behaviors.social_patterns,
                'cognitive_abilities': behaviors.cognitive_abilities
            }
        }
    
    def get_stage_appropriate_responses(self) -> Dict[str, List[str]]:
        """Returns templates for stage-appropriate responses"""
        stage_responses = {
            DevelopmentalStage.NEWBORN: {
                'comfort': ['[CRY]', '[SLEEP]', '[FEED]'],
                'social': ['[LOOK]', '[SMILE]'],
            },
            DevelopmentalStage.EARLY_INFANCY: {
                'vocal': ['[BABBLE]', '[COO]'],
                'motor': ['[REACH]', '[KICK]'],
                'social': ['[SMILE]', '[LAUGH]'],
            },
            DevelopmentalStage.LATE_INFANCY: {
                'motor': ['[CRAWL]', '[GRAB]'],
                'vocal': ['[BABBLE]', '[MAMA]', '[DADA]'],
                'social': ['[WAVE]', '[PLAY]'],
            },
            # Add more stage-appropriate responses for other stages
            # ...
        }
        return stage_responses.get(self.current_stage, {})