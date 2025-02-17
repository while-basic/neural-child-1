"""Main module for the Neural Child project.

This module implements the core functionality of the Neural Child system,
including the MotherLLM and DigitalChild classes that simulate the interaction
between a mother and a developing child.
"""

import torch
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import psutil
from dataclasses import dataclass
import logging
import os
import time

from llm_module import chat_completion
from child_model import DynamicNeuralChild
from developmental_stages import DevelopmentalStage, DevelopmentalSystem, get_stage_prompt
from memory_module import DifferentiableMemory
from moral_network import MoralPolicyNetwork
from metacognition import MetacognitionSystem
from self_supervised_trainer import AutonomousTrainer
from text_embed import get_embeddings
from autonomous_learner import AutonomousLearner
from sandbox_manager import SandboxManager
from training_system import DevelopmentalTrainer
from emotional_regulation import EmotionalRegulation
from config import config
from curriculum import DevelopmentalCurriculum

# Configure safe globals for PyTorch serialization
torch.serialization.add_safe_globals([DevelopmentalStage])

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

@dataclass
class EmotionalState:
    """Represents the emotional state with various dimensions."""
    happiness: float
    sadness: float
    anger: float
    fear: float
    surprise: float
    disgust: float
    trust: float
    anticipation: float

    def get_dominant_emotions(self) -> List[Tuple[str, float]]:
        """Return a list of dominant emotions and their strengths."""
        emotions = {
            'happy': self.happiness,
            'sad': self.sadness,
            'angry': self.anger,
            'fearful': self.fear,
            'surprised': self.surprise,
            'disgusted': self.disgust,
            'trusting': self.trust,
            'anticipating': self.anticipation
        }
        return sorted(emotions.items(), key=lambda x: x[1], reverse=True)[:3]

    def get_emotional_description(self) -> str:
        """Return a human-readable description of the emotional state."""
        emotions = {
            'happy': self.happiness,
            'sad': self.sadness,
            'angry': self.anger,
            'fearful': self.fear,
            'surprised': self.surprise,
            'disgusted': self.disgust,
            'trusting': self.trust,
            'anticipating': self.anticipation
        }
        return ', '.join(f'{e}({v:.2f})' for e, v in emotions.items() if v > 0.3)

    def to_vector(self) -> List[float]:
        """Convert the emotional state to a vector representation."""
        return [self.happiness, self.sadness, self.anger, self.fear, self.surprise, self.disgust, self.trust, self.anticipation]

    @classmethod
    def from_vector(cls, vector: List[float]) -> 'EmotionalState':
        """Create an EmotionalState instance from a vector representation."""
        return cls(
            happiness=vector[0],
            sadness=vector[1],
            anger=vector[2],
            fear=vector[3],
            surprise=vector[4],
            disgust=vector[5],
            trust=vector[6],
            anticipation=vector[7]
        )

class MotherLLM:
    """Simulates a mother's responses using advanced LLM technology.
    
    This class handles the generation of appropriate responses and stimuli
    based on the child's developmental stage, emotional state, and interaction history.
    """
    
    def __init__(self) -> None:
        """Initialize the MotherLLM with enhanced emotional processing."""
        self.emotional_history: List[EmotionalState] = []
        self.feedback_history: List[Dict[str, Any]] = []
        self.conversation_history: List[Dict[str, Any]] = []
        self.emotional_regulation = EmotionalRegulation()
        self.context_window = 5  # Number of past interactions to consider
        self.last_interaction_time = datetime.now()
        self.interaction_cooldown = 60  # Seconds between interactions
    
    def get_interaction_cooldown(self) -> int:
        """Get remaining cooldown time in seconds."""
        elapsed = (datetime.now() - self.last_interaction_time).total_seconds()
        return max(0, self.interaction_cooldown - int(elapsed))
    
    def perform_interaction(
        self,
        child: 'DigitalChild',
        category: str,
        interaction: str
    ) -> Tuple[str, EmotionalState]:
        """Perform a specific interaction with the child.
        
        Args:
            child: The DigitalChild instance to interact with
            category: The category of interaction
            interaction: The specific interaction to perform
            
        Returns:
            Tuple of (response text, emotional state)
        """
        # Verify interaction is available for current stage
        available_interactions = child.get_available_interactions()
        if (category not in available_interactions or 
            interaction not in available_interactions[category]):
            return (
                f"That interaction is not appropriate for the current developmental stage ({child.current_stage.name})",
                child.emotional_state
            )
        
        # Get interaction description
        description = child.get_interaction_description(category, interaction)
        
        # Build context for the model
        context = {
            'stage': child.current_stage,
            'age_months': child.age_months,
            'emotional_state': child.emotional_state,
            'interaction': interaction,
            'category': category,
            'description': description
        }
        
        # Generate appropriate response
        response_data = self._get_model_response(
            child.current_stage,
            f"Performing {interaction} ({category}): {description}",
            emotional_context=context
        )
        
        # Update interaction timestamp
        self.last_interaction_time = datetime.now()
        
        return response_data['text'], response_data['emotional_state']
    
    def _get_model_response(
        self, 
        stage: DevelopmentalStage, 
        user_input: str,
        emotional_context: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """Get formatted response from the model with enhanced emotional awareness.
        
        Args:
            stage: Current developmental stage of the child
            user_input: Input representing the child's current state
            emotional_context: Optional emotional context from previous interactions
            
        Returns:
            Dictionary containing the response text and emotional context
            
        Raises:
            Exception: If there's an error in getting the model response
        """
        try:
            # Get stage-specific prompt
            stage_prompt = get_stage_prompt(stage)
            
            # Build emotional context string
            context_str = ""
            if emotional_context and self.emotional_history:
                recent_emotions = self.emotional_history[-1]
                dominant_emotions = recent_emotions.get_dominant_emotions()
                context_str = f"\nPrevious emotional state: {recent_emotions.get_emotional_description()}"
                if dominant_emotions:
                    context_str += f"\nDominant emotions: {', '.join(f'{e}({v:.2f})' for e, v in dominant_emotions)}"
            
            # Build conversation context
            conv_context = ""
            if self.conversation_history:
                recent_convs = self.conversation_history[-self.context_window:]
                conv_context = "\nRecent interactions:\n" + "\n".join(
                    f"Child: {conv['input']}\nMother: {conv['response']}"
                    for conv in recent_convs
                )
            
            # Combine all context
            full_prompt = (
                f"{stage_prompt}\n"
                f"Child's current state: {user_input}\n"
                f"{context_str}\n"
                f"{conv_context}\n"
                "Respond with appropriate emotional depth and awareness:"
            )
            
            try:
                # Attempt to get chat completion with retries
                max_retries = 3
                retry_delay = 1  # seconds
                
                for attempt in range(max_retries):
                    try:
                        response = chat_completion(
                            system_prompt=full_prompt,
                            user_prompt=user_input,
                            structured_output=True
                        )
                        break
                    except Exception as api_error:
                        if attempt == max_retries - 1:  # Last attempt
                            logger.error(f"Failed to get chat completion after {max_retries} attempts: {api_error}")
                            return self._get_default_response()
                        logger.warning(f"Chat completion attempt {attempt + 1} failed: {api_error}. Retrying...")
                        time.sleep(retry_delay)
                
                # Extract and process emotional context
                emotional_context = response.get('emotional_context', {})
                emotional_state = EmotionalState(
                    happiness=float(emotional_context.get('happiness', 0.5)),
                    sadness=float(emotional_context.get('sadness', 0.5)),
                    anger=float(emotional_context.get('anger', 0.5)),
                    fear=float(emotional_context.get('fear', 0.5)),
                    surprise=float(emotional_context.get('surprise', 0.5)),
                    disgust=float(emotional_context.get('disgust', 0.0)),
                    trust=float(emotional_context.get('trust', 0.5)),
                    anticipation=float(emotional_context.get('anticipation', 0.5))
                )
                
                # Apply emotional regulation
                regulated_state = self.emotional_regulation(
                    torch.tensor(emotional_state.to_vector()),
                    context={'stage': stage.value, 'content': user_input}
                )
                
                # Convert regulated state back to EmotionalState
                regulated_emotional_state = EmotionalState.from_vector(regulated_state.tolist())
                
                return {
                    'text': response.get('response_text', 'I need a moment to think.'),
                    'emotional_state': regulated_emotional_state,
                    'action': response.get('action', None),
                    'effectiveness': float(response.get('effectiveness', 0.5))
                }
                
            except Exception as error:
                logger.error(f"Error in chat completion processing: {error}")
                return self._get_default_response()
            
        except Exception as error:
            logger.error(f"Error in _get_model_response: {error}")
            return self._get_default_response()
    
    def _get_default_response(self) -> Dict[str, Any]:
        """Provide a safe default response when normal processing fails."""
        return {
            'text': 'I need a moment to think.',
            'emotional_state': EmotionalState(
                happiness=0.5,
                sadness=0.5,
                anger=0.5,
                fear=0.5,
                surprise=0.5,
                disgust=0.0,
                trust=0.5,
                anticipation=0.5
            ),
            'action': None,
            'effectiveness': 0.5
        }

    def generate_stimulus(
        self, 
        stage: DevelopmentalStage, 
        user_input: str
    ) -> Tuple[str, EmotionalState]:
        """Generate appropriate stimulus based on developmental stage and user input.
        
        This method now includes enhanced emotional processing and context awareness.
        
        Args:
            stage: Current developmental stage of the child
            user_input: Input representing the child's current state
            
        Returns:
            Tuple containing the response text and emotional state
        """
        # Get emotional context from history
        emotional_context = None
        if self.emotional_history:
            last_state = self.emotional_history[-1]
            emotional_context = {
                'happiness': last_state.happiness,
                'sadness': last_state.sadness,
                'anger': last_state.anger,
                'fear': last_state.fear,
                'surprise': last_state.surprise,
                'disgust': last_state.disgust,
                'trust': last_state.trust,
                'anticipation': last_state.anticipation
            }
        
        # Generate response with context
        response_data = self._get_model_response(
            stage, 
            user_input, 
            emotional_context
        )
        
        # Update history
        self.emotional_history.append(response_data['emotional_state'])
        self.conversation_history.append({
            'timestamp': datetime.now(),
            'input': user_input,
            'response': response_data['text'],
            'emotional_state': response_data['emotional_state'],
            'action': response_data['action'],
            'effectiveness': response_data['effectiveness']
        })
        
        return response_data['text'], response_data['emotional_state']

class DigitalChild:
    """Simulates a developing digital child with learning capabilities."""
    
    def __init__(self) -> None:
        """Initialize the digital child with its core components."""
        # Set device configuration
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize neural components
        self.neural_model = DynamicNeuralChild().to(self.device)
        self.brain = self.neural_model  # Alias for backward compatibility
        self.memory = DifferentiableMemory()
        self.moral_network = MoralPolicyNetwork()
        self.metacognition = MetacognitionSystem()
        self.emotional_regulation = EmotionalRegulation()
        self.developmental_system = DevelopmentalSystem()
        
        # Initialize curriculum
        self.curriculum = DevelopmentalCurriculum()
        
        # Initialize autonomous learner with just the model
        self.autonomous_learner = AutonomousLearner(child_model=self.neural_model)
        
        # Initialize state
        self.current_stage = DevelopmentalStage.NEWBORN
        self.emotional_state = EmotionalState(0.5, 0.5, 0.5, 0.5, 0.5, 0.0, 0.5, 0.5)
        self.age_months = 0
        
        # Initialize warning system
        self.warning_state = "GREEN"
        self.recent_warnings = []
        self.speed_multiplier = 1.0
        self.speed_locked = False
        
        # Initialize acceleration metrics
        self.acceleration_metrics = {
            'base_learning_rate': 1.0,
            'current_multiplier': 1.0,
            'max_safe_multiplier': 5.0,
            'stability_factor': 1.0,
            'warning_penalty': 0.0,
            'stage_complexity_penalty': 0.0,
            'emotional_stability_bonus': 0.0,
            'learning_efficiency_bonus': 0.0
        }
        
        # Initialize interaction categories
        self.interaction_categories = {
            'PHYSICAL': [
                'HUG', 'ROCK', 'TICKLE', 'MASSAGE', 'DANCE',
                'WALK_TOGETHER', 'HAND_HOLDING', 'PEEK_A_BOO'
            ],
            'VERBAL': [
                'TALK', 'SING', 'READ_STORY', 'PRAISE',
                'TEACH_WORDS', 'ASK_QUESTIONS', 'EXPLAIN',
                'ENCOURAGE'
            ],
            'EMOTIONAL': [
                'SMILE', 'COMFORT', 'SOOTHE', 'EMPATHIZE',
                'VALIDATE_FEELINGS', 'SHOW_LOVE', 'CELEBRATE',
                'REASSURE'
            ],
            'COGNITIVE': [
                'SHOW_OBJECTS', 'PUZZLE_SOLVING', 'COUNTING',
                'COLOR_NAMING', 'SHAPE_SORTING', 'MEMORY_GAMES',
                'PATTERN_MATCHING', 'PROBLEM_SOLVING'
            ],
            'SOCIAL': [
                'PLAY_TOGETHER', 'SHARE', 'TAKE_TURNS',
                'INTRODUCE_OTHERS', 'GROUP_PLAY', 'ROLE_PLAY',
                'FOLLOW_LEADER', 'COOPERATIVE_GAMES'
            ],
            'CARE': [
                'FEED', 'CHANGE', 'BATHE', 'DRESS',
                'GROOM', 'SLEEP_ROUTINE', 'HEALTH_CHECK',
                'SAFETY_RULES'
            ],
            'SENSORY': [
                'SHOW_COLORS', 'PLAY_SOUNDS', 'TEXTURE_PLAY',
                'SMELL_ACTIVITIES', 'TASTE_EXPLORATION', 'LIGHT_DARK',
                'TEMPERATURE', 'BALANCE'
            ],
            'DEVELOPMENTAL': [
                'TUMMY_TIME', 'CRAWLING_PRACTICE', 'WALKING_SUPPORT',
                'FINE_MOTOR', 'GROSS_MOTOR', 'LANGUAGE_PRACTICE',
                'SELF_HELP_SKILLS', 'INDEPENDENCE'
            ]
        }
        
        # Load or initialize state
        self._initialize_or_load_state()
        
        # Ensure curriculum stage is synced
        self.curriculum.update_stage(self.current_stage)
    
    def _initialize_or_load_state(self) -> None:
        """Initialize new state or attempt to load from backup."""
        try:
            self._load_state()
        except Exception as error:
            logger.warning(f"Could not load state: {error}. Initializing fresh state.")
            self._initialize_fresh_state()
    
    def _load_state(self) -> None:
        """Load model state from backup if available."""
        try:
            # Add safe classes to torch serialization
            torch.serialization.add_safe_globals([
                DevelopmentalStage,
                EmotionalState,
                datetime,
                Dict, List, Any, Tuple  # Add common types
            ])
            
            # Try to load state with proper error handling
            try:
                state_dict = torch.load(
                    'checkpoints/child_state.pt',
                    map_location=self.device,
                    pickle_module=torch.serialization.pickle,
                    weights_only=False
                )
            except RuntimeError as e:
                logger.error(f"Error loading state file: {e}")
                raise
            except Exception as e:
                logger.error(f"Unexpected error loading state: {e}")
                raise
            
            if not isinstance(state_dict, dict):
                raise ValueError("Loaded state is not a dictionary")
            
            # Load neural model state if available
            if 'neural_model' in state_dict and isinstance(state_dict['neural_model'], dict):
                try:
                    self.neural_model.load_state_dict(state_dict['neural_model'])
                    logger.info("Successfully loaded neural model state")
                except Exception as e:
                    logger.warning(f"Could not load neural model state: {e}")
            
            # Load developmental stage
            if 'current_stage' in state_dict:
                if isinstance(state_dict['current_stage'], str):
                    self.current_stage = DevelopmentalStage[state_dict['current_stage']]
                elif isinstance(state_dict['current_stage'], DevelopmentalStage):
                    self.current_stage = state_dict['current_stage']
                else:
                    logger.warning("Invalid stage format in state file")
                    self.current_stage = DevelopmentalStage.NEWBORN
            
            # Load age
            self.age_months = state_dict.get('age_months', 0)
            
            # Load curriculum state
            if 'curriculum' in state_dict:
                try:
                    self.curriculum.load_state(state_dict['curriculum'])
                    logger.info("Successfully loaded curriculum state")
                except Exception as e:
                    logger.warning(f"Could not load curriculum state: {e}")
                    self.curriculum.reset()
            
            # Ensure curriculum stage matches child's stage
            self.curriculum.update_stage(self.current_stage)
            
            # Load emotional state
            if 'emotional_state' in state_dict:
                try:
                    emotional_data = state_dict['emotional_state']
                    if isinstance(emotional_data, dict):
                        self.emotional_state = EmotionalState(
                            happiness=float(emotional_data.get('happiness', 0.5)),
                            sadness=float(emotional_data.get('sadness', 0.5)),
                            anger=float(emotional_data.get('anger', 0.5)),
                            fear=float(emotional_data.get('fear', 0.5)),
                            surprise=float(emotional_data.get('surprise', 0.5)),
                            disgust=float(emotional_data.get('disgust', 0.0)),
                            trust=float(emotional_data.get('trust', 0.5)),
                            anticipation=float(emotional_data.get('anticipation', 0.5))
                        )
                    elif isinstance(emotional_data, (list, tuple)) and len(emotional_data) >= 4:
                        self.emotional_state = EmotionalState(
                            happiness=float(emotional_data[0]),
                            sadness=float(emotional_data[1]),
                            anger=float(emotional_data[2]),
                            fear=float(emotional_data[3]),
                            surprise=float(emotional_data[4]) if len(emotional_data) > 4 else 0.0,
                            disgust=float(emotional_data[5]) if len(emotional_data) > 5 else 0.0,
                            trust=float(emotional_data[6]) if len(emotional_data) > 6 else 0.5,
                            anticipation=float(emotional_data[7]) if len(emotional_data) > 7 else 0.5
                        )
                    logger.info("Successfully loaded emotional state")
                except Exception as e:
                    logger.warning(f"Could not load emotional state: {e}")
                    self.emotional_state = EmotionalState(0.5, 0.5, 0.5, 0.5, 0.5, 0.0, 0.5, 0.5)
            
            # Load warning system state
            self.warning_state = state_dict.get('warning_state', "GREEN")
            self.recent_warnings = state_dict.get('recent_warnings', [])
            self.speed_multiplier = float(state_dict.get('speed_multiplier', 1.0))
            self.speed_locked = bool(state_dict.get('speed_locked', False))
            
            logger.info("Successfully loaded model state")
            
        except FileNotFoundError:
            logger.info("No existing state found. Starting fresh.")
            self._initialize_fresh_state()
        except Exception as error:
            logger.error(f"Error loading state: {str(error)}")
            logger.info("Initializing fresh state due to loading error.")
            self._initialize_fresh_state()
    
    def _initialize_fresh_state(self) -> None:
        """Initialize a fresh state for the model."""
        logger.info("Initializing fresh model state")
        self.current_stage = DevelopmentalStage.NEWBORN
        self.age_months = 0
        self.emotional_state = EmotionalState(0.5, 0.5, 0.5, 0.5, 0.5, 0.0, 0.5, 0.5)
        self.curriculum.reset()  # Reset curriculum to initial state
        self.curriculum.update_stage(self.current_stage)  # Ensure stage is synced
        
        # Initialize warning system
        self.warning_state = "GREEN"
        self.recent_warnings = []
        self.speed_multiplier = 1.0
        self.speed_locked = False
    
    def save_state(self) -> None:
        """Save the current model state with error handling."""
        try:
            # Create checkpoints directory if it doesn't exist
            os.makedirs('checkpoints', exist_ok=True)
            
            # Prepare state dictionary with error handling for each component
            state_dict = {}
            
            try:
                state_dict['neural_model'] = self.neural_model.state_dict()
            except Exception as e:
                logger.error(f"Failed to save neural model state: {e}")
                state_dict['neural_model'] = {}
            
            state_dict['current_stage'] = self.current_stage
            state_dict['age_months'] = self.age_months
            
            try:
                state_dict['curriculum'] = self.curriculum.get_state()
            except Exception as e:
                logger.error(f"Failed to save curriculum state: {e}")
                state_dict['curriculum'] = {}
            
            # Handle emotional state
            if isinstance(self.emotional_state, EmotionalState):
                state_dict['emotional_state'] = {
                    'happiness': self.emotional_state.happiness,
                    'sadness': self.emotional_state.sadness,
                    'anger': self.emotional_state.anger,
                    'fear': self.emotional_state.fear,
                    'surprise': self.emotional_state.surprise,
                    'disgust': self.emotional_state.disgust,
                    'trust': self.emotional_state.trust,
                    'anticipation': self.emotional_state.anticipation
                }
            elif isinstance(self.emotional_state, torch.Tensor):
                state_dict['emotional_state'] = self.emotional_state.cpu().tolist()
            else:
                state_dict['emotional_state'] = {
                    'happiness': 0.5,
                    'sadness': 0.5,
                    'anger': 0.5,
                    'fear': 0.5,
                    'surprise': 0.5,
                    'disgust': 0.0,
                    'trust': 0.5,
                    'anticipation': 0.5
                }
            
            # Save warning system state
            state_dict['warning_state'] = self.warning_state
            state_dict['recent_warnings'] = self.recent_warnings
            state_dict['speed_multiplier'] = self.speed_multiplier
            state_dict['speed_locked'] = self.speed_locked
            
            # Save state with error handling
            try:
                torch.save(state_dict, 'checkpoints/child_state.pt')
                logger.info("Successfully saved model state")
            except Exception as e:
                logger.error(f"Error during final state save: {e}")
                # Attempt to save without neural model if full save fails
                if 'neural_model' in state_dict:
                    del state_dict['neural_model']
                    try:
                        torch.save(state_dict, 'checkpoints/child_state_partial.pt')
                        logger.info("Saved partial state without neural model")
                    except Exception as backup_error:
                        logger.error(f"Failed to save even partial state: {backup_error}")
                        
        except Exception as e:
            logger.error(f"Critical error in save_state: {e}")
            # Attempt to save minimal state
            try:
                minimal_state = {
                    'current_stage': self.current_stage,
                    'age_months': self.age_months,
                    'warning_state': self.warning_state
                }
                torch.save(minimal_state, 'checkpoints/child_state_minimal.pt')
                logger.info("Saved minimal state after critical error")
            except Exception as minimal_error:
                logger.error(f"Failed to save even minimal state: {minimal_error}")

    def update_emotions(self, mother_response: EmotionalState) -> None:
        """Update emotional state based on mother's response.
        
        Args:
            mother_response: Emotional state from mother's response
        """
        # Use emotional regulation to modulate response
        regulated_response = self.emotional_regulation(
            torch.tensor(mother_response.to_vector(), device=self.device),
            context={'stage': self.current_stage.value}
        )
        
        # Convert tensor back to EmotionalState
        if isinstance(regulated_response, torch.Tensor):
            regulated_response = EmotionalState.from_vector(regulated_response.squeeze().cpu().tolist())
        
        self.emotional_state = regulated_response

    def express_feeling(self) -> str:
        """Generate an expression of the current emotional state.
        
        Returns:
            String describing the current emotional state
        """
        # Ensure emotional_state is a dataclass instance, not a tensor
        if isinstance(self.emotional_state, torch.Tensor):
            # Convert tensor to EmotionalState if needed
            if self.emotional_state.size(0) == 4:  # Handle legacy 4-dimension tensors
                self.emotional_state = EmotionalState(
                    happiness=float(self.emotional_state[0]),
                    sadness=float(self.emotional_state[1]),
                    anger=float(self.emotional_state[2]),
                    fear=float(self.emotional_state[3]),
                    surprise=0.0,  # Default values for additional dimensions
                    disgust=0.0,
                    trust=0.5,
                    anticipation=0.5
                )
            else:  # Handle full 8-dimension tensors
                self.emotional_state = EmotionalState(
                    happiness=float(self.emotional_state[0]),
                    sadness=float(self.emotional_state[1]),
                    anger=float(self.emotional_state[2]),
                    fear=float(self.emotional_state[3]),
                    surprise=float(self.emotional_state[4]) if self.emotional_state.size(0) > 4 else 0.0,
                    disgust=float(self.emotional_state[5]) if self.emotional_state.size(0) > 5 else 0.0,
                    trust=float(self.emotional_state[6]) if self.emotional_state.size(0) > 6 else 0.5,
                    anticipation=float(self.emotional_state[7]) if self.emotional_state.size(0) > 7 else 0.5
                )
        
        # Map emotions to expressions
        emotions = {
            'happy': self.emotional_state.happiness,
            'sad': self.emotional_state.sadness,
            'angry': self.emotional_state.anger,
            'fearful': self.emotional_state.fear,
            'surprised': self.emotional_state.surprise,
            'disgusted': self.emotional_state.disgust,
            'trusting': self.emotional_state.trust,
            'anticipating': self.emotional_state.anticipation
        }
        
        # Find the dominant emotion
        dominant_emotion = max(emotions.items(), key=lambda x: x[1])
        
        # Generate appropriate expression based on dominant emotion
        expressions = {
            'happy': "I feel happy and content!",
            'sad': "I'm feeling a bit sad...",
            'angry': "I'm feeling frustrated.",
            'fearful': "I'm feeling uncertain and scared.",
            'surprised': "I'm surprised!",
            'disgusted': "I'm feeling disgusted.",
            'trusting': "I'm feeling trusting.",
            'anticipating': "I'm feeling excited!"
        }
        
        expression = expressions[dominant_emotion[0]]
        
        # Add intensity modifier based on the emotion's strength
        if dominant_emotion[1] > 0.8:
            expression = "Very " + expression.lower()
        elif dominant_emotion[1] < 0.3:
            expression = "Slightly " + expression.lower()
        
        return expression

    def learn(self, mother_feedback: str) -> None:
        """Process and learn from mother's feedback.
        
        Args:
            mother_feedback: The feedback text from the mother
        """
        # TODO: NEURAL-125 - Implement learning from feedback
        # Pass curriculum data to autonomous learner during feedback processing
        self.autonomous_learner.process_feedback(
            mother_feedback,
            current_stage=self.current_stage,
            learning_objectives=self.curriculum.get_objectives(self.current_stage)
        )
        self.metacognition.update(mother_feedback)
        
    def age(self) -> None:
        """Age the digital child by one time step."""
        # TODO: NEURAL-125 - Implement aging logic
        pass

    def get_warning_indicators(self) -> Dict[str, Any]:
        """Get current warning indicators for development monitoring.
        
        Returns:
            Dict containing:
            - warning_state: Current warning level (GREEN/YELLOW/RED)
            - metrics: Dict of monitored metrics (emotional_stability, learning_efficiency, etc.)
            - recent_warnings: List of recent warning events
            - stage_limit: Maximum safe speed for current stage
            - speed_multiplier: Current speed multiplier
        """
        try:
            # Get metrics from various systems
            metacog_metrics = self.metacognition.get_metrics()
            learning_rate = self.autonomous_learner.get_learning_rate()
            stage_reqs = self.curriculum.get_stage_requirements()
            
            # Calculate emotional stability
            emotional_state = self.emotional_state
            if isinstance(emotional_state, torch.Tensor):
                emotional_state = EmotionalState(
                    happiness=float(emotional_state[0]),
                    sadness=float(emotional_state[1]),
                    anger=float(emotional_state[2]),
                    fear=float(emotional_state[3]),
                    surprise=float(emotional_state[4]) if len(emotional_state) > 4 else 0.0,
                    disgust=float(emotional_state[5]) if len(emotional_state) > 5 else 0.0,
                    trust=float(emotional_state[6]) if len(emotional_state) > 6 else 0.5,
                    anticipation=float(emotional_state[7]) if len(emotional_state) > 7 else 0.5
                )
            
            # Calculate key metrics
            metrics = {
                'emotional_stability': float(metacog_metrics.get('emotional_stability', 0.0)),
                'learning_efficiency': float(metacog_metrics.get('learning_efficiency', 0.0)),
                'attention_level': float(metacog_metrics.get('attention_focus', 0.0)),
                'overstimulation_risk': 1.0 - float(metacog_metrics.get('stress_tolerance', 0.0))
            }
            
            # Determine warning state
            warning_state = "GREEN"
            warning_reasons = []
            
            if metrics['emotional_stability'] < 0.3 or metrics['overstimulation_risk'] > 0.7:
                warning_state = "RED"
                warning_reasons.append("Critical: Emotional stability low or overstimulation risk high")
            elif metrics['emotional_stability'] < 0.6 or metrics['overstimulation_risk'] > 0.4:
                warning_state = "YELLOW"
                warning_reasons.append("Warning: Emotional stability or overstimulation risk concerning")
            
            if metrics['learning_efficiency'] < 0.4:
                warning_state = "RED"
                warning_reasons.append("Critical: Learning efficiency severely impaired")
            elif metrics['learning_efficiency'] < 0.6:
                warning_state = max(warning_state, "YELLOW")
                warning_reasons.append("Warning: Learning efficiency below optimal")
            
            if metrics['attention_level'] < 0.3:
                warning_state = "RED"
                warning_reasons.append("Critical: Attention level critically low")
            elif metrics['attention_level'] < 0.6:
                warning_state = max(warning_state, "YELLOW")
                warning_reasons.append("Warning: Attention level below optimal")
            
            # Calculate stage-appropriate speed limit
            base_limit = 5.0
            stage_complexity = stage_reqs['complexity_range'][1]  # Use upper bound
            stage_limit = max(1.0, min(base_limit, base_limit * (1 - stage_complexity)))
            
            # Adjust speed limit based on warning state
            if warning_state == "RED":
                stage_limit = 1.0
            elif warning_state == "YELLOW":
                stage_limit = min(stage_limit, 3.0)
            
            # Get current speed multiplier
            speed_multiplier = getattr(self, 'speed_multiplier', 1.0)
            
            # Create warning event if state changed
            recent_warnings = getattr(self, 'recent_warnings', [])
            if warning_reasons:
                recent_warnings.append({
                    'timestamp': datetime.now(),
                    'state': warning_state,
                    'reason': '; '.join(warning_reasons)
                })
                # Keep only last 10 warnings
                recent_warnings = recent_warnings[-10:]
                self.recent_warnings = recent_warnings
            
            return {
                'warning_state': warning_state,
                'metrics': metrics,
                'recent_warnings': recent_warnings,
                'stage_limit': stage_limit,
                'speed_multiplier': speed_multiplier
            }
            
        except Exception as e:
            logger.error(f"Error getting warning indicators: {str(e)}")
            return {
                'warning_state': "RED",
                'metrics': {
                    'emotional_stability': 0.0,
                    'learning_efficiency': 0.0,
                    'attention_level': 0.0,
                    'overstimulation_risk': 1.0
                },
                'recent_warnings': [{
                    'timestamp': datetime.now(),
                    'state': "RED",
                    'reason': f"System error: {str(e)}"
                }],
                'stage_limit': 1.0,
                'speed_multiplier': 1.0
            }
            
    def set_development_speed(self, speed: float) -> None:
        """Set the development speed multiplier with safety checks.
        
        Args:
            speed: New speed multiplier value (1.0 = normal speed)
        """
        # Check if speed is locked due to critical warnings
        if self.speed_locked:
            logger.warning("Speed locked at 1x due to critical warnings")
            self.speed_multiplier = 1.0
            return
            
        # Get current warning state
        warning_indicators = self.get_warning_indicators()
        
        # Force speed to 1x if in RED warning state
        if warning_indicators['warning_state'] == "RED":
            logger.warning("Speed forced to 1x due to RED warning state")
            self.speed_multiplier = 1.0
            self.speed_locked = True
            return
            
        # Get acceleration metrics for limits
        metrics = self.get_acceleration_metrics()
        max_safe = metrics['max_safe_multiplier']
        
        # Clamp speed between 1.0 and max_safe
        self.speed_multiplier = max(1.0, min(speed, max_safe))
        
        if self.speed_multiplier != speed:
            logger.info(f"Speed limited to {self.speed_multiplier}x for safety (requested: {speed}x)")
        else:
            logger.info(f"Development speed set to {self.speed_multiplier}x")
            
    def unlock_speed(self) -> bool:
        """Attempt to unlock speed control after being locked.
        
        Returns:
            bool: True if successfully unlocked, False if still in critical state
        """
        warning_indicators = self.get_warning_indicators()
        if warning_indicators['warning_state'] != "RED":
            self.speed_locked = False
            logger.info("Speed control unlocked")
            return True
        return False

    def get_acceleration_metrics(self) -> Dict[str, float]:
        """Get current acceleration metrics and limits.
        
        Returns:
            Dict containing acceleration metrics and their current values
        """
        try:
            warning_indicators = self.get_warning_indicators()
            stage_reqs = self.curriculum.get_stage_requirements()
            
            # Update acceleration metrics
            self.acceleration_metrics.update({
                'current_multiplier': self.speed_multiplier,
                'stability_factor': 1.0 - warning_indicators['metrics']['overstimulation_risk'],
                'warning_penalty': 0.5 if warning_indicators['warning_state'] == "YELLOW" else 
                                 1.0 if warning_indicators['warning_state'] == "RED" else 0.0,
                'stage_complexity_penalty': stage_reqs['complexity_range'][1] * 0.5,
                'emotional_stability_bonus': warning_indicators['metrics']['emotional_stability'] * 0.2,
                'learning_efficiency_bonus': warning_indicators['metrics']['learning_efficiency'] * 0.2
            })
            
            # Calculate max safe multiplier
            max_safe = (
                self.acceleration_metrics['max_safe_multiplier'] *
                self.acceleration_metrics['stability_factor'] *
                (1.0 - self.acceleration_metrics['warning_penalty']) *
                (1.0 - self.acceleration_metrics['stage_complexity_penalty']) +
                self.acceleration_metrics['emotional_stability_bonus'] +
                self.acceleration_metrics['learning_efficiency_bonus']
            )
            
            self.acceleration_metrics['max_safe_multiplier'] = max(1.0, min(5.0, max_safe))
            
            return self.acceleration_metrics
            
        except Exception as e:
            logger.error(f"Error calculating acceleration metrics: {str(e)}")
            return {
                'base_learning_rate': 1.0,
                'current_multiplier': 1.0,
                'max_safe_multiplier': 1.0,
                'stability_factor': 1.0,
                'warning_penalty': 0.0,
                'stage_complexity_penalty': 0.0,
                'emotional_stability_bonus': 0.0,
                'learning_efficiency_bonus': 0.0
            }

    def get_available_interactions(self) -> Dict[str, List[str]]:
        """Get available interactions appropriate for the current developmental stage.
        
        Returns:
            Dict containing categories of available interactions filtered by current stage
        """
        stage_appropriate_interactions = {
            DevelopmentalStage.NEWBORN: {
                'PHYSICAL': ['HUG', 'ROCK', 'MASSAGE'],
                'VERBAL': ['TALK', 'SING'],
                'EMOTIONAL': ['SMILE', 'COMFORT', 'SOOTHE'],
                'CARE': ['FEED', 'CHANGE', 'BATHE'],
                'SENSORY': ['SHOW_COLORS', 'PLAY_SOUNDS'],
                'DEVELOPMENTAL': ['TUMMY_TIME']
            },
            DevelopmentalStage.EARLY_INFANCY: {
                'PHYSICAL': ['HUG', 'ROCK', 'MASSAGE', 'PEEK_A_BOO'],
                'VERBAL': ['TALK', 'SING', 'READ_STORY'],
                'EMOTIONAL': ['SMILE', 'COMFORT', 'SOOTHE', 'SHOW_LOVE'],
                'COGNITIVE': ['SHOW_OBJECTS'],
                'CARE': ['FEED', 'CHANGE', 'BATHE', 'SLEEP_ROUTINE'],
                'SENSORY': ['SHOW_COLORS', 'PLAY_SOUNDS', 'TEXTURE_PLAY'],
                'DEVELOPMENTAL': ['TUMMY_TIME', 'CRAWLING_PRACTICE']
            },
            DevelopmentalStage.LATE_INFANCY: {
                'PHYSICAL': ['HUG', 'ROCK', 'TICKLE', 'PEEK_A_BOO', 'HAND_HOLDING'],
                'VERBAL': ['TALK', 'SING', 'READ_STORY', 'TEACH_WORDS'],
                'EMOTIONAL': ['SMILE', 'COMFORT', 'SOOTHE', 'SHOW_LOVE', 'CELEBRATE'],
                'COGNITIVE': ['SHOW_OBJECTS', 'SHAPE_SORTING'],
                'SOCIAL': ['PLAY_TOGETHER'],
                'CARE': ['FEED', 'CHANGE', 'BATHE', 'SLEEP_ROUTINE', 'SAFETY_RULES'],
                'SENSORY': ['SHOW_COLORS', 'PLAY_SOUNDS', 'TEXTURE_PLAY', 'LIGHT_DARK'],
                'DEVELOPMENTAL': ['CRAWLING_PRACTICE', 'WALKING_SUPPORT', 'FINE_MOTOR']
            }
            # Add more stages as needed
        }
        
        # Get interactions for current stage
        available = stage_appropriate_interactions.get(
            self.current_stage,
            {k: v for k, v in self.interaction_categories.items()}  # Default to all if stage not found
        )
        
        return available

    def get_interaction_description(self, category: str, interaction: str) -> str:
        """Get a description of what a specific interaction does.
        
        Args:
            category: The category of the interaction
            interaction: The specific interaction name
            
        Returns:
            A string describing the interaction and its benefits
        """
        descriptions = {
            'PHYSICAL': {
                'HUG': "Provides physical comfort and security, promotes bonding and emotional development",
                'ROCK': "Soothes and calms, helps with vestibular development",
                'TICKLE': "Encourages laughter and social interaction, develops body awareness",
                'MASSAGE': "Promotes relaxation, improves circulation and body awareness",
                'DANCE': "Develops gross motor skills, rhythm, and coordination",
                'WALK_TOGETHER': "Practices mobility and builds confidence in movement",
                'HAND_HOLDING': "Provides security and support during mobility practice",
                'PEEK_A_BOO': "Develops object permanence and social interaction skills"
            },
            'VERBAL': {
                'TALK': "Exposes to language, promotes verbal development",
                'SING': "Develops rhythm, memory, and language skills",
                'READ_STORY': "Builds vocabulary, attention span, and imagination",
                'PRAISE': "Builds confidence and reinforces positive behaviors",
                'TEACH_WORDS': "Expands vocabulary and language comprehension",
                'ASK_QUESTIONS': "Develops critical thinking and verbal expression",
                'EXPLAIN': "Builds understanding and knowledge of the world",
                'ENCOURAGE': "Builds confidence and motivation"
            }
            # Add more categories as needed
        }
        
        return descriptions.get(category, {}).get(
            interaction,
            f"Interaction that helps develop {category.lower()} skills"
        )

def main() -> None:
    """Main function to run the Neural Child simulation."""
    try:
        mother = MotherLLM()
        child = DigitalChild()
        
        # Initialize trainer with all required components
        trainer = DevelopmentalTrainer(
            child_model=child.neural_model,
            memory=child.memory,
            emotional_regulation=child.emotional_regulation,
            curriculum_manager=child.curriculum,
            mother_llm=mother,
            metacognition_system=child.metacognition,
            config={
                'device': child.device,
                'learning_rate': 3e-4,  # Default learning rate from config.yaml
                'max_iterations': 1000,
                'checkpoint_dir': 'checkpoints',
                'log_dir': 'logs'
            }
        )
        
        logger.info("Starting Neural Child simulation...")
        
        # Main interaction loop
        while True:
            # Get child's current state
            child_expression = child.express_feeling()
            warning_indicators = child.get_warning_indicators()
            acceleration_metrics = child.get_acceleration_metrics()
            
            # Get available interactions
            available_interactions = child.get_available_interactions()
            
            # Display current state
            logger.info(f"\nChild State: {child_expression}")
            logger.info(f"Development Stage: {child.current_stage.name}")
            logger.info(f"Age (months): {child.age_months}")
            logger.info(f"Warning State: {warning_indicators['warning_state']}")
            
            # Display development metrics
            logger.info("\nDevelopment Metrics:")
            logger.info(f"Current Speed: {acceleration_metrics['current_multiplier']}x")
            logger.info(f"Max Safe Speed: {acceleration_metrics['max_safe_multiplier']}x")
            logger.info(f"Speed Locked: {'Yes' if child.speed_locked else 'No'}")
            logger.info(f"Stability Factor: {acceleration_metrics['stability_factor']:.2f}")
            logger.info(f"Warning Penalty: {acceleration_metrics['warning_penalty']:.2f}")
            
            # Display mother's interaction tab
            logger.info("\nMother's Interaction Tab:")
            logger.info("Available Categories:")
            for category, interactions in available_interactions.items():
                logger.info(f"\n{category}:")
                for interaction in interactions:
                    description = child.get_interaction_description(category, interaction)
                    cooldown = mother.get_interaction_cooldown()
                    status = "Ready" if cooldown == 0 else f"Cooldown: {cooldown}s"
                    logger.info(f"  - {interaction}: {description} ({status})")
            
            # Check interaction cooldown
            cooldown = mother.get_interaction_cooldown()
            if cooldown > 0:
                logger.info(f"\nInteraction cooldown: {cooldown} seconds remaining")
            
            # Generate mother's response
            response_text, mother_emotions = mother.generate_stimulus(
                child.current_stage, 
                child_expression
            )
            
            # Update child state
            child.update_emotions(mother_emotions)
            child.learn(response_text)
            
            # Check for development milestones
            if trainer.check_progress():
                child.age()
                logger.info(f"Child progressed to stage: {child.current_stage}")
            
            # Monitor system resources
            if psutil.virtual_memory().percent > 90:
                logger.warning("High memory usage detected")
                
            # Save state periodically
            child.save_state()
            
    except KeyboardInterrupt:
        logger.info("Simulation ended by user")
        child.save_state()  # Save state on exit
    except Exception as error:
        logger.error(f"Simulation error: {error}")
        child.save_state()  # Save state on error
        raise

if __name__ == "__main__":
    main()
