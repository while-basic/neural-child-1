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
            
            response = chat_completion(
                system_prompt=full_prompt,
                user_prompt=user_input,
                structured_output=True
            )
            
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
            state_dict = torch.load('checkpoints/child_state.pt', map_location=self.device)
            
            # Load neural model state
            self.neural_model.load_state_dict(state_dict['neural_model'])
            
            # Load developmental state
            self.current_stage = state_dict.get('current_stage', DevelopmentalStage.NEWBORN)
            self.age_months = state_dict.get('age_months', 0)
            
            # Load curriculum state if available
            if 'curriculum' in state_dict:
                self.curriculum.load_state(state_dict['curriculum'])
            
            # Ensure curriculum stage matches child's stage
            self.curriculum.update_stage(self.current_stage)
            
            # Load emotional state if available
            if 'emotional_state' in state_dict:
                emotional_data = state_dict['emotional_state']
                self.emotional_state = EmotionalState(
                    happiness=emotional_data.get('happiness', 0.5),
                    sadness=emotional_data.get('sadness', 0.5),
                    anger=emotional_data.get('anger', 0.5),
                    fear=emotional_data.get('fear', 0.5),
                    surprise=emotional_data.get('surprise', 0.5),
                    disgust=emotional_data.get('disgust', 0.0),
                    trust=emotional_data.get('trust', 0.5),
                    anticipation=emotional_data.get('anticipation', 0.5)
                )
            
            logger.info("Successfully loaded model state")
        except FileNotFoundError:
            logger.info("No existing state found. Starting fresh.")
            self._initialize_fresh_state()
        except Exception as error:
            logger.error(f"Error loading state: {error}")
            raise
    
    def _initialize_fresh_state(self) -> None:
        """Initialize a fresh state for the model."""
        logger.info("Initializing fresh model state")
        self.current_stage = DevelopmentalStage.NEWBORN
        self.age_months = 0
        self.emotional_state = EmotionalState(0.5, 0.5, 0.5, 0.5, 0.5, 0.0, 0.5, 0.5)
        self.curriculum.reset()  # Reset curriculum to initial state
        self.curriculum.update_stage(self.current_stage)  # Ensure stage is synced
    
    def save_state(self) -> None:
        """Save the current model state."""
        state_dict = {
            'neural_model': self.neural_model.state_dict(),
            'current_stage': self.current_stage,
            'age_months': self.age_months,
            'curriculum': self.curriculum.get_state(),
            'emotional_state': {
                'happiness': self.emotional_state.happiness,
                'sadness': self.emotional_state.sadness,
                'anger': self.emotional_state.anger,
                'fear': self.emotional_state.fear,
                'surprise': self.emotional_state.surprise,
                'disgust': self.emotional_state.disgust,
                'trust': self.emotional_state.trust,
                'anticipation': self.emotional_state.anticipation
            }
        }
        
        os.makedirs('checkpoints', exist_ok=True)
        torch.save(state_dict, 'checkpoints/child_state.pt')
        logger.info("Successfully saved model state")
    
    def update_emotions(self, mother_response: EmotionalState) -> None:
        """Update emotional state based on mother's response.
        
        Args:
            mother_response: Emotional state from mother's response
        """
        # Use emotional regulation to modulate response
        regulated_response = self.emotional_regulation.process(
            current_state=self.emotional_state,
            input_state=mother_response
        )
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
        """Progress the child's age and check for stage transitions."""
        self.age_months += 1
        if self.developmental_system.check_stage_progression({'age': self.age_months}):
            self.developmental_system.progress_stage()
            self.current_stage = self.developmental_system.current_stage
            self.curriculum.update_stage(self.current_stage)  # Update curriculum stage
            self.save_state()  # Save state after stage progression

def main() -> None:
    """Main function to run the Neural Child simulation."""
    try:
        mother = MotherLLM()
        child = DigitalChild()
        trainer = DevelopmentalTrainer(child)
        
        logger.info("Starting Neural Child simulation...")
        
        # Main interaction loop
        while True:
            child_expression = child.express_feeling()
            response_text, mother_emotions = mother.generate_stimulus(
                child.current_stage, 
                child_expression
            )
            
            child.update_emotions(mother_emotions)
            child.learn(response_text)
            
            # Check for development milestones
            if trainer.check_progress():
                child.age()
                logger.info(f"Child progressed to stage: {child.current_stage}")
            
            # Monitor system resources
            if psutil.virtual_memory().percent > 90:
                logger.warning("High memory usage detected")
                
    except KeyboardInterrupt:
        logger.info("Simulation ended by user")
    except Exception as error:
        logger.error(f"Simulation error: {error}")
        raise

if __name__ == "__main__":
    main()
