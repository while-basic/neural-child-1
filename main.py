# main.py
import torch
import time
from datetime import datetime
from typing import Dict, List, Any, Optional
from llm_module import chat_completion
from child_model import DynamicNeuralChild
from curriculum_manager import DevelopmentalStage, DevelopmentalSystem
from memory_module import DifferentiableMemory
from moral_network import MoralPolicyNetwork
from metacognition import MetacognitionSystem
from self_supervised_trainer import AutonomousTrainer
from text_embed import get_embeddings
from meta_learning import MetaLearningSystem, NeuralArchitectureSearch, GeneticOptimizer, create_model_from_architecture
from chat_interface import NeuralChildInterface
import numpy as np
import torch.nn as nn
from copy import deepcopy
import os
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import webbrowser
import random
from enum import Enum
import re
import logging
from ollama_client import OllamaConfig, OllamaClient

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # Output to console
        logging.FileHandler('neural_child.log')  # Output to file
    ]
)

# Create logger for this module
logger = logging.getLogger(__name__)

class DevelopmentalStage(Enum):
    EARLY_ELEMENTARY = "early_elementary"
    MIDDLE_ELEMENTARY = "middle_elementary"
    LATE_ELEMENTARY = "late_elementary"
    EARLY_ADOLESCENCE = "early_adolescence"
    MIDDLE_ADOLESCENCE = "middle_adolescence"
    LATE_ADOLESCENCE = "late_adolescence"
    YOUNG_ADULT = "young_adult"
    EARLY_ADULT = "early_adult"

class MotherLLM:
    def __init__(self):
        self.emotional_history = []
        self.stage_prompts = {
            DevelopmentalStage.EARLY_ELEMENTARY: """You are interacting with a child in early elementary stage (7-8 years).
                Focus on foundational learning, emotional support, and basic problem-solving.""",
            DevelopmentalStage.MIDDLE_ELEMENTARY: """You are interacting with a child in middle elementary stage (8-9 years).
                Encourage independent thinking, collaboration, and project-based learning.""",
            DevelopmentalStage.LATE_ELEMENTARY: """You are interacting with a child in late elementary stage (9-11 years).
                Foster analytical thinking, mentorship capabilities, and complex problem-solving.""",
            DevelopmentalStage.EARLY_ADOLESCENCE: """You are interacting with a child in early adolescence (11-13 years).
                Support identity development, abstract reasoning, and social-emotional growth.""",
            DevelopmentalStage.MIDDLE_ADOLESCENCE: """You are interacting with a teenager in middle adolescence (13-15 years).
                Focus on critical thinking, emotional intelligence, and developing personal values.""",
            DevelopmentalStage.LATE_ADOLESCENCE: """You are interacting with a teenager in late adolescence (15-17 years).
                Encourage advanced reasoning, career exploration, and preparation for independence.""",
            DevelopmentalStage.YOUNG_ADULT: """You are interacting with a young adult (17-19 years).
                Support transition to independence, complex decision-making, and life planning.""",
            DevelopmentalStage.EARLY_ADULT: """You are interacting with an early adult (19-21 years).
                Foster advanced cognitive development, professional growth, and personal autonomy."""
        }
        
        # Initialize Ollama client with custom configuration
        self.ollama_config = OllamaConfig(
            base_url="http://localhost:11434",
            default_model="mistral",
            fallback_model="llama2",
            temperature_range=(0.7, 0.9),  # Higher creativity for motherly responses
            max_tokens=150,  # Keep responses concise
            enable_response_caching=True,
            cache_duration_minutes=30,
            require_health_check=True
        )
        try:
            self.ollama_client = OllamaClient(self.ollama_config)
            logger.info("Successfully initialized Ollama client")
        except ConnectionError as e:
            logger.error(f"Failed to initialize Ollama client: {str(e)}")
            logger.info("Please ensure Ollama is running and has required models installed")
        except Exception as e:
            logger.error(f"Unexpected error initializing Ollama client: {str(e)}")
        
        # Initialize memory systems
        self.short_term_memory = []  # Recent interactions
        self.long_term_memory = {    # Core memories and patterns
            'behavioral_patterns': [],
            'emotional_milestones': [],
            'learning_achievements': [],
            'relationship_dynamics': [],
            'developmental_insights': [],
            'personal_info': [],      # Add category for personal information
            'emotional_memories': []   # Add category for emotional memories
        }
        self.memory_importance_threshold = 0.7
        self.max_short_term_memories = 50
        self.max_long_term_memories = 1000
        
        # Personal information storage
        self.personal_info = {
            'name': None,
            'age': None,
            'interests': set(),
            'relationships': {},
            'preferences': {},
            'achievements': set()
        }
        
        # Emotional state tracking
        self.current_emotion = {
            'type': 'NEUTRAL',
            'intensity': 0.5,
            'cause': None,
            'timestamp': datetime.now()
        }
        
        logger.info("MotherLLM initialized with Ollama integration")
        
    def _store_memory(self, memory_type: str, content: Dict[str, Any], importance: float):
        """Store a new memory with metadata"""
        timestamp = datetime.now()
        memory_entry = {
            'content': content,
            'importance': importance,
            'timestamp': timestamp,
            'type': memory_type,
            'context': {
                'emotional_state': content.get('emotional_state', None),
                'developmental_stage': content.get('developmental_stage', None)
            }
        }
        
        # Store in short-term memory
        self.short_term_memory.append(memory_entry)
        if len(self.short_term_memory) > self.max_short_term_memories:
            self.short_term_memory.pop(0)  # Remove oldest memory
            
        # Store important memories in long-term memory
        if importance >= self.memory_importance_threshold:
            category = self._determine_memory_category(content)
            if category:
                self.long_term_memory[category].append(memory_entry)
                # Maintain size limit for each category
                if len(self.long_term_memory[category]) > self.max_long_term_memories // 5:
                    # Remove least important memory
                    self.long_term_memory[category].sort(key=lambda x: x['importance'])
                    self.long_term_memory[category].pop(0)
    
    def _determine_memory_category(self, content: Dict[str, Any]) -> str:
        """Determine the appropriate category for a memory"""
        # Extract key indicators from content
        behavioral_keywords = ['behavior', 'action', 'reaction', 'response', 'did', 'tried', 'acted']
        emotional_keywords = ['emotion', 'feeling', 'mood', 'affect', 'happy', 'sad', 'angry', 'feel']
        learning_keywords = ['learn', 'understand', 'comprehend', 'achieve', 'study', 'practice', 'homework']
        relationship_keywords = ['interact', 'bond', 'connect', 'trust', 'friend', 'together', 'share']
        developmental_keywords = ['develop', 'grow', 'progress', 'milestone', 'change', 'improve']
        
        text = content.get('text', '').lower()
        
        # Count keyword matches for each category
        matches = {
            'behavioral_patterns': sum(1 for word in behavioral_keywords if word in text),
            'emotional_milestones': sum(1 for word in emotional_keywords if word in text),
            'learning_achievements': sum(1 for word in learning_keywords if word in text),
            'relationship_dynamics': sum(1 for word in relationship_keywords if word in text),
            'developmental_insights': sum(1 for word in developmental_keywords if word in text)
        }
        
        # Return category with most matches, or default if no matches
        max_matches = max(matches.values())
        if max_matches > 0:
            return max(matches.items(), key=lambda x: x[1])[0]
        return 'behavioral_patterns'  # Default category
    
    def _calculate_memory_importance(self, content: Dict[str, Any]) -> float:
        """Calculate the importance of a memory for storage decisions"""
        importance = 0.5  # Base importance
        
        # Emotional intensity increases importance
        if 'emotional_state' in content:
            emotional_intensity = content['emotional_state'].get('confidence', 0.5)
            importance += emotional_intensity * 0.2
            
        # Developmental significance increases importance
        if 'developmental_stage' in content:
            stage_transition = content.get('stage_transition', False)
            if stage_transition:
                importance += 0.3
                
        # Learning achievements increase importance
        if any(keyword in content.get('text', '').lower() 
               for keyword in ['learn', 'understand', 'achieve', 'milestone']):
            importance += 0.2
            
        # Behavioral changes increase importance
        if any(keyword in content.get('text', '').lower() 
               for keyword in ['change', 'different', 'new', 'first time']):
            importance += 0.2
            
        # Emotional content increases importance
        if any(keyword in content.get('text', '').lower() 
               for keyword in ['happy', 'sad', 'angry', 'feel', 'emotion']):
            importance += 0.2
            
        # Cap importance at 1.0
        return min(importance, 1.0)
    
    def _retrieve_relevant_memories(self, current_context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Retrieve memories relevant to the current context.
        
        Args:
            current_context (Dict[str, Any]): Current context including text and emotional state
            
        Returns:
            List[Dict[str, Any]]: List of relevant memories
        """
        relevant_memories = []
        search_text = current_context.get('text', '').lower()
        
        # Get recent short-term memories
        for memory in reversed(self.short_term_memory[-10:]):  # Check last 10 memories
            memory_text = str(memory['content'].get('text', '')).lower()
            if search_text in memory_text or memory_text in search_text:
                relevant_memories.append(memory)
        
        # Search through long-term memories
        for category in self.long_term_memory:
            category_memories = self.long_term_memory[category]
            for memory in category_memories:
                memory_text = str(memory['content'].get('text', '')).lower()
                memory_type = memory.get('type', '')
                
                # Check for text match
                if search_text in memory_text or memory_text in search_text:
                    relevant_memories.append(memory)
                    continue
                    
                # Check for emotional state match if present
                if ('emotional_state' in current_context and 
                    'emotional_state' in memory['content']):
                    current_state = current_context['emotional_state']
                    memory_state = memory['content']['emotional_state']
                    
                    if isinstance(current_state, torch.Tensor) and isinstance(memory_state, torch.Tensor):
                        if torch.allclose(current_state, memory_state, atol=0.3):
                            relevant_memories.append(memory)
                            continue
                            
                # Check for type match
                if memory_type and memory_type == current_context.get('type'):
                    relevant_memories.append(memory)
        
        # Sort by importance and recency
        relevant_memories.sort(
            key=lambda x: (x['importance'], x['timestamp']),
            reverse=True
        )
        
        # Return top 10 most relevant memories
        return relevant_memories[:10]
        
    def _extract_personal_info(self, text: str) -> Dict[str, Any]:
        """Extract personal information from text"""
        info = {}
        text = text.lower()
        
        # Extract name
        name_patterns = [
            r"(?:my name is|i am|i'm) (\w+)",
            r"(?:call me) (\w+)"
        ]
        for pattern in name_patterns:
            match = re.search(pattern, text)
            if match:
                info['name'] = match.group(1).capitalize()
                break
        
        # Extract age
        age_patterns = [
            r"(?:i am|i'm) (\d+)(?: years old)?",
            r"(\d+) years old",
            r"age(?:d)? (\d+)"
        ]
        for pattern in age_patterns:
            match = re.search(pattern, text)
            if match:
                try:
                    age = int(match.group(1))
                    if 0 <= age <= 150:  # Basic age validation
                        info['age'] = age
                except ValueError:
                    continue
                break
        
        # Extract interests
        interest_patterns = [
            r"(?:i (?:like|love|enjoy)) (.+?)(?:\.|\!|\n|$)",
            r"(?:my hobby is|my hobbies are) (.+?)(?:\.|\!|\n|$)",
            r"(?:i'm interested in|i am interested in) (.+?)(?:\.|\!|\n|$)"
        ]
        for pattern in interest_patterns:
            matches = re.finditer(pattern, text)
            interests = set()
            for match in matches:
                interest = match.group(1).strip()
                if interest:
                    interests.add(interest)
            if interests:
                info['interests'] = interests
                break
        
        return info
        
    def _update_personal_info(self, info: Dict[str, Any]):
        """Update stored personal information"""
        if 'name' in info and info['name']:
            self.personal_info['name'] = info['name']
        if 'age' in info and info['age']:
            self.personal_info['age'] = info['age']
        if 'interest' in info and info['interest']:
            self.personal_info['interests'].add(info['interest'])
            
    def _enhance_prompt_with_personal_info(self, prompt: str) -> str:
        """Enhance system prompt with personal information"""
        personal_context = "\nPersonal Information:"
        if self.personal_info['name']:
            personal_context += f"\n- Child's name: {self.personal_info['name']}"
        if self.personal_info['age']:
            personal_context += f"\n- Age: {self.personal_info['age']} years old"
        if self.personal_info['interests']:
            personal_context += f"\n- Interests: {', '.join(self.personal_info['interests'])}"
            
        return prompt + personal_context
        
    def _analyze_emotional_content(self, text: str) -> Dict[str, Any]:
        """Analyze text for emotional content"""
        # Emotion keywords and their intensities
        emotion_patterns = {
            'JOY': ['happy', 'joy', 'excited', 'wonderful', 'great', 'love', 'like', 'enjoy'],
            'SADNESS': ['sad', 'unhappy', 'cry', 'miss', 'lonely', 'sorry'],
            'ANGER': ['angry', 'mad', 'upset', 'frustrated', 'annoyed'],
            'FEAR': ['scared', 'afraid', 'worried', 'nervous', 'anxious'],
            'SURPRISE': ['wow', 'amazing', 'incredible', 'unexpected', 'surprised'],
            'PRIDE': ['proud', 'accomplished', 'achieved', 'success'],
            'CONCERN': ['concerned', 'care', 'worried about'],
            'EXCITEMENT': ['excited', 'thrilled', 'cant wait', 'looking forward']
        }
        
        text = text.lower()
        emotions_found = {}
        
        # Check for each emotion
        for emotion, keywords in emotion_patterns.items():
            intensity = sum(1 for word in keywords if word in text)
            if intensity > 0:
                emotions_found[emotion] = min(intensity * 0.3, 1.0)  # Scale intensity
                
        if emotions_found:
            # Get strongest emotion
            strongest_emotion = max(emotions_found.items(), key=lambda x: x[1])
            return {
                'type': strongest_emotion[0],
                'intensity': strongest_emotion[1],
                'cause': text,
                'timestamp': datetime.now()
            }
        return {
            'type': 'NEUTRAL',
            'intensity': 0.5,
            'cause': None,
            'timestamp': datetime.now()
        }
        
    def _generate_emotional_response(self, emotion: Dict[str, Any]) -> str:
        """Generate emotional expression based on current emotion"""
        intensity_expressions = {
            'low': ['slightly', 'a bit', 'somewhat'],
            'medium': ['quite', 'rather', 'notably'],
            'high': ['very', 'extremely', 'deeply']
        }
        
        # Get intensity level
        if emotion['intensity'] < 0.4:
            level = 'low'
        elif emotion['intensity'] < 0.7:
            level = 'medium'
        else:
            level = 'high'
            
        intensity_word = random.choice(intensity_expressions[level])
        
        # Emotional expressions
        expressions = {
            'JOY': [
                '*smiles warmly*',
                '*beams with happiness*',
                '*eyes light up with joy*',
                '*radiates pure delight*',
                '*glows with inner warmth*',
                '*expresses heartfelt joy*',
                '*shares a bright smile*'
            ],
            'SADNESS': [
                '*looks concerned*',
                '*shows empathy*',
                '*gives a comforting look*',
                '*offers gentle understanding*',
                '*shares in the feeling*',
                '*provides emotional support*',
                '*extends compassionate presence*'
            ],
            'ANGER': [
                '*shows understanding*',
                '*takes a calming breath*',
                '*maintains a supportive presence*',
                '*offers grounding energy*',
                '*provides steady calmness*',
                '*demonstrates patient acceptance*',
                '*shows balanced perspective*'
            ],
            'FEAR': [
                '*offers reassurance*',
                '*shows protective care*',
                '*provides gentle support*',
                '*creates safe space*',
                '*extends calming presence*',
                '*shares stabilizing energy*',
                '*offers grounding comfort*'
            ],
            'SURPRISE': [
                '*raises eyebrows with interest*',
                '*shows genuine amazement*',
                '*reacts with wonder*',
                '*expresses delighted astonishment*',
                '*shares in the revelation*',
                '*responds with fascination*',
                '*displays captivated interest*'
            ],
            'PRIDE': [
                '*beams with pride*',
                '*shows genuine admiration*',
                '*radiates approval*',
                '*expresses wholehearted recognition*',
                '*shares celebratory joy*',
                '*offers affirming presence*',
                '*glows with shared achievement*'
            ],
            'CONCERN': [
                '*shows caring attention*',
                '*leans in with concern*',
                '*offers supportive presence*',
                '*extends protective energy*',
                '*provides nurturing support*',
                '*demonstrates active care*',
                '*shares mindful attention*'
            ],
            'EXCITEMENT': [
                '*shares in the excitement*',
                '*shows enthusiastic interest*',
                '*responds with energy*',
                '*mirrors joyful anticipation*',
                '*expresses eager engagement*',
                '*radiates positive energy*',
                '*displays animated interest*'
            ],
            'NEUTRAL': [
                '*maintains warm presence*',
                '*shows attentive interest*',
                '*gives gentle attention*',
                '*offers balanced presence*',
                '*provides steady support*',
                '*demonstrates mindful awareness*',
                '*extends calm attention*'
            ]
        }
        
        expression = random.choice(expressions[emotion['type']])
        return f"{expression} {intensity_word} "
        
    def respond(self, message: str, context: Optional[List[Dict[str, str]]] = None) -> str:
        """Generate response using Ollama with proper error handling"""
        try:
            # Get development stage and appropriate prompt
            stage = self._determine_stage(context)
            stage_prompt = self.stage_prompts.get(stage, self.stage_prompts[DevelopmentalStage.EARLY_ELEMENTARY])
            
            # Enhance prompt with personal info and emotional context
            enhanced_prompt = self._enhance_prompt_with_personal_info(stage_prompt)
            
            # Construct full prompt with context
            full_prompt = f"{enhanced_prompt}\n\nChild's message: {message}\n\nRespond as a caring mother:"
            
            # Get quantum metrics before response
            pre_quantum_metrics = self.ollama_client.get_quantum_metrics()
            logger.debug(f"Pre-response quantum metrics: {pre_quantum_metrics}")
            
            # Generate response using Ollama
            try:
                response_data = self.ollama_client.generate_response(
                    prompt=full_prompt,
                    temperature=0.8  # Slightly higher for more nurturing responses
                )
                response = response_data['response']
                
                # Get quantum metrics after response
                post_quantum_metrics = self.ollama_client.get_quantum_metrics()
                logger.debug(f"Post-response quantum metrics: {post_quantum_metrics}")
                
                # Update emotional state based on response
                self._update_emotional_state(message, response)
                
                # Store interaction in memory
                self._store_memory({
                    'text': message,
                    'response': response,
                    'quantum_metrics': {
                        'pre': pre_quantum_metrics,
                        'post': post_quantum_metrics
                    },
                    'timestamp': datetime.now()
                }, importance=0.8)
                
                return response
                
            except ConnectionError:
                logger.error("Could not connect to Ollama service")
                return "*Mother is temporarily unavailable. Please ensure Ollama is running.*"
                
            except Exception as e:
                logger.error(f"Error generating response: {str(e)}")
                return "*Mother is processing your message carefully...*"
                
        except Exception as e:
            logger.error(f"Critical error in respond method: {str(e)}")
            return "*Mother needs a moment to think...*"
        
    def _update_emotional_state(self, message: str, response: str) -> None:
        """Update emotional state based on interaction"""
        # Analyze emotional content
        emotional_analysis = self._analyze_emotional_content(message)
        
        # Update current emotion
        self.current_emotion = {
            'type': emotional_analysis['type'],
            'intensity': emotional_analysis['intensity'],
            'cause': message,
            'timestamp': datetime.now()
        }
        
        # Get quantum emotional influence
        quantum_metrics = self.ollama_client.get_quantum_metrics()
        
        # Adjust emotional intensity based on quantum coherence
        self.current_emotion['intensity'] *= quantum_metrics['coherence']
        
        logger.debug(f"Updated emotional state: {self.current_emotion}")

    def _determine_stage(self, context: list) -> DevelopmentalStage:
        """Determine the appropriate developmental stage based on interaction context"""
        if not context:
            return DevelopmentalStage.EARLY_ELEMENTARY
            
        # Analyze context to determine stage
        # This is a simplified version - in practice, you'd want more sophisticated analysis
        complexity = sum(len(msg["content"].split()) for msg in context) / len(context)
        
        if complexity > 50:
            return DevelopmentalStage.EARLY_ADOLESCENCE
        elif complexity > 30:
            return DevelopmentalStage.LATE_ELEMENTARY
        elif complexity > 20:
            return DevelopmentalStage.MIDDLE_ELEMENTARY
        else:
            return DevelopmentalStage.EARLY_ELEMENTARY

class DigitalChild:
    def __init__(self):
        self.brain = DynamicNeuralChild()
        self.memory = DifferentiableMemory()
        self.morality = MoralPolicyNetwork(device=self.brain.device)
        self.metacognition = MetacognitionSystem()
        self.curriculum = DevelopmentalSystem()
        self.trainer = AutonomousTrainer(self.brain, self.memory, self.morality)
        self.mother = MotherLLM()
        self.birth_date = datetime.now()
        self.emotional_state = torch.zeros(4, device='cuda')
        
        # Initialize meta-learning components
        self.meta_learner = MetaLearningSystem(self.brain)
        self.nas = NeuralArchitectureSearch(
            input_size=768,  # Matches embedding size
            output_size=4,   # Matches emotional state dimensions (joy, trust, fear, surprise)
            max_layers=5
        )
        self.genetic_optimizer = GeneticOptimizer(
            population_size=10,
            mutation_rate=0.1,
            crossover_rate=0.7
        )
        
        # Initialize model population for genetic evolution
        self.model_population = self.genetic_optimizer.initialize_population(self.brain)
        self.population_fitness = [0.0] * len(self.model_population)
        
        # Track architecture and evolution metrics
        self.architecture_history = []
        self.evolution_metrics = []
        
        # Initialize quantum emotional processing
        self.quantum_emotional_state = {
            'superposition': torch.zeros(8, device='cuda'),  # Extended emotional dimensions
            'entanglement_matrix': torch.eye(8, device='cuda'),  # Emotional entanglement
            'coherence_factor': 1.0,  # Quantum coherence of emotional states
            'collapse_threshold': 0.7  # Threshold for emotional state collapse
        }
        
        # Track quantum metrics
        self.quantum_metrics = {
            'coherence_history': [],
            'entanglement_strength': [],
            'superposition_stability': []
        }
        
        # Initialize neural evolution components
        self.neural_evolution = {
            'growth_rate': 0.1,
            'pruning_threshold': 0.3,
            'mutation_probability': 0.05,
            'architecture_complexity': 1.0,
            'adaptation_score': 0.0
        }
        
        # Track evolution metrics
        self.evolution_tracking = {
            'architecture_changes': [],
            'complexity_scores': [],
            'adaptation_history': [],
            'mutation_events': []
        }

    def get_development_stage(self) -> str:
        """Get the current developmental stage based on age and metrics"""
        age = self.get_age()
        
        # Get additional metrics for more accurate stage determination
        emotional_maturity = np.mean([
            float(self.emotional_state[0]),  # Joy
            float(self.emotional_state[1]),  # Trust
            1.0 - float(self.emotional_state[2]),  # Inverse of Fear
            float(self.emotional_state[3])   # Surprise (curiosity)
        ])
        
        # Get quantum coherence as a measure of stability
        quantum_coherence = self.quantum_emotional_state['coherence_factor']
        
        # Get neural complexity
        neural_complexity = self.neural_evolution['architecture_complexity']
        
        # Calculate overall development score
        development_score = (
            0.4 * age +  # Age is the primary factor
            0.3 * emotional_maturity +  # Emotional development
            0.2 * quantum_coherence +  # Stability
            0.1 * neural_complexity    # Neural development
        )
        
        # Map development score to stages
        if development_score < 8.0:
            return "EARLY_ELEMENTARY"
        elif development_score < 9.0:
            return "MIDDLE_ELEMENTARY"
        elif development_score < 11.0:
            return "LATE_ELEMENTARY"
        elif development_score < 13.0:
            return "EARLY_ADOLESCENCE"
        elif development_score < 15.0:
            return "MIDDLE_ADOLESCENCE"
        elif development_score < 17.0:
            return "LATE_ADOLESCENCE"
        elif development_score < 19.0:
            return "YOUNG_ADULT"
        else:
            return "EARLY_ADULT"

    def get_age(self) -> float:
        """Get the current age in years"""
        age_in_days = (datetime.now() - self.birth_date).days
        return age_in_days / 365.0  # Convert days to years

    def age(self) -> int:
        """Get age in months (legacy method)"""
        return (datetime.now() - self.birth_date).days // 30  # months

    def update_emotions(self, mother_vector):
        """Update emotional state without recursive calls"""
        if isinstance(mother_vector, dict):
            mother_vector = torch.tensor([
                mother_vector.get('joy', 0.5),
                mother_vector.get('trust', 0.5),
                mother_vector.get('fear', 0.0),
                mother_vector.get('surprise', 0.2)
            ], device='cuda')
        
        # Update brain's emotional state first
        brain_update = self.brain.update_emotions(mother_vector)
        
        # Then update child's emotional state
        delta = mother_vector - self.emotional_state
        self.emotional_state += 0.3 * delta + 0.1 * torch.randn_like(delta)
        self.emotional_state = torch.clamp(self.emotional_state, 0, 1)
        
        return brain_update
    
    def express_feeling(self):
        """Express feelings without recursive calls"""
        # Use brain's emotional state directly
        return self.brain.express_feeling()
    
    def perceive(self, stimulus):
        """Perceive stimulus with proper error handling"""
        try:
            if not isinstance(stimulus, dict) or 'text' not in stimulus:
                return torch.zeros(1, 768, device='cuda')
                
            embeddings = get_embeddings(stimulus['text'])
            if not embeddings:
                return torch.zeros(1, 768, device='cuda')
                
            embedding_vector = embeddings[0]['embedding'] if embeddings else [0] * 768
            return torch.tensor(embedding_vector, device='cuda').unsqueeze(0)
        except Exception as e:
            print(f"Error in perception: {e}")
            return torch.zeros(1, 768, device='cuda')
        
    def respond(self, perception):
        with torch.amp.autocast("cuda"):
            return self.brain(perception)
    
    def learn(self, mother_feedback):
        # Record experience in memory
        self.memory.record_experience(
            mother_feedback['input'],
            mother_feedback['internal_state'],
            mother_feedback['reward'],
            time.time(),
            self.emotional_state  
        )
        
        try:
            # Process quantum emotions
            quantum_emotional_state = self._process_quantum_emotions(
                torch.tensor(mother_feedback['reward'], device=self.brain.device)
            )
            
            # Evolve neural architecture
            evolution_metrics = self._evolve_neural_architecture()
            
            # Ensure input has correct shape before training step
            input_tensor = mother_feedback['input']
            
            # Debug print
            print(f"Input tensor shape before projection: {input_tensor.size()}")
            print(f"Quantum emotional state: {quantum_emotional_state}")
            print(f"Evolution metrics: {evolution_metrics}")
            
            if input_tensor.size(1) == 4:  # If emotional input
                input_tensor = self.brain.emotion_projection_layer(input_tensor)
                print(f"Input tensor shape after projection: {input_tensor.size()}")
            
            # Verify input shape matches expected
            if input_tensor.size(1) != self.brain.input_size:
                print(f"Shape mismatch: got {input_tensor.size()}, expected {self.brain.input_size}")
                input_tensor = torch.zeros((1, self.brain.input_size), device=self.brain.device)
            
            # Enhanced training step with quantum and evolution integration
            loss = self.trainer.training_step(
                input_tensor, 
                quantum_state=quantum_emotional_state,
                architecture_complexity=evolution_metrics['complexity']
            )
            
            # Meta-learning update with enhanced metrics
            meta_loss = self.meta_learner.meta_update({
                'reward_score': mother_feedback['reward'],
                'success_metric': mother_feedback.get('success_metric', 0.5),
                'complexity_rating': mother_feedback.get('complexity_rating', 0.3),
                'quantum_coherence': self.quantum_emotional_state['coherence_factor'],
                'architecture_adaptation': evolution_metrics['adaptation_score']
            })
            
            return loss
            
        except Exception as e:
            print(f"Detailed error in learn step: {str(e)}")
            print(f"Input tensor shape: {input_tensor.size() if isinstance(input_tensor, torch.Tensor) else 'Not a tensor'}")
            return torch.tensor(0.0, device=self.brain.device)
    
    def _transfer_knowledge(self, old_model: nn.Module, new_model: nn.Module):
        """Transfer learned knowledge from old architecture to new one"""
        # Extract embeddings of key experiences
        key_experiences = self.memory.get_important_experiences(10)
        old_embeddings = []
        
        with torch.no_grad():
            for exp in key_experiences:
                old_embeddings.append(old_model(exp['input']))
        
        # Fine-tune new model to match old responses
        optimizer = torch.optim.Adam(new_model.parameters())
        for _ in range(100):  # Quick fine-tuning
            for exp, old_emb in zip(key_experiences, old_embeddings):
                optimizer.zero_grad()
                new_emb = new_model(exp['input'])
                loss = nn.MSELoss()(new_emb, old_emb)
                loss.backward()
                optimizer.step()

    def _process_quantum_emotions(self, stimulus_vector):
        """Process emotions using quantum-inspired algorithms"""
        # Update superposition state
        self.quantum_emotional_state['superposition'] = torch.nn.functional.softmax(
            stimulus_vector @ self.quantum_emotional_state['entanglement_matrix'], 
            dim=0
        )
        
        # Apply quantum noise (decoherence)
        noise = torch.randn_like(self.quantum_emotional_state['superposition']) * (1 - self.quantum_emotional_state['coherence_factor'])
        self.quantum_emotional_state['superposition'] += noise
        
        # Update coherence
        self.quantum_emotional_state['coherence_factor'] *= 0.99  # Gradual decoherence
        
        # Check for emotional collapse
        if torch.max(self.quantum_emotional_state['superposition']) > self.quantum_emotional_state['collapse_threshold']:
            # Collapse to classical emotional state
            classical_state = torch.zeros_like(self.quantum_emotional_state['superposition'])
            max_idx = torch.argmax(self.quantum_emotional_state['superposition'])
            classical_state[max_idx] = 1.0
            self.quantum_emotional_state['superposition'] = classical_state
            
        # Update metrics
        self.quantum_metrics['coherence_history'].append(self.quantum_emotional_state['coherence_factor'])
        self.quantum_metrics['entanglement_strength'].append(
            torch.trace(self.quantum_emotional_state['entanglement_matrix']).item()
        )
        self.quantum_metrics['superposition_stability'].append(
            torch.std(self.quantum_emotional_state['superposition']).item()
        )
        
        return self.quantum_emotional_state['superposition']

    def _evolve_neural_architecture(self):
        """Dynamically evolve neural architecture based on performance and needs"""
        current_performance = self.meta_learner.get_performance_metrics()
        
        # Calculate adaptation need
        adaptation_need = 1.0 - current_performance['success_rate']
        
        # Update architecture complexity
        if adaptation_need > self.neural_evolution['pruning_threshold']:
            # Grow network
            self.neural_evolution['architecture_complexity'] *= (1 + self.neural_evolution['growth_rate'])
            self.evolution_tracking['architecture_changes'].append(('growth', datetime.now()))
            # Add new layer
            self.brain.add_layer()
        else:
            # Prune network
            self.neural_evolution['architecture_complexity'] *= (1 - self.neural_evolution['growth_rate'])
            self.evolution_tracking['architecture_changes'].append(('pruning', datetime.now()))
            
        # Apply random mutations
        if random.random() < self.neural_evolution['mutation_probability']:
            mutation_type = random.choice(['layer_modification', 'activation_mutation'])
            self.evolution_tracking['mutation_events'].append((mutation_type, datetime.now()))
            
            if mutation_type == 'layer_modification':
                # Modify random layer
                modifiable_layers = [i for i, layer in enumerate(self.brain.layers) 
                                   if isinstance(layer, nn.Linear)]
                if modifiable_layers:
                    layer_idx = random.choice(modifiable_layers)
                    self.brain.modify_layer(layer_idx)
            elif mutation_type == 'activation_mutation':
                # Mutate activation functions
                activation_layers = [i for i, layer in enumerate(self.brain.layers) 
                                   if isinstance(layer, (nn.ReLU, nn.GELU, nn.Tanh, nn.LeakyReLU))]
                if activation_layers:
                    layer_idx = random.choice(activation_layers)
                    new_activation = random.choice([
                        nn.ReLU(),
                        nn.LeakyReLU(),
                        nn.GELU(),
                        nn.SiLU()
                    ]).to(self.brain.device)
                    self.brain.layers[layer_idx] = new_activation
                
        # Update tracking metrics
        self.evolution_tracking['complexity_scores'].append(
            self.neural_evolution['architecture_complexity']
        )
        self.evolution_tracking['adaptation_history'].append(
            current_performance['success_rate']
        )
        
        return {
            'complexity': self.neural_evolution['architecture_complexity'],
            'adaptation_score': current_performance['success_rate'],
            'mutations': len(self.evolution_tracking['mutation_events'])
        }

def main():
    print("Initializing Neural Child Development System...")
    
    # Initialize core components
    child = DigitalChild()
    mother = MotherLLM()
    
    # Create and launch interface
    interface = NeuralChildInterface(child, mother)
    
    # Set up the interface with local URL
    local_url = "http://localhost:7860"
    
    # Launch the interface
    print("\nLaunching interface...")
    interface_server = interface.create_interface().launch(
        server_name="0.0.0.0",
        server_port=7860,
        quiet=True,  # Reduces console output
        show_error=True,
        share=False  # Set to True if you want a public URL
    )
    
    # Wait a moment for server to start
    time.sleep(2)
    
    # Automatically open in default browser
    print(f"\nOpening interface in your default browser: {local_url}")
    webbrowser.open(local_url)
    
    print("\nNeural Child Development System is ready!")
    print("Press Ctrl+C to shut down the system.")
    
    try:
        # Keep the server running
        interface_server.block_thread()
    except KeyboardInterrupt:
        print("\nShutting down Neural Child Development System...")
        interface_server.close()
        print("System shutdown complete.")

if __name__ == "__main__":
    main()
