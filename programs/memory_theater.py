"""
memory_theater.py
Created: 2024-03-21
Description: A system for transforming neural child memories and dreams into interactive dramatic scenarios.

Author: Dr. Celaya
Project: Neural Child + Meta-Learning
"""

import torch
import numpy as np
import plotly.graph_objects as go
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import logging
import requests
import json
import time
from dataclasses import dataclass
from enum import Enum
import random
import gradio as gr
from child_model import DynamicNeuralChild

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class Scene:
    """Structure for a dramatic scene"""
    title: str
    setting: str
    characters: List[str]
    emotional_tone: torch.Tensor
    narrative: str
    source_memories: List[str]
    symbolic_elements: List[str]
    intensity: float
    timestamp: datetime

@dataclass
class Performance:
    """Structure for a complete theatrical performance"""
    title: str
    scenes: List[Scene]
    theme: str
    emotional_arc: List[torch.Tensor]
    duration: float
    insights: List[str]
    audience_reactions: List[str]
    timestamp: datetime

class StoryElement(Enum):
    """Different types of story elements"""
    CHARACTER = "Character"
    SETTING = "Setting"
    CONFLICT = "Conflict"
    RESOLUTION = "Resolution"
    SYMBOL = "Symbol"
    EMOTION = "Emotion"

class OllamaAPI:
    """Interface for Ollama API interactions"""
    def __init__(self, 
                 base_url: str = "http://localhost:11434",
                 max_retries: int = 3,
                 timeout: int = 30,
                 backoff_factor: float = 1.5):
        self.base_url = base_url
        self.max_retries = max_retries
        self.timeout = timeout
        self.backoff_factor = backoff_factor
        self.session = self._create_session()
        
    def _create_session(self) -> requests.Session:
        """Create a session with retry configuration"""
        session = requests.Session()
        
        # Configure retry strategy
        retries = requests.adapters.Retry(
            total=self.max_retries,
            backoff_factor=self.backoff_factor,
            status_forcelist=[408, 429, 500, 502, 503, 504],
            allowed_methods=["POST", "GET"]
        )
        
        # Add retry adapter to session
        adapter = requests.adapters.HTTPAdapter(max_retries=retries)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        return session
        
    def generate_scene(self, 
                      prompt: str, 
                      model: str = "artifish/llama3.2-uncensored",
                      temperature: float = 0.85) -> str:
        """Generate theatrical scene content using Ollama with robust error handling"""
        for attempt in range(self.max_retries):
            try:
                # Calculate timeout with exponential backoff
                current_timeout = self.timeout * (self.backoff_factor ** attempt)
                
                logger.info(f"Attempting scene generation (attempt {attempt + 1}/{self.max_retries})")
                
                # Check API health first
                if not self._check_api_health():
                    logger.error("Ollama API is not healthy")
                    time.sleep(attempt * 2)  # Wait before retry
                    continue
                
                # Make the API call
                response = self.session.post(
                    f"{self.base_url}/api/generate",
                    json={
                        "model": model,
                        "prompt": prompt,
                        "temperature": temperature,
                        "max_tokens": 300,
                        "stream": False
                    },
                    timeout=current_timeout
                )
                response.raise_for_status()
                
                result = response.json()
                if "response" in result:
                    return result["response"].strip()
                else:
                    logger.error(f"Unexpected response format: {result}")
                    
            except requests.Timeout:
                logger.warning(f"Timeout on attempt {attempt + 1} (timeout={current_timeout}s)")
                if attempt == self.max_retries - 1:
                    return self._generate_fallback_scene()
                    
            except requests.ConnectionError:
                logger.error(f"Connection error on attempt {attempt + 1}")
                if attempt == self.max_retries - 1:
                    return self._generate_fallback_scene()
                time.sleep(attempt * 2)
                
            except Exception as e:
                logger.error(f"Unexpected error on attempt {attempt + 1}: {str(e)}")
                if attempt == self.max_retries - 1:
                    return self._generate_fallback_scene()
                time.sleep(attempt * 2)
                
        return self._generate_fallback_scene()
        
    def _check_api_health(self) -> bool:
        """Check if Ollama API is healthy and responding"""
        try:
            response = self.session.get(
                f"{self.base_url}/api/tags",
                timeout=5
            )
            return response.status_code == 200
        except:
            return False
            
    def _generate_fallback_scene(self) -> str:
        """Generate a simple fallback scene when API fails"""
        return """TITLE: A Moment of Imagination
SETTING: A peaceful garden
CHARACTERS: Child, Friendly Bird, Gentle Breeze
SYMBOLS: Garden, Bird, Wind

The child sits quietly in a peaceful garden. A small bird hops nearby, tilting its head curiously.
The gentle breeze rustles the leaves, creating a soothing melody.

CHILD: (smiling softly) "Hello, little bird. Would you like to be my friend?"

The bird chirps melodiously in response, and the breeze carries the sweet scent of flowers.
Together, they share a moment of tranquil connection in the garden."""
            
    def __del__(self):
        """Clean up resources"""
        if hasattr(self, 'session'):
            self.session.close()

class MemoryTheater:
    """Core system for creating theatrical performances from memories"""
    def __init__(self, child: DynamicNeuralChild):
        self.child = child
        self.ollama = OllamaAPI()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.performances: List[Performance] = []
        self.current_performance: Optional[Performance] = None
        self.emotional_history: List[torch.Tensor] = []
        
    def create_scene(self, memories: List[str], emotional_state: torch.Tensor) -> Scene:
        """Create a new dramatic scene from memories"""
        # Generate scene prompt
        prompt = self._create_scene_prompt(memories, emotional_state)
        
        # Generate scene content
        content = self.ollama.generate_scene(prompt)
        
        # Parse scene elements
        try:
            lines = content.split("\n")
            title = ""
            setting = ""
            characters = []
            narrative = ""
            symbols = []
            
            for line in lines:
                if line.startswith("TITLE:"):
                    title = line.replace("TITLE:", "").strip()
                elif line.startswith("SETTING:"):
                    setting = line.replace("SETTING:", "").strip()
                elif line.startswith("CHARACTERS:"):
                    characters = [c.strip() for c in line.replace("CHARACTERS:", "").split(",")]
                elif line.startswith("SYMBOLS:"):
                    symbols = [s.strip() for s in line.replace("SYMBOLS:", "").split(",")]
                else:
                    narrative += line + "\n"
            
            # Create scene object
            scene = Scene(
                title=title or "Untitled Scene",
                setting=setting or "Unknown Setting",
                characters=characters,
                emotional_tone=emotional_state,
                narrative=narrative.strip(),
                source_memories=memories,
                symbolic_elements=symbols,
                intensity=torch.norm(emotional_state).item(),
                timestamp=datetime.now()
            )
            
            return scene
            
        except Exception as e:
            logger.error(f"Error creating scene: {str(e)}")
            return None
            
    def start_performance(self, theme: str, num_scenes: int = 3) -> Performance:
        """Begin a new theatrical performance"""
        try:
            # Initialize new performance
            self.current_performance = Performance(
                title=f"Performance: {theme}",
                scenes=[],
                theme=theme,
                emotional_arc=[],
                duration=0.0,
                insights=[],
                audience_reactions=[],
                timestamp=datetime.now()
            )
            
            # Generate scenes
            start_time = time.time()
            
            for i in range(num_scenes):
                # Select memories for scene
                memories = self._select_memories_for_scene(theme)
                
                # Calculate emotional state for scene
                emotional_state = self._calculate_scene_emotions(i, num_scenes)
                
                # Create scene
                scene = self.create_scene(memories, emotional_state)
                if scene:
                    self.current_performance.scenes.append(scene)
                    self.current_performance.emotional_arc.append(emotional_state)
                    
                    # Generate audience reaction
                    reaction = self._generate_audience_reaction(scene)
                    self.current_performance.audience_reactions.append(reaction)
                    
                    # Update child's emotional state based on scene
                    self.child.update_emotions(scene.emotional_tone)
                    
            # Calculate performance duration
            self.current_performance.duration = time.time() - start_time
            
            # Generate performance insights
            self.current_performance.insights = self._generate_performance_insights()
            
            # Store performance
            self.performances.append(self.current_performance)
            
            return self.current_performance
            
        except Exception as e:
            logger.error(f"Error in performance: {str(e)}")
            return None
            
    def _create_scene_prompt(self, memories: List[str], emotional_state: torch.Tensor) -> str:
        """Create prompt for scene generation"""
        emotions = emotional_state.cpu().tolist()
        
        prompt = f"""You are a creative director helping a {self.child.age}-year-old child transform their memories into a theatrical scene.

Current emotional state:
- Joy: {emotions[0]:.2f}
- Trust: {emotions[1]:.2f}
- Fear: {emotions[2]:.2f}
- Surprise: {emotions[3]:.2f}

Memories to incorporate:
{chr(10).join(f"- {memory}" for memory in memories)}

Create a dramatic scene that transforms these memories into an imaginative theatrical scenario.
The scene should be age-appropriate and reflect the emotional state.

Format the scene as follows:
TITLE: [scene title]
SETTING: [scene setting]
CHARACTERS: [list of characters]
SYMBOLS: [list of symbolic elements]

[scene narrative in present tense, including dialogue and stage directions]
"""
        
        return prompt
        
    def _select_memories_for_scene(self, theme: str) -> List[str]:
        """Select relevant memories for a scene based on theme"""
        # Get recent memories from child
        all_memories = []  # In real implementation, get from child's memory system
        
        # For demonstration, generate some example memories
        example_memories = [
            "Playing with colorful blocks in the sunlight",
            "Sharing cookies with a new friend at playtime",
            "Watching butterflies in the garden",
            "Drawing pictures of a magical forest",
            "Learning to count stars at night"
        ]
        
        # Select 2-3 memories randomly for now
        return random.sample(example_memories, random.randint(2, 3))
        
    def _calculate_scene_emotions(self, scene_index: int, total_scenes: int) -> torch.Tensor:
        """Calculate emotional state for a scene based on its position in the performance"""
        # Start with child's current emotional state
        base_emotions = self.child.emotional_state.clone()
        
        # Add dramatic arc influence
        progress = scene_index / (total_scenes - 1)
        if progress < 0.3:
            # Rising action - increase surprise and joy
            base_emotions[3] += 0.2  # Surprise
            base_emotions[0] += 0.1  # Joy
        elif progress < 0.7:
            # Climax - increase intensity of all emotions
            base_emotions *= 1.3
        else:
            # Resolution - increase trust, decrease fear
            base_emotions[1] += 0.2  # Trust
            base_emotions[2] *= 0.7  # Fear
            
        # Add random variation
        variation = torch.rand_like(base_emotions) * 0.1
        base_emotions += variation
        
        # Normalize
        return torch.clamp(base_emotions, 0, 1)
        
    def _generate_audience_reaction(self, scene: Scene) -> str:
        """Generate simulated audience reaction to a scene"""
        emotional_intensity = scene.intensity
        
        reactions = [
            "The audience watches with rapt attention",
            "Gentle smiles spread across the audience",
            "A sense of wonder fills the room",
            "The audience leans forward in anticipation",
            "Quiet gasps of surprise can be heard"
        ]
        
        return random.choice(reactions)
        
    def _generate_performance_insights(self) -> List[str]:
        """Generate insights about the overall performance"""
        if not self.current_performance or not self.current_performance.scenes:
            return []
            
        insights = []
        
        # Analyze emotional arc
        if len(self.current_performance.emotional_arc) > 1:
            start_emotions = self.current_performance.emotional_arc[0]
            end_emotions = self.current_performance.emotional_arc[-1]
            
            emotional_change = end_emotions - start_emotions
            
            # Generate insights based on emotional changes
            for i, (emotion, change) in enumerate(zip(['joy', 'trust', 'fear', 'surprise'], emotional_change)):
                if abs(change.item()) > 0.2:
                    direction = "increased" if change.item() > 0 else "decreased"
                    insights.append(f"The child's {emotion} {direction} through the performance")
                    
        # Analyze symbolic elements
        all_symbols = []
        for scene in self.current_performance.scenes:
            all_symbols.extend(scene.symbolic_elements)
            
        if all_symbols:
            # Find most common symbols
            from collections import Counter
            symbol_counts = Counter(all_symbols)
            common_symbols = symbol_counts.most_common(2)
            
            for symbol, count in common_symbols:
                insights.append(f"The symbol '{symbol}' appeared {count} times, suggesting its significance")
                
        return insights

class TheaterInterface:
    """Interface for the Memory Theater system"""
    def __init__(self, child: DynamicNeuralChild):
        self.theater = MemoryTheater(child)
        
    def create_interface(self) -> gr.Blocks:
        """Create the Gradio interface for the Memory Theater"""
        with gr.Blocks(title="Neural Child Memory Theater") as interface:
            gr.Markdown(
                """
                # 🎭 Neural Child Memory Theater
                Watch as memories transform into dramatic performances in the child's imagination.
                """
            )
            
            with gr.Row():
                with gr.Column(scale=2):
                    # Performance controls
                    theme_input = gr.Textbox(
                        label="Performance Theme",
                        placeholder="Enter a theme for the performance..."
                    )
                    num_scenes = gr.Slider(
                        minimum=1,
                        maximum=5,
                        value=3,
                        step=1,
                        label="Number of Scenes"
                    )
                    start_btn = gr.Button("Begin Performance 🎬", variant="primary")
                    
                with gr.Column(scale=2):
                    # Performance display
                    current_scene = gr.Textbox(
                        label="Current Scene",
                        value="Waiting to begin...",
                        lines=10
                    )
                    
            with gr.Row():
                # Emotional arc visualization
                emotion_plot = gr.Plot(
                    label="Emotional Arc"
                )
                
            with gr.Row():
                # Performance insights
                insights_display = gr.Markdown(
                    value="Performance insights will appear here..."
                )
                
            with gr.Row():
                # Audience reactions
                reactions_display = gr.Textbox(
                    label="Audience Reactions",
                    value="",
                    lines=3
                )
                
            def start_performance(theme: str, scene_count: int):
                """Handle performance start"""
                if not theme.strip():
                    return {
                        current_scene: "Please enter a theme for the performance",
                        emotion_plot: self._create_empty_plot(),
                        insights_display: "Waiting for performance...",
                        reactions_display: ""
                    }
                    
                # Start new performance
                performance = self.theater.start_performance(theme, int(scene_count))
                
                if not performance:
                    return {
                        current_scene: "Error starting performance",
                        emotion_plot: self._create_empty_plot(),
                        insights_display: "An error occurred",
                        reactions_display: ""
                    }
                    
                # Format scene display
                scenes_text = ""
                for i, scene in enumerate(performance.scenes, 1):
                    scenes_text += f"Scene {i}: {scene.title}\n"
                    scenes_text += f"Setting: {scene.setting}\n"
                    scenes_text += f"Characters: {', '.join(scene.characters)}\n\n"
                    scenes_text += scene.narrative + "\n\n"
                    
                # Format insights
                insights_md = "### 🎯 Performance Insights\n\n"
                for insight in performance.insights:
                    insights_md += f"- {insight}\n"
                    
                # Format reactions
                reactions_text = "\n".join(performance.audience_reactions)
                
                return {
                    current_scene: scenes_text,
                    emotion_plot: self._create_emotion_plot(performance.emotional_arc),
                    insights_display: insights_md,
                    reactions_display: reactions_text
                }
                
            # Connect event handlers
            start_btn.click(
                start_performance,
                inputs=[theme_input, num_scenes],
                outputs=[current_scene, emotion_plot, insights_display, reactions_display]
            )
            
        return interface
        
    def _create_emotion_plot(self, emotional_arc: List[torch.Tensor]) -> go.Figure:
        """Create emotion plot for performance"""
        if not emotional_arc:
            return self._create_empty_plot()
            
        fig = go.Figure()
        
        emotions = ['Joy', 'Trust', 'Fear', 'Surprise']
        x = list(range(len(emotional_arc)))
        
        for i, emotion in enumerate(emotions):
            y = [state[i].item() for state in emotional_arc]
            fig.add_trace(go.Scatter(
                x=x,
                y=y,
                mode='lines+markers',
                name=emotion,
                line=dict(width=2)
            ))
            
        fig.update_layout(
            title='Emotional Arc of Performance',
            xaxis_title='Scene',
            yaxis_title='Emotion Intensity',
            template='plotly_dark',
            showlegend=True
        )
        
        return fig
        
    def _create_empty_plot(self) -> go.Figure:
        """Create empty emotion plot"""
        fig = go.Figure()
        
        fig.update_layout(
            title='Emotional Arc of Performance',
            xaxis_title='Scene',
            yaxis_title='Emotion Intensity',
            template='plotly_dark',
            showlegend=True
        )
        
        return fig

def launch_theater():
    """Launch the Memory Theater interface"""
    child = DynamicNeuralChild()
    interface = TheaterInterface(child)
    demo = interface.create_interface()
    
    # Launch with basic configuration
    demo.queue().launch(
        server_name="localhost",
        server_port=7863,  # Different port from other services
        share=False,
        debug=True
    )

if __name__ == "__main__":
    launch_theater() 