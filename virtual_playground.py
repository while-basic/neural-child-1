"""
virtual_playground.py
Created: 2024-03-21
Description: A virtual environment where multiple neural children can interact and influence each other's emotional states.

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
from child_model import DynamicNeuralChild
from main import MotherLLM
import gradio as gr

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OllamaAPI:
    """Interface for Ollama API interactions"""
    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url
        self.session = requests.Session()
        
    def generate_dialogue(self, 
                         prompt: str, 
                         model: str = "artifish/llama3.2-uncensored",
                         temperature: float = 0.7) -> str:
        """Generate dialogue using Ollama"""
        try:
            # Send request with longer timeout and stream=False
            response = self.session.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "temperature": temperature,
                    "max_tokens": 100,
                    "stream": True
                },
                timeout=10
            )
            response.raise_for_status()
            
            # Parse response and extract text
            try:
                result = response.json()
                if isinstance(result, dict) and "response" in result:
                    return result["response"].strip()
                else:
                    logger.error(f"Unexpected response format: {result}")
                    return ""
            except json.JSONDecodeError as e:
                # Handle streaming response format
                try:
                    # Split response into lines and get last complete response
                    lines = response.text.strip().split('\n')
                    for line in reversed(lines):
                        if line.strip():
                            data = json.loads(line)
                            if "response" in data:
                                return data["response"].strip()
                    return ""
                except Exception as e2:
                    logger.error(f"Failed to parse streaming response: {str(e2)}")
                    return ""
                    
        except requests.Timeout:
            logger.error("Ollama API timeout - check if model is loaded")
            return ""
        except requests.ConnectionError:
            logger.error("Failed to connect to Ollama API - check if Ollama is running")
            return ""
        except Exception as e:
            logger.error(f"Ollama API error: {str(e)}")
            return ""
            
    def __del__(self):
        """Clean up session"""
        self.session.close()

    def generate_dialogue_with_retry(self, 
                                   prompt: str, 
                                   max_retries: int = 3,
                                   **kwargs) -> str:
        """Generate dialogue with retry logic"""
        for attempt in range(max_retries):
            try:
                response = self.generate_dialogue(prompt, **kwargs)
                if response:
                    return response
                time.sleep(1)  # Wait before retry
            except Exception as e:
                logger.error(f"Attempt {attempt + 1} failed: {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(1)
        return "..."  # Return ellipsis if all retries fail

class VirtualChild:
    def __init__(self, name: str, age: float = 7.0):
        """Initialize a virtual child with a name and neural components"""
        self.name = name
        self.brain = DynamicNeuralChild()
        self.brain.age = age
        self.device = self.brain.device  # Get device from brain
        self.position = torch.rand(2, device=self.device)  # 2D position in playground
        self.velocity = torch.zeros(2, device=self.device)  # 2D velocity
        self.interaction_radius = 0.5   # Increased radius for more interactions
        self.last_interaction = None
        self.interaction_history = []
        self.dialogue_history = []
        self.ollama = OllamaAPI()
        
    def move(self, dt: float = 0.1):
        """Update position based on current velocity"""
        # Add more movement to encourage interactions
        self.velocity += torch.randn(2, device=self.device) * 0.2  # Increased random movement
        self.position += self.velocity * dt
        # Bound position to playground (0,1) x (0,1)
        self.position = torch.clamp(self.position, 0, 1)
        
    def generate_dialogue(self, other_child: 'VirtualChild', distance: float) -> str:
        """Generate dialogue based on emotional states and distance"""
        my_emotions = self.brain.emotional_state.cpu().tolist()
        other_emotions = other_child.brain.emotional_state.cpu().tolist()
        
        # More engaging prompt for better dialogues
        prompt = f"""You are {self.name}, a {self.brain.age}-year-old child playing in a virtual playground. 
        Your current emotions are:
        - Joy: {my_emotions[0]:.2f} (feeling {'very happy' if my_emotions[0] > 0.7 else 'happy' if my_emotions[0] > 0.3 else 'not so happy'})
        - Trust: {my_emotions[1]:.2f} (feeling {'very trusting' if my_emotions[1] > 0.7 else 'trusting' if my_emotions[1] > 0.3 else 'cautious'})
        - Fear: {my_emotions[2]:.2f} (feeling {'very scared' if my_emotions[2] > 0.7 else 'nervous' if my_emotions[2] > 0.3 else 'brave'})
        - Surprise: {my_emotions[3]:.2f} (feeling {'very excited' if my_emotions[3] > 0.7 else 'curious' if my_emotions[3] > 0.3 else 'calm'})
        
        You see {other_child.name} who is {other_child.brain.age} years old {distance:.1f} units away from you.
        Their emotions show they are feeling:
        - {'very happy' if other_emotions[0] > 0.7 else 'happy' if other_emotions[0] > 0.3 else 'not so happy'}
        - {'very trusting' if other_emotions[1] > 0.7 else 'trusting' if other_emotions[1] > 0.3 else 'cautious'}
        - {'very scared' if other_emotions[2] > 0.7 else 'nervous' if other_emotions[2] > 0.3 else 'brave'}
        - {'very excited' if other_emotions[3] > 0.7 else 'curious' if other_emotions[3] > 0.3 else 'calm'}
        
        Say ONE natural, child-like sentence to {other_child.name} that reflects both your emotional states.
        Keep it brief, playful, and age-appropriate.
        """
        
        return self.ollama.generate_dialogue(prompt)
        
    def update_emotional_state(self, nearby_children: List['VirtualChild']):
        """Update emotional state based on nearby children"""
        if not nearby_children:
            # Add more random movement when alone to seek interactions
            self.velocity += torch.randn(2, device=self.device) * 0.25
            return
            
        # Calculate combined emotional influence
        combined_emotion = torch.zeros(4, device=self.device)
        influence_count = 0
        
        for child in nearby_children:
            distance = torch.norm(self.position - child.position.to(self.device))
            if distance < self.interaction_radius and child != self:
                # Increased interaction probability
                if torch.rand(1).item() < 0.7:  # 70% chance to interact when nearby
                    # Emotional influence decreases with distance
                    influence = 1.0 - (distance / self.interaction_radius)
                    combined_emotion += child.brain.emotional_state.to(self.device) * influence
                    influence_count += 1
                    
                    # Generate and record dialogue
                    dialogue = self.generate_dialogue(child, distance.item())
                    if dialogue:
                        interaction = {
                            'time': datetime.now(),
                            'speaker': self.name,
                            'listener': child.name,
                            'distance': distance.item(),
                            'dialogue': dialogue
                        }
                        self.dialogue_history.append(interaction)
                        logger.info(f"Dialogue: {self.name} to {child.name}: {dialogue}")
                    
                    # Record interaction
                    self.interaction_history.append({
                        'time': datetime.now(),
                        'partner': child.name,
                        'distance': distance.item(),
                        'influence': influence.item()
                    })
        
        if influence_count > 0:
            combined_emotion /= influence_count
            # Update emotional state with some randomness
            self.brain.update_emotions(combined_emotion)
            
            # Update velocity based on emotional state
            # Joy and trust increase approach behavior
            # Fear increases avoidance
            approach = (self.brain.emotional_state[0] + self.brain.emotional_state[1]) * 0.5
            avoidance = self.brain.emotional_state[2]
            
            for child in nearby_children:
                direction = child.position.to(self.device) - self.position
                self.velocity += direction * (approach - avoidance) * 0.15  # Increased movement speed
                
            # Add some random movement
            self.velocity += torch.randn(2, device=self.device) * 0.1
            
            # Limit velocity magnitude
            speed = torch.norm(self.velocity)
            if speed > 1.0:
                self.velocity /= speed

class VirtualPlayground:
    def __init__(self):
        """Initialize the virtual playground environment"""
        self.children: Dict[str, VirtualChild] = {}
        self.mother = MotherLLM()
        self.time_step = 0.1
        self.emotional_history = []
        self.interaction_log = []
        self.dialogue_history = []
        # Get device from first child or default to CPU
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def add_child(self, name: str, age: Optional[float] = None):
        """Add a new child to the playground"""
        try:
            if len(self.children) >= 10:  # Soft limit for performance
                logger.warning("Adding more than 10 children may impact performance")
                
            if age is None:
                age = 7.0 + torch.rand(1, device=self.device).item() * 2  # Random age between 7-9
                
            # Ensure unique name
            base_name = name
            counter = 1
            while name in self.children:
                name = f"{base_name}_{counter}"
                counter += 1
                
            self.children[name] = VirtualChild(name, age)
            
            # Add initial state to history
            self._record_emotional_states()
            
            logger.info(f"Added child {name} (age: {age:.1f}) to playground")
            
        except Exception as e:
            logger.error(f"Error adding child: {str(e)}")
            raise
        
    def remove_child(self, name: str):
        """Remove a child from the playground"""
        if name in self.children:
            del self.children[name]
            logger.info(f"Removed child {name} from playground")
            
    def update(self):
        """Update all children in the playground"""
        # Update positions
        for child in self.children.values():
            child.move(self.time_step)
            
        # Update emotional states
        for child in self.children.values():
            nearby = self._get_nearby_children(child)
            child.update_emotional_state(nearby)
            
        # Record emotional states
        self._record_emotional_states()
        
    def _get_nearby_children(self, target_child: VirtualChild) -> List[VirtualChild]:
        """Get list of children within interaction radius"""
        nearby = []
        for child in self.children.values():
            if child != target_child:
                # Ensure tensors are on the same device before computing distance
                target_pos = target_child.position.to(target_child.device)
                child_pos = child.position.to(target_child.device)
                distance = torch.norm(child_pos - target_pos)
                if distance < target_child.interaction_radius:
                    nearby.append(child)
        return nearby
        
    def _record_emotional_states(self):
        """Record current emotional states of all children"""
        try:
            state_snapshot = {
                'time': datetime.now(),
                'states': {}
            }
            
            for name, child in self.children.items():
                # Convert tensors to CPU and then to Python lists
                position = child.position.detach().cpu().numpy().tolist()
                emotional_state = child.brain.emotional_state.detach().cpu().numpy().tolist()
                velocity = child.velocity.detach().cpu().numpy().tolist()
                
                state_snapshot['states'][name] = {
                    'position': position,
                    'emotional_state': emotional_state,
                    'velocity': velocity
                }
                
            self.emotional_history.append(state_snapshot)
            
        except Exception as e:
            logger.error(f"Error recording emotional states: {str(e)}")
            # Create minimal valid state to prevent breaking the visualization
            state_snapshot = {
                'time': datetime.now(),
                'states': {}
            }
            self.emotional_history.append(state_snapshot)
        
    def create_playground_plot(self) -> go.Figure:
        """Create an interactive plot of the playground"""
        fig = go.Figure()
        
        # Plot each child's position
        for name, child in self.children.items():
            # Create emotion color (red for fear, green for joy, blue for trust)
            emotion_color = [
                float(child.brain.emotional_state.cpu()[0]),  # Joy (green)
                float(child.brain.emotional_state.cpu()[1]),  # Trust (blue)
                float(child.brain.emotional_state.cpu()[2])   # Fear (red)
            ]
            
            # Plot child position
            fig.add_trace(go.Scatter(
                x=[float(child.position.cpu()[0])],
                y=[float(child.position.cpu()[1])],
                mode='markers+text',
                name=name,
                text=[name],
                marker=dict(
                    size=15,
                    color=f'rgb({emotion_color[2]*255},{emotion_color[0]*255},{emotion_color[1]*255})'
                )
            ))
            
            # Plot interaction radius
            theta = np.linspace(0, 2*np.pi, 100)
            radius = child.interaction_radius
            x = float(child.position.cpu()[0]) + radius * np.cos(theta)
            y = float(child.position.cpu()[1]) + radius * np.sin(theta)
            
            fig.add_trace(go.Scatter(
                x=x, y=y,
                mode='lines',
                line=dict(color='rgba(100,100,100,0.2)'),
                showlegend=False
            ))
        
        fig.update_layout(
            title='Virtual Playground',
            xaxis_title='X Position',
            yaxis_title='Y Position',
            xaxis=dict(range=[0, 1]),
            yaxis=dict(range=[0, 1]),
            template='plotly_dark',
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            )
        )
        
        return fig
        
    def create_emotion_plot(self) -> go.Figure:
        """Create a plot showing emotional states over time"""
        fig = go.Figure()
        
        if not self.emotional_history or not self.children:
            # Return empty plot with proper layout
            fig.update_layout(
                title='Emotional States Over Time',
                xaxis_title='Time Step',
                yaxis_title='Emotion Intensity',
                template='plotly_dark',
                showlegend=True
            )
            return fig
            
        emotions = ['Joy', 'Trust', 'Fear', 'Surprise']
        
        try:
            # Only plot for currently existing children
            current_children = set(self.children.keys())
            
            for name in current_children:
                for i, emotion in enumerate(emotions):
                    # Extract emotional states for this child
                    y = []
                    for state in self.emotional_history:
                        if name in state.get('states', {}):
                            try:
                                y.append(state['states'][name]['emotional_state'][i])
                            except (KeyError, IndexError):
                                y.append(None)  # Use None for missing data points
                        else:
                            y.append(None)  # Use None for missing states
                            
                    # Remove any None values by only plotting valid data points
                    x = []
                    y_clean = []
                    for idx, val in enumerate(y):
                        if val is not None:
                            x.append(idx)
                            y_clean.append(val)
                            
                    if x and y_clean:  # Only plot if we have valid data
                        fig.add_trace(go.Scatter(
                            x=x,
                            y=y_clean,
                            mode='lines',
                            name=f'{name} - {emotion}',
                            line=dict(width=2)
                        ))
                        
        except Exception as e:
            logger.error(f"Error creating emotion plot: {str(e)}")
            # Return basic plot on error
            fig.update_layout(
                title='Emotional States Over Time (Error occurred)',
                xaxis_title='Time Step',
                yaxis_title='Emotion Intensity',
                template='plotly_dark',
                showlegend=True
            )
            return fig
        
        fig.update_layout(
            title='Emotional States Over Time',
            xaxis_title='Time Step',
            yaxis_title='Emotion Intensity',
            template='plotly_dark',
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01,
                font=dict(size=8)  # Smaller font for many children
            ),
            height=500  # Taller plot for better visibility
        )
        
        return fig

    def get_recent_dialogues(self, limit: int = 5) -> List[Dict]:
        """Get the most recent dialogues across all children"""
        all_dialogues = []
        for child in self.children.values():
            all_dialogues.extend(child.dialogue_history)
        
        # Sort by time and get most recent
        all_dialogues.sort(key=lambda x: x['time'], reverse=True)
        return all_dialogues[:limit]

class PlaygroundInterface:
    def __init__(self):
        """Initialize the playground interface"""
        self.playground = VirtualPlayground()
        self.is_running = False
        self.update_interval = None
        
    def create_interface(self) -> gr.Blocks:
        """Create the Gradio interface for the virtual playground"""
        with gr.Blocks(title="Neural Children Virtual Playground") as interface:
            gr.Markdown(
                """
                # 🎮 Neural Children Virtual Playground
                Watch and interact with multiple neural children in a virtual environment.
                """
            )
            
            with gr.Row():
                with gr.Column(scale=2):
                    playground_plot = gr.Plot(
                        label="Playground View",
                        value=self.playground.create_playground_plot()
                    )
                    
                with gr.Column(scale=2):
                    emotion_plot = gr.Plot(
                        label="Emotional Evolution",
                        value=self.playground.create_emotion_plot()
                    )
            
            with gr.Row():
                child_name = gr.Textbox(
                    label="Child Name",
                    placeholder="Enter name for new child..."
                )
                child_age = gr.Slider(
                    minimum=7.0,
                    maximum=9.0,
                    value=7.0,
                    step=0.1,
                    label="Child Age"
                )
                add_button = gr.Button("Add Child", variant="primary")
                
            with gr.Row():
                start_button = gr.Button("Start Simulation ▶️", variant="primary")
                stop_button = gr.Button("Stop Simulation ⏹️", variant="secondary")
                clear_button = gr.Button("Clear Playground 🔄")
            
            # Add dialogue display
            with gr.Row():
                dialogue_box = gr.Textbox(
                    label="Recent Conversations",
                    value="Children's conversations will appear here...",
                    lines=5,
                    max_lines=5,
                    interactive=False
                )
            
            # Add interval for continuous updates
            simulation_interval = gr.Number(value=1, visible=False)  # 1 second interval
            
            def add_child(name, age):
                if name.strip():
                    self.playground.add_child(name, age)
                    return (
                        "",  # Clear name input
                        self.playground.create_playground_plot(),
                        self.playground.create_emotion_plot(),
                        "Added new child to playground..."
                    )
                return (
                    "Please enter a name",
                    playground_plot.value,
                    emotion_plot.value,
                    dialogue_box.value
                )
            
            def update_simulation():
                try:
                    if self.is_running:
                        # Update playground state
                        self.playground.update()
                        
                        # Get recent dialogues with more detailed formatting
                        recent_dialogues = self.playground.get_recent_dialogues()
                        if recent_dialogues:
                            dialogue_text = "\n".join([
                                f"[{d['time'].strftime('%H:%M:%S')}] {d['speaker']} → {d['listener']}: {d['dialogue']}"
                                for d in recent_dialogues
                            ])
                        else:
                            dialogue_text = "No recent conversations... (Add children and wait for them to interact)"
                        
                        # Create updated plots and return
                        return (
                            self.playground.create_playground_plot(),
                            self.playground.create_emotion_plot(),
                            dialogue_text
                        )
                    
                    # If not running, return current state
                    return (
                        playground_plot.value,
                        emotion_plot.value,
                        dialogue_box.value
                    )
                        
                except Exception as e:
                    logger.error(f"Error in simulation update: {str(e)}")
                    self.is_running = False
                    return (
                        playground_plot.value,
                        emotion_plot.value,
                        f"Error in simulation: {str(e)}"
                    )
            
            def start_simulation():
                self.is_running = True
                return update_simulation()
            
            def stop_simulation():
                self.is_running = False
                return (
                    playground_plot.value,
                    emotion_plot.value,
                    "Simulation stopped."
                )
            
            def clear_playground():
                self.playground = VirtualPlayground()
                self.is_running = False
                return (
                    self.playground.create_playground_plot(),
                    self.playground.create_emotion_plot(),
                    "Playground cleared..."
                )
            
            # Connect event handlers
            add_button.click(
                add_child,
                inputs=[child_name, child_age],
                outputs=[child_name, playground_plot, emotion_plot, dialogue_box]
            )
            
            start_button.click(
                start_simulation,
                outputs=[playground_plot, emotion_plot, dialogue_box]
            )
            
            stop_button.click(
                stop_simulation,
                outputs=[playground_plot, emotion_plot, dialogue_box]
            )
            
            clear_button.click(
                clear_playground,
                outputs=[playground_plot, emotion_plot, dialogue_box]
            )
            
            # Add update button for manual updates
            update_button = gr.Button("Update ⟳", variant="secondary")
            update_button.click(
                update_simulation,
                outputs=[playground_plot, emotion_plot, dialogue_box]
            )
            
            # Set up interval for continuous updates
            gr.Markdown("").update(every=1, inputs=None, outputs=[playground_plot, emotion_plot, dialogue_box], _js="""
                () => {
                    if (window.simulationInterval) {
                        clearInterval(window.simulationInterval);
                    }
                    window.simulationInterval = setInterval(() => {
                        document.querySelector('#update-btn').click();
                    }, 1000);
                    return [];
                }
            """)
            
        return interface

def launch_playground():
    """Launch the virtual playground interface"""
    interface = PlaygroundInterface()
    demo = interface.create_interface()
    
    # Launch with basic configuration
    demo.queue().launch(
        server_name="localhost",
        server_port=7861,
        share=False,
        debug=True
    )

if __name__ == "__main__":
    launch_playground() 