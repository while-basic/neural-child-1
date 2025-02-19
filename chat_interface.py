import gradio as gr
from typing import Dict, Any
import torch
from datetime import datetime
import plotly.graph_objects as go
import numpy as np

class NeuralChildInterface:
    def __init__(self, digital_child, mother_llm):
        self.child = digital_child
        self.mother = mother_llm
        self.chat_history = []
        self.current_focus = "mother"  # Toggle between "mother" and "child"
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.emotional_history = []  # Store emotional history for plotting
        
    def create_emotion_plot(self) -> go.Figure:
        """Create a real-time emotion plot using Plotly"""
        if not self.emotional_history:
            # Create empty plot if no history
            fig = go.Figure()
            fig.update_layout(
                title='Emotional State Evolution',
                xaxis_title='Interaction Steps',
                yaxis_title='Emotion Intensity',
                template='plotly_dark'
            )
            return fig
            
        emotions = ['Joy', 'Trust', 'Fear', 'Surprise']
        x = list(range(len(self.emotional_history)))
        
        fig = go.Figure()
        
        for i, emotion in enumerate(emotions):
            y = [state[i] for state in self.emotional_history]
            fig.add_trace(go.Scatter(
                x=x, 
                y=y,
                mode='lines+markers',
                name=emotion,
                line=dict(width=2)
            ))
            
        fig.update_layout(
            title='Emotional State Evolution',
            xaxis_title='Interaction Steps',
            yaxis_title='Emotion Intensity',
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
        
    def create_interface(self):
        """Create a Gradio interface for interaction"""
        with gr.Blocks(
            title="Neural Child Development Interface",
            css="h1 { font-family: system-ui, -apple-system, sans-serif; }"
        ) as interface:
            with gr.Row():
                with gr.Column(scale=2):
                    # Status displays
                    gr.Markdown("# 👶 Neural Child Development System")
                    age_display = gr.Markdown(value=self._get_age_display())
                    emotional_state = gr.Label(
                        label="Current Emotional State",
                        value={"NEUTRAL": 1.0}  # Format as dictionary
                    )
                    development_stage = gr.Label(
                        label="Development Stage",
                        value={"EARLY_ELEMENTARY": 1.0}  # Format as dictionary
                    )
                    
                    # Add emotional plot
                    emotion_plot = gr.Plot(
                        label="Emotional Evolution",
                        value=self.create_emotion_plot()
                    )
                
                with gr.Column(scale=2):
                    # Chat interface
                    chatbot = gr.Chatbot(
                        label="Interaction Log",
                        height=400,
                        value=[],
                        type="messages"  # Use OpenAI-style message format
                    )
                    msg = gr.Textbox(
                        label="Your Message",
                        placeholder="Type your message here...",
                        lines=2
                    )
                    
                    with gr.Row():
                        submit = gr.Button("Send", variant="primary")
                        clear = gr.Button("Clear")
                        refresh = gr.Button("Refresh Status")
                        
                    with gr.Row():
                        talk_to_mother = gr.Button("Talk to Mother 👩")
                        talk_to_child = gr.Button("Talk to Child 👶")
            
            # Event handlers
            msg.submit(
                self.process_message,
                inputs=[msg, chatbot],
                outputs=[msg, chatbot, emotional_state, development_stage, emotion_plot]
            )
            
            submit.click(
                self.process_message,
                inputs=[msg, chatbot],
                outputs=[msg, chatbot, emotional_state, development_stage, emotion_plot]
            )
            
            clear.click(
                lambda: ("", [], {"NEUTRAL": 1.0}, {"EARLY_ELEMENTARY": 1.0}, self.create_emotion_plot()),
                outputs=[msg, chatbot, emotional_state, development_stage, emotion_plot]
            )
            
            refresh.click(
                self._update_status,
                outputs=[emotional_state, development_stage, emotion_plot]
            )
            
            talk_to_mother.click(
                lambda: self.set_focus("mother"),
                outputs=[]
            )
            
            talk_to_child.click(
                lambda: self.set_focus("child"),
                outputs=[]
            )
            
        return interface
    
    def process_message(self, message: str, history: list) -> tuple:
        """Process incoming messages and generate responses"""
        try:
            if not message.strip():
                return "", history, {"NEUTRAL": 1.0}, {"EARLY_ELEMENTARY": 1.0}, self.create_emotion_plot()
                
            print("\n" + "="*50)
            print(f"Processing message: '{message}'")
            print(f"Current focus: {self.current_focus}")
            print("="*50)
            
            # Add user message to history
            history.append({"role": "user", "content": message})
            self.chat_history.append({"role": "user", "content": message})
            
            # Evaluate message emotional content
            print("\n📊 Evaluating emotional context...")
            emotional_context = self._evaluate_emotional_context(message)
            print(f"Raw emotional tensor: {emotional_context}")
            
            # Store emotional state in history
            print("📈 Storing emotional state in history...")
            self.emotional_history.append(emotional_context.cpu().numpy())
            print(f"History length: {len(self.emotional_history)}")
            
            # Generate response based on current focus
            try:
                if self.current_focus == "mother":
                    print("\n👩 Requesting response from mother...")
                    mother_context = {
                        "role": "system",
                        "content": """You are a caring mother. Keep responses brief (1-2 sentences). 
                        Use warm, nurturing language. Be direct and practical."""
                    }
                    
                    context_history = [mother_context] + self.chat_history[-1:]
                    print(f"Context length: {len(context_history)}")
                    
                    response = self.mother.respond(message, context=context_history)
                    print(f"Mother's response: '{response}'")
                    
                    # Update child's emotional state based on mother's response
                    print("\n🔄 Updating child's emotional state from mother's response...")
                    if hasattr(self.mother, 'current_emotion'):
                        mother_emotion = self.mother.current_emotion
                        if isinstance(mother_emotion, dict):
                            print(f"Mother's emotion: {mother_emotion}")
                            mother_tensor = torch.tensor([
                                mother_emotion.get('joy', 0.5),
                                mother_emotion.get('trust', 0.5),
                                mother_emotion.get('fear', 0.0),
                                mother_emotion.get('surprise', 0.2)
                            ], device=self.device)
                            print(f"Mother's emotional tensor: {mother_tensor}")
                            self.child.update_emotions(mother_tensor)
                        else:
                            print("Using emotional context as fallback (mother's emotion not dict)")
                            self.child.update_emotions(emotional_context)
                    else:
                        print("Using emotional context as fallback (no mother emotion)")
                        self.child.update_emotions(emotional_context)
                else:
                    print("\n👶 Requesting response from child...")
                    try:
                        # Update emotional state
                        print("Updating child's emotional state...")
                        self.child.update_emotions(emotional_context)
                        
                        # Get child's response
                        print("Getting child's response...")
                        if hasattr(self.child, 'process_interaction'):
                            print("Using process_interaction method")
                            response = self.child.process_interaction(message)
                        elif hasattr(self.child, 'respond'):
                            print("Using respond method")
                            response = self.child.respond(message)
                        else:
                            print("No response method found, using fallback")
                            response = "*The child looks at you with understanding*"
                            
                        print(f"Child's response: '{response}'")
                        
                        if not response:
                            print("Empty response, using fallback")
                            response = "*The child nods silently*"
                            
                    except AttributeError as e:
                        print(f"Child response error (AttributeError): {str(e)}")
                        response = "*The child acknowledges your love with a warm smile*"
                        if hasattr(self.child, 'emotional_state'):
                            print("Updating emotional state directly")
                            self.child.emotional_state = emotional_context
                            
                    except Exception as e:
                        print(f"Unexpected error in child response: {str(e)}")
                        response = "*The child looks at you warmly*"
                
                # Update history with response
                history.append({"role": "assistant", "content": response})
                self.chat_history.append({"role": "assistant", "content": response})
                
                # Get updated states
                try:
                    print("\n📊 Getting updated emotional state...")
                    emotional_tensor = self.child.emotional_state
                    if emotional_tensor is None:
                        print("No emotional state found, using neutral")
                        emotional_tensor = torch.tensor([0.5, 0.5, 0.0, 0.2], device=self.device)
                    
                    print(f"Current emotional tensor: {emotional_tensor}")
                    
                    emotional_state = {
                        'joy': float(emotional_tensor[0]),
                        'trust': float(emotional_tensor[1]),
                        'fear': float(emotional_tensor[2]),
                        'surprise': float(emotional_tensor[3])
                    }
                    
                    print("\n🔍 Final Emotional State:")
                    print(f"Joy: {emotional_state['joy']:.2f}")
                    print(f"Trust: {emotional_state['trust']:.2f}")
                    print(f"Fear: {emotional_state['fear']:.2f}")
                    print(f"Surprise: {emotional_state['surprise']:.2f}")
                    
                    development_stage = self.child.get_development_stage()
                    print(f"Development stage: {development_stage}")
                    
                    # Format states for display
                    max_emotion = max(emotional_state.items(), key=lambda x: x[1])
                    formatted_emotional = {max_emotion[0].upper(): float(max_emotion[1])}
                    
                    if isinstance(development_stage, str):
                        formatted_development = {development_stage: 1.0}
                    else:
                        formatted_development = {"EARLY_ELEMENTARY": 1.0}
                    
                    print(f"\nDominant Emotion: {max_emotion[0].upper()} ({max_emotion[1]:.2f})")
                    print("="*50)
                        
                except Exception as e:
                    print(f"Error getting states: {str(e)}")
                    formatted_emotional = {"NEUTRAL": 1.0}
                    formatted_development = {"EARLY_ELEMENTARY": 1.0}
                
                return "", history, formatted_emotional, formatted_development, self.create_emotion_plot()
                
            except Exception as e:
                print(f"Error generating response: {str(e)}")
                error_msg = "I apologize, but I encountered an error while processing your message."
                history.append({"role": "assistant", "content": error_msg})
                return "", history, {"NEUTRAL": 1.0}, {"EARLY_ELEMENTARY": 1.0}, self.create_emotion_plot()
                
        except Exception as e:
            print(f"Critical error in process_message: {str(e)}")
            return "", history, {"NEUTRAL": 1.0}, {"EARLY_ELEMENTARY": 1.0}, self.create_emotion_plot()
    
    def set_focus(self, target: str):
        """Switch focus between mother and child"""
        self.current_focus = target
    
    def _get_age_display(self) -> str:
        """Get formatted age display"""
        return f"Current Neural Age: {self.child.get_age():.2f} years"
    
    def _update_status(self):
        """Update interface status displays"""
        try:
            # Get emotional state directly from the child's emotional state tensor
            emotional_tensor = self.child.emotional_state
            emotional_state = {
                'joy': float(emotional_tensor[0]),
                'trust': float(emotional_tensor[1]),
                'fear': float(emotional_tensor[2]),
                'surprise': float(emotional_tensor[3])
            }
            
            # Find the strongest emotion
            max_emotion = max(emotional_state.items(), key=lambda x: x[1])
            formatted_emotional = {max_emotion[0].upper(): float(max_emotion[1])}
            
            # Get development stage
            development_stage = self.child.get_development_stage()
            if isinstance(development_stage, str):
                formatted_development = {development_stage: 1.0}
            else:
                formatted_development = {"EARLY_ELEMENTARY": 1.0}
                
            return formatted_emotional, formatted_development, self.create_emotion_plot()
            
        except Exception as e:
            print(f"Error updating status: {str(e)}")
            return {"NEUTRAL": 1.0}, {"EARLY_ELEMENTARY": 1.0}, self.create_emotion_plot()
    
    def _evaluate_emotional_context(self, message: str) -> torch.Tensor:
        """Evaluate the emotional context of a message"""
        # Define emotional keywords and their weights
        emotional_keywords = {
            'joy': [
                ('happy', 0.8), ('excited', 0.9), ('great', 0.7), ('wonderful', 0.8),
                ('love', 0.9), ('fun', 0.7), ('amazing', 0.8), ('good', 0.6),
                ('yummy', 0.7), ('delicious', 0.8), ('play', 0.7)
            ],
            'trust': [
                ('trust', 0.9), ('believe', 0.7), ('safe', 0.8), ('confident', 0.8),
                ('sure', 0.6), ('friend', 0.7), ('together', 0.6), ('help', 0.6),
                ('mom', 0.8), ('mommy', 0.9), ('mama', 0.9), ('mother', 0.8),
                ('hungry', 0.6), ('food', 0.5), ('eat', 0.5)
            ],
            'fear': [
                ('scared', 0.9), ('afraid', 0.9), ('worried', 0.8), ('nervous', 0.8),
                ('fear', 0.9), ('scary', 0.8), ('anxious', 0.8), ('bad', 0.6),
                ('starving', 0.7), ('very hungry', 0.6)
            ],
            'surprise': [
                ('wow', 0.9), ('amazing', 0.8), ('unexpected', 0.9), ('surprised', 0.9),
                ('incredible', 0.8), ('unbelievable', 0.9), ('sudden', 0.7), ('strange', 0.6)
            ]
        }
        
        # Physical needs keywords that affect emotional state
        physical_needs = {
            'hunger': [
                ('hungry', 0.7), ('starving', 0.9), ('food', 0.6), ('eat', 0.6),
                ('meal', 0.6), ('snack', 0.5), ('breakfast', 0.6), ('lunch', 0.6),
                ('dinner', 0.6)
            ]
        }
        
        # Initialize emotional values
        emotions = {
            'joy': 0.0,
            'trust': 0.0,
            'fear': 0.0,
            'surprise': 0.0
        }
        
        # Convert message to lowercase for matching
        message_lower = message.lower()
        
        # Evaluate each emotion
        for emotion, keywords in emotional_keywords.items():
            for keyword, weight in keywords:
                if keyword in message_lower:
                    emotions[emotion] = max(emotions[emotion], weight)
        
        # Check for physical needs and adjust emotions accordingly
        for need, keywords in physical_needs.items():
            for keyword, weight in keywords:
                if keyword in message_lower:
                    if need == 'hunger':
                        # Hunger increases trust (seeking care) and slightly increases fear
                        emotions['trust'] = max(emotions['trust'], weight)
                        emotions['fear'] = max(emotions['fear'], weight * 0.3)
        
        # Ensure some minimal emotional response
        if all(v == 0 for v in emotions.values()):
            emotions['trust'] = 0.3  # Default to slight trust
        
        # Convert to tensor
        try:
            emotional_tensor = torch.tensor([
                emotions['joy'],
                emotions['trust'],
                emotions['fear'],
                emotions['surprise']
            ], device=self.device)
        except Exception as e:
            print(f"Warning: Could not create tensor on device {self.device}, falling back to CPU")
            emotional_tensor = torch.tensor([
                emotions['joy'],
                emotions['trust'],
                emotions['fear'],
                emotions['surprise']
            ]).to('cpu')
        
        return emotional_tensor

if __name__ == "__main__":
    from child_model import DynamicNeuralChild
    from main import MotherLLM
    import os
    
    # Initialize the systems
    child = DynamicNeuralChild()
    mother = MotherLLM()
    
    # Create and launch interface
    interface = NeuralChildInterface(child, mother)
    demo = interface.create_interface()
    
    # Configure launch settings
    launch_kwargs = {
        "server_name": "localhost",
        "server_port": 7860,
        "share": False,
        "show_error": True,
        "debug": True,
        "max_threads": 10
    }
    
    # Launch with proper configuration
    demo.queue().launch(**launch_kwargs) 