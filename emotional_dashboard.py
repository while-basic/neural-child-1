"""
emotional_dashboard.py
Created: 2024
Description: A Gradio dashboard for visualizing mother-child interactions with emotions in real-time.
"""

import gradio as gr
import torch
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
from typing import Dict, List, Tuple
from main import MotherLLM, DigitalChild
import logging
import time
from llm_module import lm_studio

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EmotionalDashboard:
    def __init__(self):
        self.mother = MotherLLM()
        self.child = DigitalChild()
        self.conversation_history = []
        self.emotional_history = []
        
    def check_lm_studio_connection(self) -> bool:
        """Check if LM Studio is available and properly configured"""
        try:
            if not lm_studio.health_check():
                return False
            # Test connection with a simple message
            test_response = lm_studio.chat_completion(
                messages=[{"role": "user", "content": "test"}],
                temperature=0.7
            )
            return bool(test_response)
        except Exception as e:
            logger.error(f"LM Studio connection test failed: {str(e)}")
            return False
            
    def create_emotion_plot(self) -> go.Figure:
        """Create a real-time emotion plot using Plotly"""
        if not self.emotional_history:
            # Create empty plot if no history
            return go.Figure()
            
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
        
    def create_trust_gauge(self, trust_level: float) -> go.Figure:
        """Create a trust level gauge using Plotly"""
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=trust_level * 100,
            title={'text': "Trust Level"},
            gauge={
                'axis': {'range': [None, 100]},
                'bar': {'color': "rgb(50,205,50)"},
                'steps': [
                    {'range': [0, 30], 'color': "rgb(255,99,71)"},
                    {'range': [30, 70], 'color': "rgb(255,215,0)"},
                    {'range': [70, 100], 'color': "rgb(50,205,50)"}
                ]
            }
        ))
        
        fig.update_layout(
            template='plotly_dark',
            height=300
        )
        
        return fig
        
    def process_interaction(self, 
                          message: str, 
                          chat_history: List[Tuple[str, str]], 
                          emotional_plot: go.Figure,
                          trust_gauge: go.Figure) -> Tuple:
        """Process a new interaction and update visualizations"""
        try:
            # Check LM Studio connection first
            if not self.check_lm_studio_connection():
                error_msg = ("⚠️ LM Studio is not available. Please make sure LM Studio is running "
                           "at http://localhost:1234 and a model is loaded.")
                logger.error(error_msg)
                return chat_history, emotional_plot, trust_gauge, error_msg
                
            # Get mother's response
            mother_response = self.mother.respond(message, context=self.conversation_history)
            
            # Update child's emotional state
            child_emotions = self.child.update_emotions(self.mother.current_emotion)
            
            # Store emotional state
            self.emotional_history.append([
                child_emotions['joy'],
                child_emotions['trust'],
                child_emotions['fear'],
                child_emotions['surprise']
            ])
            
            # Update conversation history
            chat_history.append((message, mother_response))
            self.conversation_history.append({"role": "user", "content": message})
            self.conversation_history.append({"role": "assistant", "content": mother_response})
            
            # Update visualizations
            updated_plot = self.create_emotion_plot()
            updated_gauge = self.create_trust_gauge(child_emotions['trust'])
            
            # Get child's expression
            child_feeling = self.child.express_feeling()
            
            # Create status message
            status = f"Child is feeling: {child_feeling} (Trust Level: {child_emotions['trust']:.2f})"
            
            return chat_history, updated_plot, updated_gauge, status
            
        except Exception as e:
            logger.error(f"Error in interaction processing: {str(e)}", exc_info=True)
            error_msg = ("⚠️ An error occurred while processing your message. "
                        "Please check the logs for details.")
            return chat_history, emotional_plot, trust_gauge, error_msg
    
    def create_interface(self) -> gr.Blocks:
        """Create the Gradio interface"""
        with gr.Blocks(
            title="Neural Child Emotional Dashboard",
            css="h1 { font-family: 'Space Grotesk', sans-serif; }"
        ) as interface:
            gr.Markdown(
                """
                # 🤖 Neural Child Emotional Dashboard
                Observe and interact with the emotional development of the digital child.
                
                ℹ️ Make sure LM Studio is running at http://localhost:1234 with a model loaded.
                """
            )
            
            with gr.Row():
                with gr.Column(scale=2):
                    chatbot = gr.Chatbot(
                        label="Interaction Log",
                        height=400,
                        type="messages"  # Use OpenAI-style message format
                    )
                    message = gr.Textbox(
                        label="Your message",
                        placeholder="Type your message here...",
                        lines=2
                    )
                    status = gr.Textbox(
                        label="Child's Emotional Status",
                        interactive=False
                    )
                
                with gr.Column(scale=3):
                    with gr.Row():
                        emotion_plot = gr.Plot(
                            label="Emotional Evolution"
                        )
                    with gr.Row():
                        trust_gauge = gr.Plot(
                            label="Trust Level"
                        )
            
            # Initialize empty plots
            emotion_plot.value = self.create_emotion_plot()
            trust_gauge.value = self.create_trust_gauge(0.5)
            
            # Handle message submission
            message.submit(
                self.process_interaction,
                inputs=[message, chatbot, emotion_plot, trust_gauge],
                outputs=[chatbot, emotion_plot, trust_gauge, status]
            )
            
        return interface

def find_available_port(start_port: int = 7861, max_attempts: int = 10) -> int:
    """Find an available port starting from start_port"""
    import socket
    
    for port in range(start_port, start_port + max_attempts):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('127.0.0.1', port))
                return port
        except OSError:
            continue
    raise OSError(f"Could not find an available port in range {start_port}-{start_port + max_attempts - 1}")

def launch_dashboard():
    """Launch the emotional dashboard"""
    dashboard = EmotionalDashboard()
    
    # Check LM Studio connection before launching
    if not dashboard.check_lm_studio_connection():
        print("\n⚠️ Warning: LM Studio is not available!")
        print("Please make sure:")
        print("1. LM Studio is running at http://localhost:1234")
        print("2. A model is loaded in LM Studio")
        print("3. The API is accessible\n")
    
    try:
        # Find an available port
        port = find_available_port()
        print(f"\n🚀 Launching dashboard on port {port}...")
        
        interface = dashboard.create_interface()
        interface.launch(
            server_name="127.0.0.1",  # Only allow local connections
            server_port=port,
            share=False,  # Disable public URL for security
            show_error=True
        )
    except OSError as e:
        print(f"\n❌ Error: {str(e)}")
        print("\nTrying to kill existing process and restart...")
        
        # Try to kill existing process
        import psutil
        for proc in psutil.process_iter(['pid', 'name', 'connections']):
            try:
                for conn in proc.connections():
                    if conn.laddr.port == 7861:
                        proc.kill()
                        print(f"Killed process {proc.pid} using port 7861")
                        break
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        
        # Try launching again on the original port
        try:
            print("\n🔄 Retrying launch...")
            interface = dashboard.create_interface()
            interface.launch(
                server_name="127.0.0.1",
                server_port=7861,
                share=False,
                show_error=True
            )
        except Exception as e2:
            print(f"\n❌ Failed to launch dashboard: {str(e2)}")
            print("\nPlease try:")
            print("1. Close any other Gradio applications that might be running")
            print("2. Wait a few minutes and try again")
            print("3. Use a different port by setting the GRADIO_SERVER_PORT environment variable")
            raise
    except Exception as e:
        print(f"\n❌ Unexpected error launching dashboard: {str(e)}")
        raise

if __name__ == "__main__":
    launch_dashboard() 