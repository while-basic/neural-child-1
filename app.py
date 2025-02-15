import streamlit as st
import torch
import time
from datetime import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from developmental_stages import DevelopmentalStage
from config import STAGE_DEFINITIONS

try:
    from main import DigitalChild, MotherLLM
except ImportError:
    st.error("Failed to import DigitalChild and MotherLLM. Make sure main.py exists and is properly configured.")
    DigitalChild = None
    MotherLLM = None

# Initialize session state at the top level
if DigitalChild is not None:
    if 'initialized' not in st.session_state:
        try:
            st.session_state.child = DigitalChild()
            st.session_state.conversation_history = []
            st.session_state.emotional_history = []
            st.session_state.development_metrics = {
                'success_rate': [],
                'abstraction': [],
                'self_awareness': []
            }
            st.session_state.initialized = True
        except Exception as e:
            st.error(f"Error initializing digital child: {str(e)}")

def main():
    if DigitalChild is None:
        st.error("Cannot run application: Required modules not found")
        return
    
    if not st.session_state.get('initialized', False):
        st.error("Session state not properly initialized")
        return

    st.title("Digital Child Development System")
    
    # Enhanced sidebar for development stage and metrics
    with st.sidebar:
        st.subheader("Development Information")
        current_age = st.session_state.child.age()
        st.write(f"Child's Age: {current_age} months")
        
        # Add progress indicator for current stage
        current_stage = st.session_state.child.curriculum.current_stage
        st.write("Developmental Progress:")
        stage_progress = {
            'Stage': current_stage.name,
            'Progress': len(st.session_state.child.curriculum.stage_history) / 100.0  # Normalize to 0-1
        }
        
        # Visual progress bar for stage completion
        st.progress(min(stage_progress['Progress'], 1.0))
        
        # Show requirements for next stage
        next_stage_value = min(current_stage.value + 1, len(DevelopmentalStage) - 1)
        next_stage = DevelopmentalStage(next_stage_value)
        
        with st.expander("Current Stage Requirements"):
            requirements = st.session_state.child.curriculum.get_stage_requirements()
            st.write("Required Skills:")
            for skill in requirements['behaviors']['required_skills']:
                st.write(f"- {skill}")
            st.write("\nLearning Focus:")
            for focus in requirements['behaviors']['learning_focus']:
                st.write(f"- {focus}")
                
        with st.expander("Next Stage"):
            st.write(f"Next Stage: {next_stage.name}")
            next_requirements = STAGE_DEFINITIONS[next_stage]
            st.write("Skills Needed:")
            for skill in next_requirements.required_skills:
                st.write(f"- {skill}")

        # Display emotional state
        st.subheader("Current Emotional State")
        emotional_state = st.session_state.child.emotional_state.cpu().numpy()
        
        # Create gauge charts for emotions
        fig = make_subplots(
            rows=2, cols=2,
            specs=[[{'type': 'indicator'}, {'type': 'indicator'}],
                  [{'type': 'indicator'}, {'type': 'indicator'}]],
            subplot_titles=("Happiness", "Trust", "Fear", "Surprise")
        )
        
        for idx, emotion in enumerate(['Happiness', 'Trust', 'Fear', 'Surprise']):
            fig.add_trace(
                go.Indicator(
                    mode="gauge+number",
                    value=emotional_state[idx] * 100,
                    title={'text': emotion},
                    gauge={'axis': {'range': [0, 100]}}
                ),
                row=(idx // 2) + 1,
                col=(idx % 2) + 1
            )
        
        fig.update_layout(height=500)
        st.plotly_chart(fig)

    # Add stage-appropriate interaction suggestions
    st.subheader("Interaction Guidelines")
    current_behaviors = st.session_state.child.curriculum.get_stage_requirements()['behaviors']
    with st.expander("Suggested Interactions"):
        st.write("Appropriate Actions:")
        for action in current_behaviors['allowed_actions']:
            st.write(f"- {action}")
        st.write("\nEmotional Range:")
        for emotion in current_behaviors['emotional_range']:
            st.write(f"- {emotion}")

    # Enhanced interaction area
    st.subheader("Interact with Your Digital Child")
    
    # Display current feeling with stage-appropriate context
    current_feeling = st.session_state.child.express_feeling()
    st.info(f"Child is feeling: {current_feeling}")
    st.write("Stage-appropriate response range:", 
             ', '.join(current_behaviors['emotional_range']))

    # Add interaction templates based on current stage
    template_options = list(current_behaviors['allowed_actions'])
    selected_template = st.selectbox("Interaction Type:", ["Custom"] + template_options)
    
    if selected_template == "Custom":
        user_input = st.text_input("Say something to the child:", key="user_input")
    else:
        # Pre-fill input with template
        user_input = st.text_input("Say something to the child:", 
                                  value=f"[{selected_template.upper()}]",
                                  key="user_input")

    if st.button("Interact"):
        if user_input:
            # Generate mother's response
            stimulus = st.session_state.child.mother.generate_stimulus(
                st.session_state.child.curriculum.current_stage,
                user_input
            )
            
            # Update child's emotional state
            st.session_state.child.update_emotions(stimulus['emotional_vector'])
            
            # Process child's perception and response
            perception = st.session_state.child.perceive(stimulus)
            response = st.session_state.child.respond(perception)
            
            # Add to conversation history
            st.session_state.conversation_history.append({
                "user": user_input,
                "mother": stimulus['text'],
                "child": response,
                "emotion": st.session_state.child.express_feeling()
            })
            
            # Store emotional state history
            st.session_state.emotional_history.append(
                st.session_state.child.emotional_state.cpu().numpy()
            )
    
    # Display conversation history
    st.subheader("Conversation History")
    for interaction in reversed(st.session_state.conversation_history):
        with st.expander(f"Interaction (Child feeling: {interaction['emotion']})"):
            st.write("You:", interaction['user'])
            st.write("Mother:", interaction['mother'])
            st.write("Child:", interaction['child'])
    
    # Development progress graphs
    if len(st.session_state.development_metrics['success_rate']) > 0:
        st.subheader("Development Progress")
        progress_fig = go.Figure()
        
        for metric in st.session_state.development_metrics:
            progress_fig.add_trace(go.Scatter(
                y=st.session_state.development_metrics[metric],
                name=metric.replace('_', ' ').title()
            ))
        
        progress_fig.update_layout(
            title="Development Metrics Over Time",
            xaxis_title="Interactions",
            yaxis_title="Score"
        )
        
        st.plotly_chart(progress_fig)
    
    # Add development metrics visualization
    if st.session_state.conversation_history:
        st.subheader("Development Metrics")
        metrics_fig = make_subplots(rows=1, cols=3, 
                                  subplot_titles=("Social Awareness", 
                                                "Language Complexity", 
                                                "Emotional Stability"))
        
        # Extract metrics from interaction history
        if 'metrics_history' not in st.session_state:
            st.session_state.metrics_history = {
                'social': [],
                'language': [],
                'emotional': []
            }
        
        # Update visualizations
        for idx, (key, values) in enumerate(st.session_state.metrics_history.items()):
            metrics_fig.add_trace(
                go.Scatter(y=values, name=key.capitalize()),
                row=1, col=idx+1
            )
        
        metrics_fig.update_layout(height=300, showlegend=False)
        st.plotly_chart(metrics_fig)
    
    # Save/Load functionality
    st.subheader("Save/Load System")
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Save Child State"):
            torch.save(st.session_state.child.brain.state_dict(), 
                      f"digital_child_{st.session_state.child.age()}mo.pth")
            st.success("Child state saved successfully!")
    
    with col2:
        uploaded_file = st.file_uploader("Load Child State", type="pth")
        if uploaded_file is not None:
            state_dict = torch.load(uploaded_file)
            st.session_state.child.brain.load_state_dict(state_dict)
            st.success("Child state loaded successfully!")

if __name__ == "__main__":
    main()