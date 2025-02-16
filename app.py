import streamlit as st
import torch
import time
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from developmental_stages import DevelopmentalStage
from config import STAGE_DEFINITIONS

try:
    from main import DigitalChild, MotherLLM
except ImportError:
    st.error("Failed to import DigitalChild and MotherLLM. Make sure main.py exists and is properly configured.")
    DigitalChild = None
    MotherLLM = None

# Initialize session state
if DigitalChild is not None:
    if 'initialized' not in st.session_state:
        try:
            st.session_state.child = DigitalChild()
            st.session_state.mother = MotherLLM()
            st.session_state.conversation_history = []
            st.session_state.emotional_history = []
            st.session_state.learning_history = []
            st.session_state.milestone_history = []
            st.session_state.complexity_history = []
            st.session_state.teaching_history = []
            st.session_state.development_metrics = {
                'success_rate': [],
                'abstraction': [],
                'self_awareness': [],
                'complexity_level': [],
                'emotional_stability': []
            }
            st.session_state.initialized = True
        except Exception as e:
            st.error(f"Error initializing digital child: {str(e)}")

def calculate_complexity_level():
    """Calculate current complexity level based on various factors"""
    child = st.session_state.child
    stage_value = child.curriculum.current_stage.value
    emotional_complexity = np.mean(child.emotional_state.cpu().numpy())
    learning_progress = len(st.session_state.learning_history)
    return (stage_value * 0.4 + emotional_complexity * 0.3 + 
            (learning_progress/100) * 0.3) * 10  # Scale to 0-100

def render_milestone_timeline():
    """Render interactive milestone timeline"""
    milestones = st.session_state.milestone_history
    if not milestones:
        return
    
    df = pd.DataFrame(milestones)
    fig = px.timeline(df, x_start='date', x_end='date',
                     y='category', color='type',
                     hover_data=['description'])
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

def render_complexity_gauge():
    """Render complexity level gauge"""
    complexity = calculate_complexity_level()
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=complexity,
        title={'text': "Complexity Level"},
        gauge={'axis': {'range': [0, 100]},
               'steps': [
                   {'range': [0, 33], 'color': "lightgray"},
                   {'range': [33, 66], 'color': "gray"},
                   {'range': [66, 100], 'color': "darkgray"}
               ],
               'threshold': {
                   'line': {'color': "red", 'width': 4},
                   'thickness': 0.75,
                   'value': complexity
               }}))
    st.plotly_chart(fig, use_container_width=True)

def create_emotion_radar_chart(emotional_state):
    """Create radar chart for emotional state visualization"""
    categories = ['Happiness', 'Trust', 'Fear', 'Surprise', 'Anger', 'Sadness']
    values = list(emotional_state) + [emotional_state[0]]  # Close the polygon
    
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories + [categories[0]],
        fill='toself',
        name='Current Emotional State'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )),
        showlegend=False,
        height=300
    )
    return fig

def calculate_emotional_stability():
    """Calculate emotional stability percentage"""
    if not st.session_state.emotional_history:
        return 0.0
    
    recent_states = st.session_state.emotional_history[-10:]
    variations = []
    for i in range(1, len(recent_states)):
        variation = np.mean(np.abs(np.array(recent_states[i]) - np.array(recent_states[i-1])))
        variations.append(variation)
    
    stability = 100 * (1 - np.mean(variations))
    return max(0, min(100, stability))

def create_adaptation_chart():
    """Create chart showing mother's teaching style adaptation"""
    if not st.session_state.teaching_history:
        return go.Figure()
    
    df = pd.DataFrame(st.session_state.teaching_history)
    fig = go.Figure()
    
    # Plot teaching style changes
    fig.add_trace(go.Scatter(
        x=df['date'],
        y=df['complexity'],
        name='Teaching Complexity',
        mode='lines+markers'
    ))
    
    fig.add_trace(go.Scatter(
        x=df['date'],
        y=df['effectiveness'],
        name='Teaching Effectiveness',
        mode='lines+markers'
    ))
    
    fig.update_layout(
        height=300,
        xaxis_title="Time",
        yaxis_title="Score"
    )
    return fig

def create_cognitive_development_chart():
    """Create cognitive development visualization"""
    metrics = st.session_state.development_metrics
    
    fig = go.Figure()
    for metric in ['abstraction', 'self_awareness', 'complexity_level']:
        if metrics[metric]:
            fig.add_trace(go.Scatter(
                y=metrics[metric],
                name=metric.replace('_', ' ').title(),
                mode='lines+markers'
            ))
    
    fig.update_layout(
        height=300,
        xaxis_title="Interactions",
        yaxis_title="Score"
    )
    return fig

def create_emotional_development_chart():
    """Create emotional development visualization"""
    if not st.session_state.emotional_history:
        return go.Figure()
    
    emotion_labels = ['Happiness', 'Trust', 'Fear', 'Surprise']
    fig = go.Figure()
    
    for i, emotion in enumerate(emotion_labels):
        values = [state[i] * 100 for state in st.session_state.emotional_history]
        fig.add_trace(go.Scatter(
            y=values,
            name=emotion,
            mode='lines'
        ))
    
    fig.update_layout(
        height=300,
        xaxis_title="Time",
        yaxis_title="Intensity (%)"
    )
    return fig

def render_learning_achievements():
    """Display learning achievements"""
    achievements = [
        item for item in st.session_state.milestone_history 
        if item['type'] == 'learning'
    ]
    
    if not achievements:
        st.write("No achievements recorded yet.")
        return
    
    for achievement in achievements[-5:]:  # Show last 5 achievements
        with st.expander(f"{achievement['date']} - {achievement['description']}", expanded=False):
            st.write(f"Category: {achievement['category']}")
            st.write(f"Impact: {achievement['impact']}")
            if 'milestone' in achievement:
                st.write(f"Milestone: {achievement['milestone']}")
                st.write(f"Stage: {achievement['stage']}")

def render_upcoming_milestones():
    """Display upcoming milestones"""
    stage = st.session_state.child.curriculum.current_stage
    stage_reqs = STAGE_DEFINITIONS[stage]
    
    st.write("Current Milestones:")
    for milestone in stage_reqs.current_milestones:
        st.write(f"✓ {milestone}")
    
    st.write("\nUpcoming Milestones:")
    for milestone in stage_reqs.upcoming_milestones:
        st.write(f"○ {milestone}")
    
    # If available, show next stage's milestones
    if stage.value < len(DevelopmentalStage) - 1:
        next_stage = DevelopmentalStage(stage.value + 1)
        next_stage_reqs = STAGE_DEFINITIONS[next_stage]
        st.write(f"\nNext Stage ({next_stage.name}) Milestones:")
        for milestone in next_stage_reqs.current_milestones:
            st.write(f"◇ {milestone}")

def create_complexity_growth_chart():
    """Create complexity growth visualization"""
    if not st.session_state.complexity_history:
        return go.Figure()
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        y=st.session_state.complexity_history,
        mode='lines+markers',
        name='Complexity Level'
    ))
    
    fig.update_layout(
        height=300,
        xaxis_title="Time",
        yaxis_title="Complexity Level"
    )
    return fig

def create_learning_rate_chart():
    """Create learning rate visualization"""
    if not st.session_state.learning_history:
        return go.Figure()
    
    # Calculate learning rate over time
    learning_counts = np.cumsum([1] * len(st.session_state.learning_history))
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        y=learning_counts,
        mode='lines',
        name='Cumulative Learning'
    ))
    
    fig.update_layout(
        height=300,
        xaxis_title="Interactions",
        yaxis_title="Total Concepts Learned"
    )
    return fig

def create_decision_analysis_chart():
    """Create decision making analysis visualization"""
    if not hasattr(st.session_state.child, 'decision_history'):
        return go.Figure()
    
    decisions = st.session_state.child.decision_history
    if not decisions:
        return go.Figure()
    
    # Calculate success rate over time
    success_rate = [sum(decisions[:i+1])/len(decisions[:i+1]) 
                   for i in range(len(decisions))]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        y=success_rate,
        mode='lines',
        name='Decision Success Rate'
    ))
    
    fig.update_layout(
        height=300,
        xaxis_title="Decisions Made",
        yaxis_title="Success Rate"
    )
    return fig

def main():
    if DigitalChild is None:
        st.error("Cannot run application: Required modules not found")
        return
    
    if not st.session_state.get('initialized', False):
        st.error("Session state not properly initialized")
        return

    st.title("Digital Child Development System")
    
    # Top-level metrics dashboard
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Current Age", f"{st.session_state.child.age()} months")
    with col2:
        st.metric("Development Stage", st.session_state.child.curriculum.current_stage.name)
    with col3:
        st.metric("Total Interactions", len(st.session_state.conversation_history))
    with col4:
        st.metric("Learning Progress", f"{len(st.session_state.learning_history)} concepts")
    
    # Main content tabs
    tabs = st.tabs([
        "Digital Child", 
        "Mother's Interface",
        "Development Tracking",
        "Milestones & Progress",
        "Analytics"
    ])
    
    with tabs[0]:  # Digital Child Tab
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Current State & Emotions")
            current_feeling = st.session_state.child.express_feeling()
            st.info(f"Child is feeling: {current_feeling}")
            
            # Emotional State Visualization
            emotional_state = st.session_state.child.emotional_state.cpu().numpy()
            emotions_fig = create_emotion_radar_chart(emotional_state)
            st.plotly_chart(emotions_fig, use_container_width=True)
            
            # Recent Experiences
            st.subheader("Recent Experiences")
            for exp in st.session_state.learning_history[-5:]:
                st.write(f"- {exp}")
        
        with col2:
            st.subheader("Complexity Level")
            render_complexity_gauge()
            
            st.subheader("Learning Stats")
            st.metric("Concepts Learned", len(st.session_state.learning_history))
            st.metric("Emotional Stability", f"{calculate_emotional_stability():.2f}%")
    
    with tabs[1]:  # Mother's Interface Tab
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Teaching & Nurturing Interface")
            
            # Interact with Child
            st.subheader("Interact with Child")
            current_behaviors = st.session_state.child.curriculum.get_stage_requirements()['behaviors']
            template_options = list(current_behaviors['allowed_actions'])
            
            col1, col2 = st.columns([3, 1])
            with col1:
                selected_template = st.selectbox(
                    "Interaction Type:", 
                    ["Custom"] + template_options,
                    key="interaction_type"
                )
                
                if selected_template == "Custom":
                    user_input = st.text_input(
                        "Say something to the child:", 
                        key="user_input"
                    )
                else:
                    user_input = st.text_input(
                        "Say something to the child:",
                        value=f"[{selected_template.upper()}]",
                        key="user_input"
                    )
            
            with col2:
                st.write("")
                st.write("")
                interact_button = st.button(
                    "Interact", 
                    use_container_width=True,
                    key="interact_button"
                )
            
            if interact_button and user_input:
                # Generate mother's response
                stimulus = st.session_state.mother.generate_stimulus(
                    st.session_state.child.curriculum.current_stage,
                    user_input
                )
                
                # Update child's emotional state
                st.session_state.child.update_emotions(stimulus['emotional_vector'])
                
                # Process child's perception and response
                perception = st.session_state.child.perceive(stimulus)
                response = st.session_state.child.respond(perception)
                
                # Add to conversation history
                interaction_data = {
                    "timestamp": datetime.now(),
                    "user": user_input,
                    "mother": stimulus['text'],
                    "child": response,
                    "emotion": st.session_state.child.express_feeling(),
                    "stage": st.session_state.child.curriculum.current_stage.name
                }
                st.session_state.conversation_history.append(interaction_data)
                
                # Store emotional state history
                st.session_state.emotional_history.append(
                    st.session_state.child.emotional_state.cpu().numpy()
                )
                
                # Update teaching history
                teaching_data = {
                    "date": datetime.now(),
                    "topic": selected_template if selected_template != "Custom" else "Custom Interaction",
                    "method": stimulus['text'],
                    "response": response,
                    "effectiveness": float(stimulus.get('effectiveness', 0.5)),
                    "complexity": float(stimulus.get('complexity', 0.5))
                }
                st.session_state.teaching_history.append(teaching_data)
                
                # Update complexity history
                current_complexity = calculate_complexity_level()
                st.session_state.complexity_history.append(current_complexity)
                
                # Show the interaction result
                st.success("Interaction recorded!")
                with st.expander("Last Interaction", expanded=True):
                    st.caption(f"Child feeling: {interaction_data['emotion']}")
                    st.write("You:", interaction_data['user'])
                    st.write("Mother:", interaction_data['mother'])
                    st.write("Child:", interaction_data['child'])
            
            # Display recent interactions
            st.subheader("Recent Interactions")
            for interaction in reversed(st.session_state.conversation_history[-3:]):
                with st.expander(
                    f"{interaction['timestamp'].strftime('%H:%M:%S')} - {interaction['stage']}", 
                    expanded=False
                ):
                    st.caption(f"Child feeling: {interaction['emotion']}")
                    st.write("You:", interaction['user'])
                    st.write("Mother:", interaction['mother'])
                    st.write("Child:", interaction['child'])
            
            # Teaching History
            st.subheader("Teaching History")
            st.write("Recent Teaching Activities:")
            for teaching in st.session_state.teaching_history[-5:]:
                with st.expander(f"{teaching['date']} - {teaching['topic']}", expanded=False):
                    st.write(f"Method: {teaching['method']}")
                    st.write(f"Response: {teaching['response']}")
                    st.write(f"Effectiveness: {teaching['effectiveness']}")
        
        with col2:
            st.subheader("Mother's Adaptation")
            # Show how mother's teaching style adapts
            if st.session_state.teaching_history:
                adaptation_fig = create_adaptation_chart()
                st.plotly_chart(adaptation_fig, use_container_width=True)
    
    with tabs[2]:  # Development Tracking Tab
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Cognitive Development")
            cognitive_fig = create_cognitive_development_chart()
            st.plotly_chart(cognitive_fig, use_container_width=True)
        
        with col2:
            st.subheader("Emotional Development")
            emotional_fig = create_emotional_development_chart()
            st.plotly_chart(emotional_fig, use_container_width=True)
    
    with tabs[3]:  # Milestones & Progress Tab
        st.subheader("Development Milestones")
        render_milestone_timeline()
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Learning Achievements")
            render_learning_achievements()
        
        with col2:
            st.subheader("Next Milestones")
            render_upcoming_milestones()
    
    with tabs[4]:  # Analytics Tab
        st.subheader("Development Analytics")
        
        # Complexity Growth Over Time
        st.subheader("Complexity Growth")
        complexity_fig = create_complexity_growth_chart()
        st.plotly_chart(complexity_fig, use_container_width=True)
        
        # Learning Rate Analysis
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Learning Rate")
            learning_fig = create_learning_rate_chart()
            st.plotly_chart(learning_fig, use_container_width=True)
        
        with col2:
            st.subheader("Decision Making")
            decision_fig = create_decision_analysis_chart()
            st.plotly_chart(decision_fig, use_container_width=True)

    # Footer with save/load functionality
    st.divider()
    save_load_cols = st.columns([1, 1, 2])
    
    with save_load_cols[0]:
        if st.button("Save State", use_container_width=True):
            # Save both child and session state
            save_data = {
                'child_state': st.session_state.child.brain.state_dict(),
                'conversation_history': st.session_state.conversation_history,
                'emotional_history': st.session_state.emotional_history,
                'learning_history': st.session_state.learning_history,
                'milestone_history': st.session_state.milestone_history,
                'complexity_history': st.session_state.complexity_history,
                'teaching_history': st.session_state.teaching_history,
                'development_metrics': st.session_state.development_metrics
            }
            save_path = f"digital_child_{st.session_state.child.age()}mo_full.pth"
            torch.save(save_data, save_path)
            st.success(f"Full state saved to {save_path}")
    
    with save_load_cols[1]:
        uploaded_file = st.file_uploader("Load State", type="pth")
        if uploaded_file is not None:
            try:
                save_data = torch.load(uploaded_file)
                # Load child state
                st.session_state.child.brain.load_state_dict(save_data['child_state'])
                # Load session state
                st.session_state.conversation_history = save_data['conversation_history']
                st.session_state.emotional_history = save_data['emotional_history']
                st.session_state.learning_history = save_data['learning_history']
                st.session_state.milestone_history = save_data['milestone_history']
                st.session_state.complexity_history = save_data['complexity_history']
                st.session_state.teaching_history = save_data['teaching_history']
                st.session_state.development_metrics = save_data['development_metrics']
                st.success("Full state loaded successfully!")
            except Exception as e:
                st.error(f"Error loading state: {str(e)}")
    
    with save_load_cols[2]:
        st.info("Save/Load functionality preserves all history and development progress.")

if __name__ == "__main__":
    main()