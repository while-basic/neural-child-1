import streamlit as st
import numpy as np
import torch
import time
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import json
import torch.serialization
import requests
import sseclient
import threading
import queue
import os
import psutil
from developmental_stages import (
    DevelopmentalStage,
    DevelopmentalSystem,
    STAGE_DEFINITIONS,
    StageCharacteristics
)
from typing import Optional, Dict, List, Any, Union, Tuple
from pathlib import Path
from emotional_state import EmotionalState

# Define stage templates for interactions
stage_templates = {
    DevelopmentalStage.NEWBORN: [
        "[FEED] Feed the baby",
        "[COMFORT] Comfort the crying baby",
        "[SLEEP] Help the baby sleep",
        "[TALK] Talk softly to the baby"
    ],
    DevelopmentalStage.EARLY_INFANCY: [
        "[PLAY] Play peek-a-boo",
        "[SMILE] Smile at the baby",
        "[SING] Sing a lullaby",
        "[TOUCH] Gentle touch and massage"
    ],
    DevelopmentalStage.LATE_INFANCY: [
        "[CRAWL] Encourage crawling",
        "[EXPLORE] Help explore objects",
        "[BABBLE] Respond to babbling",
        "[REACH] Help reach for toys"
    ],
    DevelopmentalStage.EARLY_TODDLER: [
        "[WALK] Support walking practice",
        "[WORDS] Teach simple words",
        "[POINT] Point to objects",
        "[STACK] Stack blocks together"
    ],
    DevelopmentalStage.LATE_TODDLER: [
        "[SHARE] Practice sharing",
        "[DRAW] Draw together",
        "[DANCE] Dance and move",
        "[COUNT] Count objects"
    ],
    DevelopmentalStage.EARLY_PRESCHOOL: [
        "[READ] Read a story",
        "[COLORS] Learn colors",
        "[SHAPES] Identify shapes",
        "[PRETEND] Pretend play"
    ],
    DevelopmentalStage.LATE_PRESCHOOL: [
        "[WRITE] Practice writing",
        "[PUZZLE] Solve puzzles",
        "[FRIENDS] Make friends",
        "[CREATE] Creative activities"
    ],
    DevelopmentalStage.EARLY_CHILDHOOD: [
        "[LEARN] Basic math concepts",
        "[WRITE] Write simple words",
        "[SCIENCE] Simple experiments",
        "[SOCIAL] Group activities"
    ],
    DevelopmentalStage.MIDDLE_CHILDHOOD: [
        "[HOMEWORK] Help with homework",
        "[SPORTS] Physical activities",
        "[MUSIC] Music lessons",
        "[PROJECT] School projects"
    ],
    DevelopmentalStage.LATE_CHILDHOOD: [
        "[RESEARCH] Research topics",
        "[DEBATE] Practice debating",
        "[GOALS] Set personal goals",
        "[SKILLS] Learn new skills"
    ],
    DevelopmentalStage.EARLY_ELEMENTARY: [
        "[STUDY] Study techniques",
        "[ORGANIZE] Organization skills",
        "[PRESENT] Presentation skills",
        "[TEAM] Team projects"
    ],
    DevelopmentalStage.MIDDLE_ELEMENTARY: [
        "[ANALYZE] Critical thinking",
        "[EXPERIMENT] Scientific method",
        "[WRITE] Essay writing",
        "[LEAD] Leadership activities"
    ],
    DevelopmentalStage.LATE_ELEMENTARY: [
        "[RESEARCH] Independent research",
        "[CRITIQUE] Critical analysis",
        "[DEBATE] Advanced debates",
        "[CREATE] Creative projects"
    ],
    DevelopmentalStage.EARLY_ADOLESCENCE: [
        "[IDENTITY] Self-discovery",
        "[FUTURE] Career exploration",
        "[ETHICS] Moral discussions",
        "[SOCIAL] Social skills"
    ],
    DevelopmentalStage.MIDDLE_ADOLESCENCE: [
        "[PLAN] Life planning",
        "[VALUES] Personal values",
        "[CAREER] Career guidance",
        "[RESPONSIBILITY] Taking responsibility"
    ],
    DevelopmentalStage.LATE_ADOLESCENCE: [
        "[DECIDE] Decision making",
        "[PREPARE] College preparation",
        "[INDEPENDENCE] Independent living",
        "[GOALS] Long-term goals"
    ],
    DevelopmentalStage.YOUNG_ADULT: [
        "[CAREER] Career development",
        "[FINANCE] Financial planning",
        "[RELATIONSHIP] Relationship advice",
        "[LIFE] Life skills"
    ],
    DevelopmentalStage.MATURE_ADULT: [
        "[WISDOM] Share wisdom",
        "[MENTOR] Mentorship",
        "[LEGACY] Build legacy",
        "[REFLECT] Life reflection"
    ]
}

try:
    from main import DigitalChild, MotherLLM
except ImportError:
    st.error("Failed to import DigitalChild and MotherLLM. Make sure main.py exists and is properly configured.")
    DigitalChild = None
    MotherLLM = None

def validate_stage_definitions():
    """Validate that all developmental stages have proper definitions"""
    missing_stages = []
    
    # First verify STAGE_DEFINITIONS is imported correctly
    if not STAGE_DEFINITIONS:
        return ["Error: STAGE_DEFINITIONS is empty or not imported correctly"]
    
    for stage in DevelopmentalStage:
        try:
            # Verify stage exists in STAGE_DEFINITIONS
            if stage not in STAGE_DEFINITIONS:
                missing_stages.append(f"{stage.name} (missing definition)")
                continue
                
            # Get stage definition
            stage_def = STAGE_DEFINITIONS[stage]
            
            # Verify required fields
            required_fields = {
                'age_range': tuple,
                'complexity_range': tuple,
                'emotional_range': tuple,
                'required_skills': list,
                'learning_focus': list,
                'current_milestones': list,
                'upcoming_milestones': list
            }
            
            for field, expected_type in required_fields.items():
                if field not in stage_def:
                    missing_stages.append(f"{stage.name} (missing field: {field})")
                elif not isinstance(stage_def[field], expected_type):
                    missing_stages.append(
                        f"{stage.name} (invalid type for {field}: expected {expected_type.__name__})"
                    )
            
            # Verify field contents
            if stage_def.get('age_range') and len(stage_def['age_range']) != 2:
                missing_stages.append(f"{stage.name} (invalid age_range format)")
            
            if stage_def.get('complexity_range') and len(stage_def['complexity_range']) != 2:
                missing_stages.append(f"{stage.name} (invalid complexity_range format)")
            
            if stage_def.get('emotional_range') and len(stage_def['emotional_range']) != 2:
                missing_stages.append(f"{stage.name} (invalid emotional_range format)")
            
            # Create a temporary system to test stage requirements
            system = DevelopmentalSystem()
            system.current_stage = stage
            
            # Try to get requirements for each stage
            reqs = system.get_stage_requirements()
            
            # Verify behaviors exist
            if not reqs.get('behaviors'):
                missing_stages.append(f"{stage.name} (missing behaviors)")
            elif not isinstance(reqs['behaviors'], list):
                missing_stages.append(f"{stage.name} (invalid behaviors format)")
            elif len(reqs['behaviors']) < 1:
                missing_stages.append(f"{stage.name} (empty behaviors list)")
                
        except Exception as e:
            missing_stages.append(f"{stage.name} (error: {str(e)})")
            
    return missing_stages

def initialize_new_session():
    """Initialize a new session with default values"""
    # Validate stage definitions
    missing_stages = validate_stage_definitions()
    if missing_stages:
        st.error(f"Missing stage definitions for: {', '.join(missing_stages)}")
        st.stop()
    
    st.session_state.child = DigitalChild()
    st.session_state.birth_time = datetime.now()  # Store birth time in session state
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
    st.info("New session initialized")

# Initialize session state
if DigitalChild is not None:
    if 'initialized' not in st.session_state:
        try:
            # Check for latest backup
            backup_path = os.path.join("checkpoints", "digital_child_state_latest.pth")
            if os.path.exists(backup_path):
                try:
                    # Add datetime to safe globals
                    torch.serialization.add_safe_globals(['datetime'])
                    
                    # Load the backup
                    save_data = torch.load(
                        backup_path,
                        weights_only=False,
                        map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                    )
                    
                    # Initialize child and mother
                    st.session_state.child = DigitalChild()
                    st.session_state.mother = MotherLLM()
                    
                    # Load child state
                    try:
                        st.session_state.child.brain.load_state_dict(save_data['child_state_dict'])
                    except Exception as model_e:
                        st.warning(f"Could not load model state due to architecture mismatch: {str(model_e)}")
                        st.info("Continuing with fresh model weights but preserving other state data.")
                    
                    # Restore emotional state
                    if isinstance(save_data['emotional_state'], dict):
                        emotional_values = [
                            save_data['emotional_state']['happiness'],
                            save_data['emotional_state']['sadness'],
                            save_data['emotional_state']['anger'],
                            save_data['emotional_state']['fear']
                        ]
                        st.session_state.child.emotional_state = torch.tensor(
                            emotional_values,
                            device=st.session_state.child.device
                        )
                    else:
                        st.session_state.child.emotional_state = torch.tensor(
                            save_data['emotional_state'],
                            device=st.session_state.child.device
                        )
                    
                    # Restore session state
                    st.session_state.birth_time = datetime.fromisoformat(save_data['birth_date'])
                    st.session_state.conversation_history = [
                        {**item, 'timestamp': datetime.fromisoformat(item['timestamp']) 
                         if isinstance(item['timestamp'], str) else item['timestamp']}
                        for item in save_data['conversation_history']
                    ]
                    st.session_state.emotional_history = [
                        torch.tensor(e, device=st.session_state.child.device) 
                        if isinstance(e, list) else e 
                        for e in save_data['emotional_history']
                    ]
                    st.session_state.learning_history = save_data['learning_history']
                    st.session_state.milestone_history = save_data['milestone_history']
                    st.session_state.complexity_history = save_data['complexity_history']
                    st.session_state.teaching_history = [
                        {**item, 'date': datetime.fromisoformat(item['date']) 
                         if isinstance(item['date'], str) else item['date']}
                        for item in save_data['teaching_history']
                    ]
                    st.session_state.development_metrics = save_data['development_metrics']
                    
                    # Restore current stage
                    if 'current_stage' in save_data:
                        st.session_state.child.curriculum.current_stage = DevelopmentalStage[save_data['current_stage']]
                    
                    st.session_state.initialized = True
                    st.success("Previous state loaded automatically!")
                    
                except Exception as e:
                    st.error(f"Error loading backup state: {str(e)}")
                    if st.sidebar.checkbox("Debug Mode", key="debug_mode_init", value=False):
                        st.exception(e)
                    # Fall back to new initialization
                    initialize_new_session()
            else:
                # No backup found, initialize new session
                initialize_new_session()
                
        except Exception as e:
            st.error(f"Error initializing digital child: {str(e)}")
            if st.sidebar.checkbox("Debug Mode", key="debug_mode_init", value=False):
                st.exception(e)

def ensure_tensor_device(tensor, target_device=None):
    """Ensure tensor is on the correct device"""
    if target_device is None:
        target_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if isinstance(tensor, torch.Tensor):
        return tensor.to(target_device)
    return tensor

def calculate_complexity_level():
    """Calculate current complexity level based on multiple factors.
    
    Returns:
        Tuple of (complexity_score, contributing_factors)
    """
    if 'child' not in st.session_state:
        return 0.0, {}
        
    child = st.session_state.child
    
    # Get metrics from various systems
    metacog_metrics = child.metacognition.get_metrics()
    learning_rate = child.autonomous_learner.get_learning_rate()
    
    # Convert emotional state to EmotionalState if needed
    emotional_state = child.emotional_state
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
    
    # Safely convert emotional state values to float
    emotional_values = np.array([
        float(emotional_state.happiness),
        float(emotional_state.sadness),
        float(emotional_state.anger),
        float(emotional_state.fear),
        float(emotional_state.surprise),
        float(emotional_state.disgust),
        float(emotional_state.trust),
        float(emotional_state.anticipation)
    ])
    
    # Calculate component scores
    cognitive_score = metacog_metrics.get('learning_efficiency', 0.0)
    attention_score = metacog_metrics.get('attention_focus', 0.0)
    emotional_score = float(child.acceleration_metrics.get('emotional_stability', 0.0))
    
    # Weight the components
    weights = {
        'cognitive_development': 0.35,
        'attention_capacity': 0.25,
        'emotional_stability': 0.25,
        'learning_rate': 0.15
    }
    
    # Calculate weighted scores
    contributing_factors = {
        'cognitive_development': cognitive_score * weights['cognitive_development'],
        'attention_capacity': attention_score * weights['attention_capacity'],
        'emotional_stability': emotional_score * weights['emotional_stability'],
        'learning_rate': min(learning_rate * 100, 1.0) * weights['learning_rate']
    }
    
    # Calculate total complexity score
    total_score = sum(contributing_factors.values())
    
    return total_score, contributing_factors

def render_complexity_gauge():
    """Render complexity level gauge"""
    complexity, _ = calculate_complexity_level()
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
    """Create radar chart for emotional state visualization.
    
    Args:
        emotional_state: EmotionalState object or tensor containing emotion values
        
    Returns:
        Plotly figure object with the radar chart
    """
    try:
        # Convert tensor to EmotionalState if needed
        if isinstance(emotional_state, torch.Tensor):
            if emotional_state.dim() == 1:
                emotional_state = EmotionalState(
                    happiness=float(emotional_state[0].cpu()),
                    sadness=float(emotional_state[1].cpu()),
                    anger=float(emotional_state[2].cpu()),
                    fear=float(emotional_state[3].cpu()),
                    surprise=float(emotional_state[4].cpu()) if len(emotional_state) > 4 else 0.0,
                    disgust=float(emotional_state[5].cpu()) if len(emotional_state) > 5 else 0.0,
                    trust=float(emotional_state[6].cpu()) if len(emotional_state) > 6 else 0.5,
                    anticipation=float(emotional_state[7].cpu()) if len(emotional_state) > 7 else 0.5
                )
    
        # Ensure we're working with CPU values
        categories = ['Happiness', 'Trust', 'Fear', 'Surprise', 'Sadness', 'Anger', 'Disgust', 'Anticipation']
        
        # Get values and handle potential CUDA tensors
        values = [
            float(emotional_state.happiness),
            float(emotional_state.trust),
            float(emotional_state.fear),
            float(emotional_state.surprise),
            float(emotional_state.sadness),
            float(emotional_state.anger),
            float(emotional_state.disgust),
            float(emotional_state.anticipation)
        ]
        
        # Normalize values to [0,1] range
        values = [max(0.0, min(1.0, v)) for v in values]
        
        # Add first value again to close the polygon
        values.append(values[0])
        categories.append(categories[0])
        
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            fillcolor='rgba(64, 144, 248, 0.3)',
            line=dict(color='rgb(64, 144, 248)', width=2),
            name='Current State'
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1],
                    tickfont=dict(size=10),
                    gridcolor='rgba(0,0,0,0.1)',
                    linecolor='rgba(0,0,0,0.1)',
                ),
                angularaxis=dict(
                    tickfont=dict(size=10),
                    gridcolor='rgba(0,0,0,0.1)',
                    linecolor='rgba(0,0,0,0.1)',
                ),
                bgcolor='rgba(255,255,255,0.9)'
            ),
            showlegend=False,
            margin=dict(l=80, r=80, t=20, b=20),
            height=400,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        
        return fig
    except Exception as e:
        st.error(f"Error creating emotion radar chart: {str(e)}")
        # Return an empty figure as fallback
        return go.Figure()

def calculate_emotional_stability():
    """Calculate emotional stability percentage."""
    if not hasattr(st.session_state, 'emotional_history'):
        st.session_state.emotional_history = []
    
    if not st.session_state.emotional_history:
        return 0.0
    
    recent_states = st.session_state.emotional_history[-10:]
    if len(recent_states) < 2:
        return 0.0
    
    variations = []
    for i in range(1, len(recent_states)):
        # Convert tensors to EmotionalState if needed
        curr_state = recent_states[i]
        prev_state = recent_states[i-1]
        
        if isinstance(curr_state, torch.Tensor):
            curr_state = EmotionalState(
                happiness=float(curr_state[0]),
                sadness=float(curr_state[1]),
                anger=float(curr_state[2]),
                fear=float(curr_state[3]),
                surprise=float(curr_state[4]) if len(curr_state) > 4 else 0.0,
                disgust=float(curr_state[5]) if len(curr_state) > 5 else 0.0,
                trust=float(curr_state[6]) if len(curr_state) > 6 else 0.5,
                anticipation=float(curr_state[7]) if len(curr_state) > 7 else 0.5
            )
            
        if isinstance(prev_state, torch.Tensor):
            prev_state = EmotionalState(
                happiness=float(prev_state[0]),
                sadness=float(prev_state[1]),
                anger=float(prev_state[2]),
                fear=float(prev_state[3]),
                surprise=float(prev_state[4]) if len(prev_state) > 4 else 0.0,
                disgust=float(prev_state[5]) if len(prev_state) > 5 else 0.0,
                trust=float(prev_state[6]) if len(prev_state) > 6 else 0.5,
                anticipation=float(prev_state[7]) if len(prev_state) > 7 else 0.5
            )
        
        # Calculate variation using EmotionalState values
        curr_values = np.array(curr_state.to_vector())
        prev_values = np.array(prev_state.to_vector())
        variation = np.mean(np.abs(curr_values - prev_values))
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

def track_milestone_progress(milestone, progress_value):
    """Track progress towards individual milestones"""
    if 'milestone_progress' not in st.session_state:
        st.session_state.milestone_progress = {}
    
    st.session_state.milestone_progress[milestone] = progress_value

def render_development_progress():
    """Display detailed development progress metrics"""
    if not hasattr(st.session_state.child, 'curriculum'):
        return
        
    # Get current development status
    status = st.session_state.child.curriculum.get_development_status()
    
    st.subheader("📊 Development Progress")
    
    # Display current stage and duration
    col1, col2 = st.columns(2)
    with col1:
        st.metric(
            "Current Stage",
            status['current_stage'],
            help="""Current developmental stage of the child.
            Each stage represents a distinct period of growth with specific milestones and requirements.
            Progress through stages is based on skill mastery and interaction diversity."""
        )
    with col2:
        st.metric(
            "Stage Duration",
            f"{status['stage_duration']} interactions",
            help="""Number of interactions completed in the current stage.
            More interactions generally lead to better skill development.
            Different stages may require different numbers of interactions for mastery."""
        )
    
    # Display skill progress
    st.subheader("Skill Development")
    skill_cols = st.columns(4)
    
    # Define skill descriptions for tooltips
    skill_descriptions = {
        'emotional_expression': """Ability to express and communicate emotions.
        Developed through social interactions and emotional responses.
        Key for social bonding and communication.""",
        
        'emotional_response': """Ability to respond appropriately to emotional stimuli.
        Includes recognition of others' emotions and appropriate reactions.
        Important for social development and empathy.""",
        
        'voice_recognition': """Ability to recognize and respond to familiar voices.
        Crucial for early social development and bonding.
        Develops through consistent vocal interactions.""",
        
        'visual_tracking': """Ability to follow moving objects with eyes.
        Important for spatial awareness and motor development.
        Develops through visual stimulation and tracking exercises.""",
        
        'sound_response': """Ability to respond to various sounds and stimuli.
        Important for auditory processing and language development.
        Develops through diverse auditory experiences.""",
        
        'vocalization': """Ability to produce sounds and early speech.
        Critical for language development and communication.
        Progresses from cooing to babbling to early words.""",
        
        'social_bonding': """Ability to form and maintain social connections.
        Essential for emotional development and relationships.
        Develops through positive social interactions.""",
        
        'motor_skills': """Physical movement and coordination abilities.
        Includes both fine and gross motor skills.
        Develops through physical activities and practice."""
    }
    
    for i, (skill, progress) in enumerate(status['skill_progress'].items()):
        with skill_cols[i % 4]:
            st.metric(
                skill.replace('_', ' ').title(),
                f"{progress*100:.1f}%",
                help=skill_descriptions.get(skill, f"Development progress for {skill.replace('_', ' ').title()}")
            )
            st.progress(progress)
    
    # Display interaction counts
    st.subheader("Interaction History")
    
    # Define interaction type descriptions
    interaction_descriptions = {
        'learning': """Educational and cognitive development interactions.
        Includes activities that promote understanding and skill acquisition.
        Important for cognitive growth and knowledge building.""",
        
        'emotional': """Interactions focused on emotional development.
        Includes expressing and processing emotions.
        Critical for emotional intelligence and regulation.""",
        
        'social': """Social engagement and relationship building.
        Includes communication and interpersonal skills.
        Essential for social development and understanding.""",
        
        'physical': """Physical activities and motor skill development.
        Includes both fine and gross motor activities.
        Important for physical development and coordination."""
    }
    
    interaction_cols = st.columns(len(status['interaction_counts']))
    for col, (interaction, count) in zip(interaction_cols, status['interaction_counts'].items()):
        with col:
            st.metric(
                interaction.replace('_', ' ').title(),
                count,
                help=interaction_descriptions.get(interaction, f"Number of {interaction.replace('_', ' ')} interactions")
            )
    
    # Show progression readiness
    st.subheader("Stage Progression")
    if status['ready_for_progression']:
        st.success("✨ Ready to progress to next stage!")
        st.info("""
        **Requirements Met:**
        - Basic skills mastered (>70% completion)
        - Sufficient interaction diversity (>5 of each type)
        - Emotional readiness achieved
        - Overall development score >70%
        """)
    else:
        # Show what's missing for progression
        missing_requirements = []
        if 'metrics' in status:
            metrics = status['metrics']
            if metrics.get('average_completion', 0) < 0.7:
                missing_requirements.append("- Basic skills need more development (>70% required)")
            if status['stage_duration'] < 10:
                missing_requirements.append("- More interactions needed (minimum 10)")
            if any(count < 5 for count in status['interaction_counts'].values()):
                missing_requirements.append("- More diverse interactions required (>5 of each type)")
        
        if missing_requirements:
            st.info("Requirements for progression:")
            for req in missing_requirements:
                st.write(req)
        else:
            st.info("Continue interactions to develop required skills")
        
    # Show detailed metrics
    with st.expander("View Detailed Metrics", expanded=False):
        if 'metrics' in status:
            metrics_descriptions = {
                'average_completion': "Average completion rate across all skills",
                'objectives_completed': "Number of learning objectives fully mastered",
                'total_objectives': "Total number of learning objectives for this stage",
                'stage_progress': "Overall progress in current stage (capped at 50 interactions)",
                'skill_mastery': "Average mastery level across all skills"
            }
            
            for metric, value in status['metrics'].items():
                st.metric(
                    metric.replace('_', ' ').title(),
                    f"{value*100:.1f}%" if isinstance(value, float) else value,
                    help=metrics_descriptions.get(metric, f"Metric: {metric.replace('_', ' ').title()}")
                )

def render_upcoming_milestones():
    """Display upcoming milestones with progress bars"""
    try:
        stage = st.session_state.child.curriculum.current_stage
        
        # Initialize milestone progress if not exists
        if 'milestone_progress' not in st.session_state:
            st.session_state.milestone_progress = {}
        
        # Get stage requirements using the new API
        stage_reqs = st.session_state.child.curriculum.get_stage_requirements()
        
        # Get current development status
        dev_status = st.session_state.child.curriculum.get_development_status()
        
        # Ensure all required skills are initialized
        required_skills = {
            'motor_skills': 0.0,
            'emotional_expression': 0.0,
            'sound_response': 0.0,
            'voice_recognition': 0.0,
            'visual_tracking': 0.0,
            'vocalization': 0.0,
            'social_bonding': 0.0,
            'object_permanence': 0.0
        }
        
        # Update with actual progress where available
        if 'skill_progress' in dev_status:
            for skill, progress in dev_status['skill_progress'].items():
                required_skills[skill] = progress
        
        # Display current milestones with completion status
        st.write("Current Milestones:")
        current_milestones = stage_reqs.get('current_milestones', [])
        if current_milestones:
            for milestone in current_milestones:
                # Calculate progress based on relevant skills
                skill_mapping = {
                    'Basic reflexes': ['motor_skills'],
                    'Can cry to express needs': ['emotional_expression', 'sound_response'],
                    'Recognize mother\'s voice': ['voice_recognition'],
                    'Follow moving objects with eyes': ['visual_tracking'],
                    'Respond to loud sounds': ['sound_response'],
                    'Make cooing sounds': ['vocalization']
                }
                
                if milestone in skill_mapping:
                    # Safely get progress for each skill
                    skills = skill_mapping[milestone]
                    progress = np.mean([
                        required_skills.get(skill, 0.0)
                        for skill in skills
                    ])
                else:
                    progress = st.session_state.milestone_progress.get(milestone, 0.0)
                
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.write(f"{'✓' if progress > 0.9 else '○'} {milestone}")
                with col2:
                    st.progress(progress)
        else:
            st.write("No current milestones defined")
        
        # Display upcoming milestones with progress
        st.write("\nUpcoming Milestones:")
        upcoming_milestones = stage_reqs.get('upcoming_milestones', [])
        if upcoming_milestones:
            for milestone in upcoming_milestones:
                # Calculate progress based on relevant skills
                skill_mapping = {
                    'Social smiling': ['social_bonding', 'emotional_expression'],
                    'Object manipulation': ['motor_skills', 'object_permanence'],
                    'Babbling': ['vocalization', 'sound_response'],
                    'Sitting without support': ['motor_skills'],
                    'Track moving objects': ['visual_tracking'],
                    'Respond to sounds': ['sound_response']
                }
                
                if milestone in skill_mapping:
                    # Safely get progress for each skill
                    skills = skill_mapping[milestone]
                    progress = np.mean([
                        required_skills.get(skill, 0.0)
                        for skill in skills
                    ])
                else:
                    progress = st.session_state.milestone_progress.get(milestone, 0.0)
                
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.write(f"{'○' if progress > 0.5 else '◇'} {milestone}")
                with col2:
                    st.progress(progress)
        else:
            st.write("No upcoming milestones defined")
        
        # Display progression hints
        if not dev_status['ready_for_progression']:
            st.info("💡 **Development Hints:**")
            hints = []
            
            # Add specific hints based on skill progress
            for skill, progress in dev_status['skill_progress'].items():
                if progress < 0.6:
                    hint_mapping = {
                        'voice_recognition': "Try talking and singing more to help with voice recognition",
                        'visual_tracking': "Show moving objects and encourage eye tracking",
                        'sound_response': "Make different sounds and observe responses",
                        'vocalization': "Encourage babbling and cooing through interaction",
                        'social_bonding': "Increase face-to-face interaction and emotional exchanges",
                        'motor_skills': "Provide opportunities for physical movement and reaching"
                    }
                    if skill in hint_mapping:
                        hints.append(hint_mapping[skill])
            
            for hint in hints:
                st.write(f"- {hint}")
    
    except Exception as e:
        st.error("Error displaying milestones")
        if st.sidebar.checkbox("Debug Mode", value=False):
            st.exception(e)
            st.write("Debug Info:")
            st.write("- Current stage:", getattr(stage, 'name', 'Unknown') if 'stage' in locals() else "Not available")
            st.write("- Available stages:", [s.name for s in DevelopmentalStage])

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

def format_json_response(response_data):
    """Format JSON response for better readability"""
    try:
        if isinstance(response_data, str):
            response_data = json.loads(response_data)
        return json.dumps(response_data, indent=2)
    except Exception:
        return str(response_data)

def stream_llm_logs():
    """Stream logs from LM Studio server"""
    try:
        response = requests.get('http://localhost:1234/v1/chat/completions', stream=True)
        client = sseclient.SSEClient(response)
        for event in client.events():
            if event.data:
                yield event.data
    except Exception as e:
        yield f"Error connecting to LM Studio server: {str(e)}"

def add_time_controls():
    """Add time acceleration controls to the sidebar"""
    st.sidebar.markdown("---")
    st.sidebar.subheader("⏰ Time Controls")
    
    # Add speed multiplier slider with warning indicators
    if 'speed_multiplier' not in st.session_state:
        st.session_state.speed_multiplier = 1.0
    
    # Get warning indicators
    warning_indicators = st.session_state.child.get_warning_indicators()
    warning_state = warning_indicators['warning_state']
    
    # Display warning state with color coding
    warning_colors = {
        "RED": "🔴",
        "YELLOW": "🟡",
        "GREEN": "🟢"
    }
    
    st.sidebar.markdown(f"**Current Warning State: {warning_colors[warning_state]} {warning_state}**")
    
    # Display warning metrics
    with st.sidebar.expander("Warning Metrics", expanded=True):
        metrics = warning_indicators['metrics']
        
        # Emotional Stability
        stability = metrics['emotional_stability']
        stability_color = "🟢" if stability > 0.6 else "🟡" if stability > 0.3 else "🔴"
        st.markdown(f"{stability_color} Emotional Stability: {stability:.2%}")
        
        # Learning Efficiency
        efficiency = metrics['learning_efficiency']
        efficiency_color = "🟢" if efficiency > 0.6 else "🟡" if efficiency > 0.4 else "🔴"
        st.markdown(f"{efficiency_color} Learning Efficiency: {efficiency:.2%}")
        
        # Attention Level
        attention = metrics['attention_level']
        attention_color = "🟢" if attention > 0.6 else "🟡" if attention > 0.3 else "🔴"
        st.markdown(f"{attention_color} Attention Level: {attention:.2%}")
        
        # Overstimulation Risk
        risk = metrics['overstimulation_risk']
        risk_color = "🟢" if risk < 0.4 else "🟡" if risk < 0.7 else "🔴"
        st.markdown(f"{risk_color} Overstimulation Risk: {risk:.2%}")
    
    # Show recent warnings if any
    if warning_indicators['recent_warnings']:
        with st.sidebar.expander("Recent Warnings", expanded=False):
            for warning in warning_indicators['recent_warnings']:
                st.markdown(f"**{warning['timestamp'].strftime('%H:%M:%S')}** - {warning['state']}")
                st.markdown(f"_{warning['reason']}_")
                st.markdown("---")
    
    # Add speed multiplier slider with dynamic max value and safety override
    current_speed = warning_indicators['speed_multiplier']
    
    # Display current speed with color indicator
    speed_color = "🟢" if current_speed <= 200 else "🟡" if current_speed <= 350 else "🔴"
    st.sidebar.markdown(f"**Current Speed: {speed_color} {current_speed:.1f}%**")
    
    # Add safety override checkbox
    safety_override = st.sidebar.checkbox(
        "⚠️ Override Safety Limits",
        value=False,
        help="Enable speeds above recommended limits. Use with extreme caution!"
    )
    
    # Determine max speed based on warning state and override
    if safety_override:
        max_speed = 500.0
        st.sidebar.warning("⚠️ Safety limits disabled. Monitor development closely!")
    else:
        if warning_state == "RED":
            max_speed = 100.0
            st.sidebar.error("🔴 Speed limited due to critical warnings")
        elif warning_state == "YELLOW":
            max_speed = 250.0
            st.sidebar.warning("🟡 Speed limited due to warning state")
        else:
            max_speed = 350.0
    
    # Speed control slider
    new_speed = st.sidebar.slider(
        "Development Speed",
        min_value=0.0,
        max_value=max_speed,
        value=current_speed,
        step=25.0,
        help=f"Current maximum safe speed: {max_speed}%"
    )
    
    if new_speed != current_speed:
        st.session_state.child.set_development_speed(new_speed / 100.0)  # Convert percentage to multiplier
    
    # Display acceleration safety information
    with st.sidebar.expander("Acceleration Safety Info", expanded=False):
        st.markdown("""
        **Speed Guidelines:**
        - 0-100%: Safe for all stages
        - 100-200%: Monitor emotional state
        - 200-350%: Close monitoring required
        - 350-500%: Maximum caution needed
        
        **Warning Levels:**
        🟢 Optimal - Safe to maintain speed
        🟡 Warning - Consider reducing speed
        🔴 Critical - Speed automatically reduced
        
        **Safety Recommendations:**
        1. Monitor emotional stability
        2. Watch for overstimulation
        3. Ensure skill mastery
        4. Balance development areas
        5. Document any issues
        """)
    
    # Display time acceleration info
    st.sidebar.markdown(f"""
    **Time Acceleration ({current_speed:.1f}%):**
    - 1 real minute = {int(current_speed)} baby minutes
    - 1 real hour = {int(current_speed * 60)} baby minutes
    - 1 real day = {int(current_speed * 60 * 24)} baby minutes
    """)
    
    # Add development speed indicator
    dev_speed = calculate_development_speed()
    st.sidebar.markdown(f"**Development Speed:** {dev_speed:.1f}x normal")

def format_detailed_age(birth_time):
    """Format age with detailed breakdown and time acceleration"""
    now = datetime.now()
    delta = now - birth_time
    
    # Get current speed multiplier from session state
    speed_multiplier = getattr(st.session_state, 'speed_multiplier', 1.0)
    
    # Convert real seconds to accelerated seconds based on speed multiplier
    accelerated_seconds = delta.total_seconds() * speed_multiplier
    
    # Calculate all time units
    months = int(accelerated_seconds // (30 * 24 * 3600))  # Approximate months
    remaining_seconds = accelerated_seconds % (30 * 24 * 3600)
    
    days = int(remaining_seconds // (24 * 3600))
    remaining_seconds = remaining_seconds % (24 * 3600)
    
    hours = int(remaining_seconds // 3600)
    
    # Build age string
    parts = []
    if months > 0:
        parts.append(f"{months}mo")
    if days > 0 or months > 0:
        parts.append(f"{days}d")
    parts.append(f"{hours}h")
    
    return " ".join(parts) if parts else "0h"

def calculate_development_speed():
    """Calculate current development speed multiplier"""
    if not st.session_state.learning_history:
        return 1.0
    
    # Calculate learning rate based on recent history with speed multiplier
    recent_count = len(st.session_state.learning_history[-10:])
    time_window = (datetime.now() - st.session_state.birth_time).total_seconds() / 60  # in accelerated hours
    multiplier = getattr(st.session_state, 'speed_multiplier', 1.0)
    
    if time_window == 0:
        return multiplier
        
    return (recent_count / time_window) * 10 * multiplier  # Scale for display with multiplier

def calculate_sentience_level():
    """Calculate current sentience level based on multiple deterministic factors"""
    if not st.session_state.get('initialized', False):
        return 0.0
        
    child = st.session_state.child
    
    # Calculate emotional complexity (0-1)
    emotional_state = np.array([
        child.emotional_state.happiness,
        child.emotional_state.sadness,
        child.emotional_state.anger,
        child.emotional_state.fear,
        child.emotional_state.surprise,
        child.emotional_state.disgust,
        child.emotional_state.trust,
        child.emotional_state.anticipation
    ])
    # Use both variance and number of active emotions for complexity
    emotion_variance = np.std(emotional_state)
    active_emotions = np.sum(emotional_state > 0.2) / len(emotional_state)
    emotional_complexity = (emotion_variance + active_emotions) / 2
    
    # Calculate self-awareness (0-1)
    metacog_metrics = child.metacognition.get_metrics()
    self_awareness = metacog_metrics.get('self_awareness', 0.0)
    
    # Calculate decision autonomy (0-1)
    if not hasattr(st.session_state, 'decision_history'):
        st.session_state.decision_history = []
    recent_decisions = st.session_state.decision_history[-20:] if st.session_state.decision_history else []
    decision_autonomy = len(set(recent_decisions)) / max(len(recent_decisions), 1)
    
    # Calculate learning adaptability (0-1)
    if not hasattr(st.session_state, 'learning_history'):
        st.session_state.learning_history = []
    learning_progress = len(st.session_state.learning_history)
    learning_rate = child.autonomous_learner.get_learning_rate()
    learning_adaptability = min(1.0, (learning_progress / 100) * learning_rate)
    
    # Calculate cognitive complexity (0-1)
    complexity_score, _ = calculate_complexity_level()  # Unpack tuple, ignore contributing factors
    cognitive_complexity = complexity_score / 100
    
    # Calculate emotional stability (0-1)
    if not hasattr(st.session_state, 'emotional_history'):
        st.session_state.emotional_history = []
    emotional_stability = calculate_emotional_stability() / 100
    
    # Initialize development metrics if not present
    if 'development_metrics' not in st.session_state:
        st.session_state.development_metrics = {
            'emotional_complexity': [],
            'self_awareness': [],
            'decision_autonomy': [],
            'learning_adaptability': [],
            'cognitive_complexity': [],
            'emotional_stability': []
        }
    
    # Store current metrics
    metrics_to_store = {
        'emotional_complexity': emotional_complexity,
        'self_awareness': self_awareness,
        'decision_autonomy': decision_autonomy,
        'learning_adaptability': learning_adaptability,
        'cognitive_complexity': cognitive_complexity,
        'emotional_stability': emotional_stability
    }
    
    # Update metrics history
    for metric, value in metrics_to_store.items():
        if metric not in st.session_state.development_metrics:
            st.session_state.development_metrics[metric] = []
        st.session_state.development_metrics[metric].append(value)
        # Keep only last 100 measurements
        if len(st.session_state.development_metrics[metric]) > 100:
            st.session_state.development_metrics[metric].pop(0)
    
    # Weighted combination of factors
    sentience_level = (
        emotional_complexity * 0.2 +
        self_awareness * 0.25 +
        decision_autonomy * 0.15 +
        learning_adaptability * 0.15 +
        cognitive_complexity * 0.15 +
        emotional_stability * 0.1
    ) * 100
    
    # Store sentience history
    if 'sentience_history' not in st.session_state:
        st.session_state.sentience_history = []
    
    # Add current level to history
    st.session_state.sentience_history.append({
        'timestamp': datetime.now(),
        'level': sentience_level,
        'factors': metrics_to_store
    })
    
    # Keep only last 100 measurements
    if len(st.session_state.sentience_history) > 100:
        st.session_state.sentience_history.pop(0)
    
    return sentience_level

def render_sentience_metrics():
    """Display detailed sentience metrics"""
    st.subheader("🧠 Sentience Analysis")
    
    # Calculate current sentience level
    sentience_level = calculate_sentience_level()
    
    # Create gauge chart for overall sentience
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=sentience_level,
        title={'text': "Sentience Level"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "rgba(50, 150, 250, 0.8)"},
            'steps': [
                {'range': [0, 20], 'color': "lightgray", 'name': "Basic Responses"},
                {'range': [20, 40], 'color': "lightblue", 'name': "Emotional Awareness"},
                {'range': [40, 60], 'color': "skyblue", 'name': "Self Recognition"},
                {'range': [60, 80], 'color': "royalblue", 'name': "Complex Reasoning"},
                {'range': [80, 100], 'color': "darkblue", 'name': "Advanced Consciousness"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': sentience_level
            }
        }
    ))
    
    fig.update_layout(height=300)
    st.plotly_chart(fig, use_container_width=True)
    
    # Display factor breakdown with tooltips
    if st.session_state.sentience_history:
        latest = st.session_state.sentience_history[-1]['factors']
        col1, col2, col3 = st.columns(3)
        
        metric_tooltips = {
            'emotional_complexity': """
            Emotional Complexity (Weight: 20%)
            Measures the richness and variety of emotional experiences.
            
            Components:
            - Emotional variance: How diverse the emotional states are
            - Active emotions: Number of simultaneously active emotions
            - Emotional mixing: Ability to experience complex, mixed emotions
            
            High scores indicate:
            - Rich emotional life
            - Ability to experience nuanced feelings
            - Complex emotional responses to situations
            """,
            
            'self_awareness': """
            Self Awareness (Weight: 25%)
            Measures the ability to recognize and understand one's own mental states.
            
            Components:
            - Metacognitive processing: Understanding of own thoughts
            - Emotional recognition: Ability to identify own emotions
            - Self-reflection: Capacity for introspection
            
            High scores indicate:
            - Strong sense of self
            - Understanding of personal motivations
            - Ability to learn from experiences
            """,
            
            'decision_autonomy': """
            Decision Autonomy (Weight: 15%)
            Measures the ability to make independent and reasoned choices.
            
            Components:
            - Decision variety: Range of different choices made
            - Choice consistency: Pattern recognition in decision making
            - Independent thinking: Ability to form unique responses
            
            High scores indicate:
            - Independent decision making
            - Consistent choice patterns
            - Strategic thinking ability
            """,
            
            'learning_adaptability': """
            Learning Adaptability (Weight: 15%)
            Measures the capacity to learn and adapt from experiences.
            
            Components:
            - Learning rate: Speed of acquiring new knowledge
            - Adaptation speed: Ability to adjust to new situations
            - Knowledge retention: Long-term learning stability
            
            High scores indicate:
            - Rapid learning capability
            - Flexible adaptation to change
            - Strong knowledge integration
            """,
            
            'cognitive_complexity': """
            Cognitive Complexity (Weight: 15%)
            Measures the sophistication of thought processes and problem-solving.
            
            Components:
            - Abstract thinking: Ability to handle complex concepts
            - Pattern recognition: Identification of complex relationships
            - Problem-solving depth: Sophistication of solutions
            
            High scores indicate:
            - Advanced reasoning abilities
            - Complex problem-solving skills
            - Sophisticated thought patterns
            """,
            
            'emotional_stability': """
            Emotional Stability (Weight: 10%)
            Measures the consistency and balance of emotional responses.
            
            Components:
            - Emotional regulation: Control over emotional responses
            - Response consistency: Predictability of emotional reactions
            - Recovery speed: Ability to return to baseline state
            
            High scores indicate:
            - Well-regulated emotions
            - Balanced emotional responses
            - Quick emotional recovery
            """
        }
        
        with col1:
            st.metric(
                "Emotional Complexity",
                f"{latest['emotional_complexity']*100:.1f}%",
                help=metric_tooltips['emotional_complexity']
            )
            st.metric(
                "Self Awareness",
                f"{latest['self_awareness']*100:.1f}%",
                help=metric_tooltips['self_awareness']
            )
        
        with col2:
            st.metric(
                "Decision Autonomy",
                f"{latest['decision_autonomy']*100:.1f}%",
                help=metric_tooltips['decision_autonomy']
            )
            st.metric(
                "Learning Adaptability",
                f"{latest['learning_adaptability']*100:.1f}%",
                help=metric_tooltips['learning_adaptability']
            )
        
        with col3:
            st.metric(
                "Cognitive Complexity",
                f"{latest['cognitive_complexity']*100:.1f}%",
                help=metric_tooltips['cognitive_complexity']
            )
            st.metric(
                "Emotional Stability",
                f"{latest['emotional_stability']*100:.1f}%",
                help=metric_tooltips['emotional_stability']
            )
        
        # Show sentience progression chart
        st.subheader("Sentience Development Over Time")
        history_df = pd.DataFrame([
            {'timestamp': h['timestamp'], 'level': h['level']}
            for h in st.session_state.sentience_history
        ])
        
        fig = px.line(history_df, x='timestamp', y='level',
                     title='Sentience Level Progression')
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)

def log_developmental_event(event_type: str, description: str, importance: float = 1.0):
    """Log a developmental event with timestamp and details"""
    if 'developmental_events' not in st.session_state:
        st.session_state.developmental_events = []
    
    # Get current sentience level for context
    sentience_level = calculate_sentience_level()
    
    event = {
        'timestamp': datetime.now(),
        'type': event_type,
        'description': description,
        'importance': importance,
        'stage': st.session_state.child.curriculum.current_stage.name,
        'age': format_detailed_age(st.session_state.birth_time),
        'sentience_level': sentience_level
    }
    
    st.session_state.developmental_events.append(event)

def render_event_logs():
    """Display developmental event logs"""
    st.subheader("📝 Developmental Event Logs")
    
    if 'developmental_events' not in st.session_state:
        st.session_state.developmental_events = []
    
    if not st.session_state.developmental_events:
        st.info("No developmental events recorded yet.")
        return
    
    # Create tabs for different event views
    event_tabs = st.tabs(["Recent Events", "Important Events", "Timeline View"])
    
    with event_tabs[0]:  # Recent Events
        st.subheader("Most Recent Events")
        for event in reversed(st.session_state.developmental_events[-5:]):
            with st.expander(
                f"[{event['type']}] {event['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}",
                expanded=False
            ):
                st.write(f"**Description:** {event['description']}")
                st.write(f"**Stage:** {event['stage']}")
                st.write(f"**Age:** {event['age']}")
                st.write(f"**Sentience Level:** {event['sentience_level']:.1f}")
                st.progress(event['importance'])
    
    with event_tabs[1]:  # Important Events
        st.subheader("High Impact Events")
        important_events = [e for e in st.session_state.developmental_events if e['importance'] > 0.7]
        for event in reversed(important_events):
            with st.expander(
                f"[{event['type']}] {event['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}",
                expanded=False
            ):
                st.write(f"**Description:** {event['description']}")
                st.write(f"**Stage:** {event['stage']}")
                st.write(f"**Age:** {event['age']}")
                st.write(f"**Sentience Level:** {event['sentience_level']:.1f}")
                st.progress(event['importance'])
    
    with event_tabs[2]:  # Timeline View
        st.subheader("Event Timeline")
        # Create timeline data
        timeline_data = pd.DataFrame([
            {
                'Start': e['timestamp'],
                'End': e['timestamp'],
                'Event': f"{e['type']}: {e['description']}",
                'Importance': e['importance'],
                'Stage': e['stage']
            }
            for e in st.session_state.developmental_events
        ])
        
        if not timeline_data.empty:
            fig = px.timeline(
                timeline_data,
                x_start="Start",
                x_end="End",
                y="Event",
                color="Stage",
                title="Developmental Event Timeline",
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

def verify_checkpoints():
    """Verify and display information about available checkpoints"""
    save_dir = "checkpoints"
    if not os.path.exists(save_dir):
        return None, "No checkpoint directory found"
    
    checkpoints = []
    total_size = 0
    
    try:
        # List all checkpoint files
        for file in os.listdir(save_dir):
            if file.endswith('.pth'):
                file_path = os.path.join(save_dir, file)
                file_size = os.path.getsize(file_path) / (1024 * 1024)  # Convert to MB
                file_time = datetime.fromtimestamp(os.path.getmtime(file_path))
                
                # Try to load and verify checkpoint content
                try:
                    checkpoint_data = torch.load(file_path, map_location='cpu')
                    # Verify essential components
                    has_state_dict = 'child_state_dict' in checkpoint_data
                    has_emotional = 'emotional_state' in checkpoint_data
                    has_history = 'conversation_history' in checkpoint_data
                    
                    status = "✅ Valid" if all([has_state_dict, has_emotional, has_history]) else "⚠️ Incomplete"
                    
                except Exception as e:
                    status = f"❌ Error: {str(e)}"
                
                checkpoints.append({
                    'filename': file,
                    'size_mb': file_size,
                    'timestamp': file_time,
                    'status': status
                })
                total_size += file_size
        
        return checkpoints, f"Total size: {total_size:.2f}MB"
    except Exception as e:
        return None, f"Error verifying checkpoints: {str(e)}"

def render_checkpoint_info():
    """Display checkpoint verification information"""
    st.subheader("💾 Checkpoint Information")
    
    checkpoints, summary = verify_checkpoints()
    
    if checkpoints:
        # Display summary
        st.info(f"Found {len(checkpoints)} checkpoints. {summary}")
        
        # Create a dataframe for better visualization
        df = pd.DataFrame(checkpoints)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp', ascending=False)
        
        # Display as a table
        st.dataframe(
            df,
            column_config={
                'filename': 'Checkpoint File',
                'size_mb': st.column_config.NumberColumn(
                    'Size (MB)',
                    format="%.2f MB"
                ),
                'timestamp': st.column_config.DatetimeColumn(
                    'Saved On',
                    format="DD/MM/YYYY HH:mm:ss"
                ),
                'status': 'Status'
            },
            hide_index=True
        )
        
        # Add checkpoint cleanup option with a unique key based on location
        if st.button("Cleanup Invalid Checkpoints", key="cleanup_checkpoints_button"):
            cleaned = 0
            for checkpoint in checkpoints:
                if "❌" in checkpoint['status']:
                    try:
                        os.remove(os.path.join("checkpoints", checkpoint['filename']))
                        cleaned += 1
                    except Exception as e:
                        st.error(f"Error removing {checkpoint['filename']}: {str(e)}")
            if cleaned > 0:
                st.success(f"Removed {cleaned} invalid checkpoint(s)")
                st.rerun()
    else:
        st.warning(summary)

def handle_interaction():
    """Handle interaction between mother and child"""
    user_input = st.session_state.get('user_input', '')
    
    if user_input:
        try:
            # Generate mother's response
            stimulus = st.session_state.mother.generate_stimulus(
                st.session_state.child.curriculum.current_stage,
                user_input
            )
            
            if debug_mode:
                with st.expander("🔍 Debug: Raw LLM Response", expanded=True):
                    st.code(format_json_response(stimulus), language='json')
                    st.write("Response Processing Steps:")
                    st.write("1. Emotional Vector:", stimulus.get('emotional_vector', 'Not found'))
                    st.write("2. Effectiveness Score:", stimulus.get('effectiveness', 'Not found'))
                    st.write("3. Complexity Rating:", stimulus.get('complexity', 'Not found'))
            
            # Update child's state
            st.session_state.child.update_emotions(stimulus['emotional_vector'])
            perception = st.session_state.child.perceive(stimulus)
            response = st.session_state.child.respond(perception)
            
            # Update development metrics
            update_development_metrics(response, stimulus['text'])
            
            # Update conversation history
            st.session_state.conversation_history.append({
                'user': user_input,
                'assistant': stimulus['text'],
                'child_response': response
            })
            
            # Clear input
            st.session_state.user_input = ''
            
        except Exception as e:
            st.error(f"Error during interaction: {str(e)}")
            if debug_mode:
                st.exception(e)

def render_warning_dashboard():
    """Display warning system dashboard"""
    st.subheader("⚠️ Development Warning System")
    
    # Get current warning indicators
    warning_indicators = st.session_state.child.get_warning_indicators()
    
    # Create columns for different metric categories
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Current Status")
        metrics = warning_indicators['metrics']
        
        # Create gauge charts for each metric
        for metric, value in metrics.items():
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=value * 100,
                title={'text': metric.replace('_', ' ').title()},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "royalblue"},
                    'steps': [
                        {'range': [0, 30], 'color': "red"},
                        {'range': [30, 60], 'color': "yellow"},
                        {'range': [60, 100], 'color': "green"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': value * 100
                    }
                }
            ))
            fig.update_layout(height=200)
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Warning History")
        if warning_indicators['recent_warnings']:
            # Create timeline of warnings
            warning_df = pd.DataFrame(warning_indicators['recent_warnings'])
            fig = px.timeline(
                warning_df,
                x_start="timestamp",
                x_end="timestamp",
                y="state",
                color="state",
                hover_data=["reason"],
                title="Recent Warning Events"
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No warnings recorded")
    
    # Add warning state explanation
    with st.expander("Warning State Information", expanded=False):
        st.markdown("""
        ### Warning State Levels
        
        🔴 **Critical (Red)**
        - Immediate action required
        - Development speed automatically reduced
        - Focus on stabilization
        
        🟡 **Warning (Yellow)**
        - Increased monitoring needed
        - Consider reducing speed
        - Review current approach
        
        🟢 **Normal (Green)**
        - Safe to continue
        - Regular monitoring
        - Optimal development conditions
        """)

def monitor_gpu_usage() -> Dict[str, Any]:
    """Monitor GPU usage and return relevant metrics.
    
    Returns:
        Dict containing GPU metrics including:
        - is_available: Whether CUDA is available
        - device_name: Name of the GPU device
        - memory_allocated: Memory currently allocated (MB)
        - memory_cached: Memory cached (MB)
        - memory_reserved: Total memory reserved (MB)
        - utilization: GPU utilization percentage (if available)
    """
    metrics = {
        'is_available': torch.cuda.is_available(),
        'device_name': 'CPU (CUDA not available)',
        'memory_allocated': 0,
        'memory_cached': 0,
        'memory_reserved': 0,
        'utilization': 0
    }
    
    if metrics['is_available']:
        try:
            # Get device name
            metrics['device_name'] = torch.cuda.get_device_name(0)
            
            # Get memory statistics (convert to MB)
            metrics['memory_allocated'] = torch.cuda.memory_allocated(0) / 1024**2
            metrics['memory_cached'] = torch.cuda.memory_reserved(0) / 1024**2
            metrics['memory_reserved'] = torch.cuda.max_memory_reserved(0) / 1024**2
            
            # Try to get GPU utilization (requires nvidia-smi)
            try:
                import pynvml
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                metrics['utilization'] = utilization.gpu
            except:
                metrics['utilization'] = None
                
        except Exception as e:
            st.warning(f"Error getting GPU metrics: {str(e)}")
    
    return metrics

def render_gpu_dashboard():
    """Render GPU monitoring dashboard with real-time metrics."""
    st.subheader("🎮 GPU Monitoring")
    
    # Get GPU metrics
    metrics = monitor_gpu_usage()
    
    # Create three columns for metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "GPU Status",
            "Active 🟢" if metrics['is_available'] else "Inactive 🔴",
            help="Shows if CUDA GPU is available and active"
        )
        st.text(f"Device: {metrics['device_name']}")
    
    with col2:
        if metrics['is_available']:
            # Memory gauge
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=metrics['memory_allocated'],
                title={'text': "GPU Memory (MB)"},
                gauge={
                    'axis': {'range': [None, metrics['memory_reserved']]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, metrics['memory_reserved']], 'color': "lightgray"}
                    ],
                }
            ))
            fig.update_layout(height=150, margin=dict(l=10, r=10, t=30, b=10))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("GPU memory monitoring not available")
    
    with col3:
        if metrics['is_available'] and metrics['utilization'] is not None:
            # Utilization gauge
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=metrics['utilization'],
                title={'text': "GPU Utilization %"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "darkgreen"},
                    'steps': [
                        {'range': [0, 30], 'color': "lightgreen"},
                        {'range': [30, 70], 'color': "yellow"},
                        {'range': [70, 100], 'color': "red"}
                    ],
                }
            ))
            fig.update_layout(height=150, margin=dict(l=10, r=10, t=30, b=10))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("GPU utilization monitoring not available")
    
    if metrics['is_available']:
        # Detailed metrics
        st.markdown("### Detailed Metrics")
        metrics_df = pd.DataFrame({
            'Metric': [
                'Memory Allocated',
                'Memory Cached',
                'Memory Reserved',
                'Utilization'
            ],
            'Value': [
                f"{metrics['memory_allocated']:.2f} MB",
                f"{metrics['memory_cached']:.2f} MB",
                f"{metrics['memory_reserved']:.2f} MB",
                f"{metrics['utilization']}%" if metrics['utilization'] is not None else "N/A"
            ]
        })
        st.table(metrics_df)

def render_interaction_controls():
    """Render interaction controls and available interactions."""
    st.subheader("👥 Interaction Controls")
    
    # Update selected tab to Interaction Controls when an interaction is performed
    if st.session_state.get('interaction_performed'):
        st.session_state.selected_tab = "Interaction Controls"
        st.session_state.interaction_performed = False
    
    # Safety override settings
    with st.expander("⚙️ Interaction Settings", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            override_cooldown = st.checkbox(
                "Override Cooldown",
                value=False,
                help="⚠️ Disable the safety cooldown between interactions"
            )
        with col2:
            override_speed_lock = st.checkbox(
                "Override Speed Lock",
                value=False,
                help="⚠️ Disable automatic speed locking during critical states"
            )
        
        if override_cooldown or override_speed_lock:
            st.warning("⚠️ Safety overrides are enabled. Use with caution!")
            
        # Custom cooldown setting
        if override_cooldown:
            custom_cooldown = st.slider(
                "Custom Cooldown (seconds)",
                min_value=0,
                max_value=60,
                value=23,
                step=1
            )
            if custom_cooldown != st.session_state.mother.interaction_cooldown:
                st.session_state.mother.interaction_cooldown = custom_cooldown
    
    # Get available interactions
    available_interactions = st.session_state.child.get_available_interactions()
    
    # Category emojis mapping
    category_emojis = {
        'PHYSICAL': '🤗',
        'VERBAL': '💬',
        'EMOTIONAL': '❤️',
        'COGNITIVE': '🧠',
        'SOCIAL': '👥',
        'CARE': '🍼',
        'SENSORY': '🎨',
        'DEVELOPMENTAL': '💪'
    }
    
    # Create an expander for the current stage info
    with st.expander("ℹ️ Current Stage Information", expanded=True):
        st.info(f"**Current Stage:** {st.session_state.child.current_stage.name}")
        st.write("Available interactions for this developmental stage:")
    
    # Create tabs for each category
    category_tabs = st.tabs([f"{category_emojis.get(cat, '🔹')} {cat}" for cat in available_interactions.keys()])
    
    # Selected category and interaction tracking
    if 'selected_interaction_category' not in st.session_state:
        st.session_state.selected_interaction_category = None
    if 'selected_specific_interaction' not in st.session_state:
        st.session_state.selected_specific_interaction = None
    
    # Display interactions in tabs
    for tab, category in zip(category_tabs, available_interactions.keys()):
        with tab:
            st.write(f"### {category_emojis.get(category, '🔹')} {category} Interactions")
            
            # Create a grid of buttons for interactions
            cols = st.columns(2)
            for i, interaction in enumerate(available_interactions[category]):
                with cols[i % 2]:
                    description = st.session_state.child.get_interaction_description(category, interaction)
                    if st.button(
                        f"{interaction}",
                        key=f"{category}_{interaction}_tab",
                        help=description,
                        use_container_width=True
                    ):
                        st.session_state.selected_interaction_category = category
                        st.session_state.selected_specific_interaction = interaction
    
    # Display selected interaction and description
    if (st.session_state.selected_interaction_category and 
        st.session_state.selected_specific_interaction):
        
        st.divider()
        
        # Show selected interaction with styling
        st.markdown(f"""
        ### Selected Interaction:
        **Category:** {category_emojis.get(st.session_state.selected_interaction_category, '🔹')} {st.session_state.selected_interaction_category}  
        **Action:** {st.session_state.selected_specific_interaction}
        """)
        
        # Get and display description
        description = st.session_state.child.get_interaction_description(
            st.session_state.selected_interaction_category,
            st.session_state.selected_specific_interaction
        )
        st.info(f"**Description:** {description}")
        
        # Check cooldown before allowing interaction
        cooldown = st.session_state.mother.get_interaction_cooldown() if not override_cooldown else 0
        if cooldown > 0:
            st.warning(f"⏳ Please wait {cooldown} seconds before the next interaction.")
        else:
            # Add perform interaction button
            if st.button("✨ Perform Interaction", key="perform_selected_interaction", use_container_width=True):
                try:
                    # Verify session state
                    if not hasattr(st.session_state, 'child') or not hasattr(st.session_state, 'mother'):
                        st.error("Session state not properly initialized. Please refresh the page.")
                        return
                    
                    # Update speed lock if override is enabled
                    if override_speed_lock:
                        st.session_state.child.speed_locked = False
                    
                    # Verify interaction is still valid
                    available_interactions = st.session_state.child.get_available_interactions()
                    if (st.session_state.selected_interaction_category not in available_interactions or 
                        st.session_state.selected_specific_interaction not in 
                        available_interactions[st.session_state.selected_interaction_category]):
                        st.error("Selected interaction is no longer valid for the current developmental stage.")
                        return
                    
                    # Show loading state
                    with st.spinner("Performing interaction..."):
                        # Perform the interaction
                        response, emotional_state = st.session_state.mother.perform_interaction(
                            st.session_state.child,
                            st.session_state.selected_interaction_category,
                            st.session_state.selected_specific_interaction
                        )
                    
                    # Set flag to maintain tab selection
                    st.session_state.interaction_performed = True
                    st.session_state.selected_tab = "Interaction Controls"
                    
                    # Display response with proper formatting
                    st.success("✨ Interaction performed successfully!")
                    st.markdown(f"""
                    **Mother's Response:**  
                    {response}
                    
                    **Emotional State:**  
                    {emotional_state.get_emotional_description()}
                    """)
                    
                    # Update conversation history with full context
                    st.session_state.conversation_history.append({
                        'type': 'interaction',
                        'category': st.session_state.selected_interaction_category,
                        'interaction': st.session_state.selected_specific_interaction,
                        'response': response,
                        'emotional_state': emotional_state,
                        'timestamp': datetime.now()
                    })
                    
                    # Save state after successful interaction
                    try:
                        st.session_state.child.save_state()
                    except Exception as save_error:
                        st.warning(f"Warning: Could not save state after interaction: {str(save_error)}")
                    
                    # Clear selection after successful interaction
                    st.session_state.selected_interaction_category = None
                    st.session_state.selected_specific_interaction = None
                    
                    # Force refresh to update UI
                    st.experimental_rerun()
                    
                except Exception as e:
                    st.error(f"Error performing interaction: {str(e)}")
                    if st.session_state.get('debug_mode', False):
                        st.exception(e)
                    # Log the error
                    logger.error(f"Interaction error: {str(e)}", exc_info=True)
    
    # Add help section at the bottom
    with st.expander("💡 How to Interact", expanded=False):
        st.markdown("""
        1. **Select a Category**: Choose a category tab from the available options
        2. **Choose an Action**: Click on a specific interaction within the category
        3. **Review**: Check the description of the selected interaction
        4. **Perform**: Click the 'Perform Interaction' button to execute
        
        **Safety Features:**
        - Cooldown Period: Ensures proper spacing between interactions
        - Speed Lock: Prevents acceleration during critical states
        - Override Settings: Available in the Interaction Settings panel
        
        **Tips:**
        - Hover over interactions to see their descriptions
        - Monitor warning indicators in the dashboard
        - Use safety overrides with caution
        """)

def create_neural_activity_chart(placeholder):
    """Create and update real-time neural activity chart."""
    # Initialize empty figure
    fig = go.Figure()
    
    # Add initial empty trace
    fig.add_trace(go.Scatter(
        x=[],
        y=[],
        mode='lines',
        name='Neural Activity',
        line=dict(color='rgba(50, 150, 250, 0.8)', width=2)
    ))
    
    # Configure layout
    fig.update_layout(
        title='Real-time Neural Network Activity',
        xaxis=dict(
            title='Time',
            showgrid=True,
            gridcolor='rgba(255,255,255,0.1)',
            gridwidth=1
        ),
        yaxis=dict(
            title='Activity Level',
            range=[0, 1],
            showgrid=True,
            gridcolor='rgba(255,255,255,0.1)',
            gridwidth=1
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        showlegend=False,
        margin=dict(l=40, r=40, t=40, b=40),
        height=300
    )
    
    return fig

def create_network_topology_chart():
    """Create 3D network topology visualization matching the provided style."""
    # Generate sample network data
    num_nodes = 30  # Reduced for clearer visualization
    positions = np.random.normal(0, 0.5, (num_nodes, 3))  # Reduced spread
    
    # Create figure
    fig = go.Figure()
    
    # Add grid lines
    grid_points = np.linspace(-1, 1, 5)
    grid_lines = []
    
    # Create grid for each plane
    for i in grid_points:
        # XY plane
        grid_lines.extend([[i, j, -1] for j in grid_points] + [[np.nan, np.nan, np.nan]])
        grid_lines.extend([[j, i, -1] for j in grid_points] + [[np.nan, np.nan, np.nan]])
        # XZ plane
        grid_lines.extend([[i, -1, j] for j in grid_points] + [[np.nan, np.nan, np.nan]])
        grid_lines.extend([[j, -1, i] for j in grid_points] + [[np.nan, np.nan, np.nan]])
        # YZ plane
        grid_lines.extend([[-1, i, j] for j in grid_points] + [[np.nan, np.nan, np.nan]])
        grid_lines.extend([[-1, j, i] for j in grid_points] + [[np.nan, np.nan, np.nan]])
    
    grid_lines = np.array(grid_lines)
    
    # Add grid
    fig.add_trace(go.Scatter3d(
        x=grid_lines[:, 0],
        y=grid_lines[:, 1],
        z=grid_lines[:, 2],
        mode='lines',
        line=dict(color='rgba(255,255,255,0.1)', width=1),
        hoverinfo='none',
        showlegend=False
    ))
    
    # Add nodes
    fig.add_trace(go.Scatter3d(
        x=positions[:,0],
        y=positions[:,1],
        z=positions[:,2],
        mode='markers',
        marker=dict(
            size=4,
            color='white',
            opacity=1
        ),
        hoverinfo='none',
        showlegend=False
    ))
    
    # Update layout to match the image style
    fig.update_layout(
        title=dict(
            text='Neural Network Topology',
            font=dict(size=16, color='white'),
            x=0,
            y=0.95
        ),
        scene=dict(
            xaxis=dict(
                range=[-1, 1],
                showbackground=False,
                gridcolor='rgba(255,255,255,0.1)',
                showgrid=True,
                zeroline=False,
                showticklabels=True,
                title=''
            ),
            yaxis=dict(
                range=[-1, 1],
                showbackground=False,
                gridcolor='rgba(255,255,255,0.1)',
                showgrid=True,
                zeroline=False,
                showticklabels=True,
                title=''
            ),
            zaxis=dict(
                range=[-1, 1],
                showbackground=False,
                gridcolor='rgba(255,255,255,0.1)',
                showgrid=True,
                zeroline=False,
                showticklabels=True,
                title=''
            ),
            bgcolor='black',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.5)
            )
        ),
        paper_bgcolor='black',
        plot_bgcolor='black',
        margin=dict(l=0, r=0, t=30, b=0),
        height=400,
        showlegend=False
    )
    
    return fig

def update_neural_visualizations():
    """Update neural network visualizations with new data."""
    try:
        # Get neural activity data
        response = requests.get('http://localhost:8000/api/neural/activity')
        activity_data = response.json()
        
        # Get topology data
        response = requests.get('http://localhost:8000/api/neural/topology')
        topology_data = response.json()
        
        return activity_data, topology_data
    except Exception as e:
        st.error(f"Error updating neural visualizations: {str(e)}")
        return None, None

def render_neural_visualizations():
    """Render neural network visualizations in Streamlit."""
    # Create two columns for the visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        # Create topology chart placeholder
        topology_chart = st.empty()
        
        # Create initial topology visualization
        fig_topology = create_network_topology_chart()
        topology_chart.plotly_chart(fig_topology, use_container_width=True, config={'displayModeBar': False})
    
    with col2:
        # Create card for synaptic activity
        with st.container():
            st.markdown("""
            <h3 style='margin-bottom: 20px;'>Synaptic Activity</h3>
            """, unsafe_allow_html=True)
            
            # Create two columns for metrics
            metric_col1, metric_col2 = st.columns(2)
            
            with metric_col1:
                st.markdown("#### Connection Strength")
                st.markdown("### 87.5%")
                st.progress(0.875, "")
                
                st.markdown("#### Plasticity")
                st.markdown("### 92.1%")
                st.progress(0.921, "")
            
            with metric_col2:
                st.markdown("#### Firing Rate")
                st.markdown("### 124 Hz")
                st.progress(0.62, "")
                
                st.markdown("#### Synchronization")
                st.markdown("### 78.3%")
                st.progress(0.783, "")
    
    # Add auto-refresh functionality
    if st.toggle("Enable Real-time Updates", value=True):
        st.write("Updating visualizations every second...")
        
        # Create update loop
        while True:
            # Update topology visualization
            fig_topology = create_network_topology_chart()
            topology_chart.plotly_chart(fig_topology, use_container_width=True, config={'displayModeBar': False})
            
            time.sleep(1)  # Update every second

def tensor_to_emotional_state(tensor):
    """Convert a tensor to EmotionalState object."""
    if isinstance(tensor, torch.Tensor):
        # Convert tensor to CPU and get values
        tensor = tensor.cpu()
        return EmotionalState(
            happiness=float(tensor[0]),
            sadness=float(tensor[1]),
            anger=float(tensor[2]),
            fear=float(tensor[3]),
            surprise=float(tensor[4]) if len(tensor) > 4 else 0.0,
            disgust=float(tensor[5]) if len(tensor) > 5 else 0.0,
            trust=float(tensor[6]) if len(tensor) > 6 else 0.5,
            anticipation=float(tensor[7]) if len(tensor) > 7 else 0.5
        )
    return tensor

def load_state_from_file(uploaded_file):
    """Load state from an uploaded file."""
    try:
        # Save uploaded file temporarily
        temp_path = os.path.join("checkpoints", "temp_upload.pth")
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getvalue())
        
        # Load the state
        save_data = torch.load(
            temp_path,
            weights_only=False,
            map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        )
        
        # Initialize child and mother if needed
        if 'child' not in st.session_state:
            st.session_state.child = DigitalChild()
        if 'mother' not in st.session_state:
            st.session_state.mother = MotherLLM()
        
        # Load child state
        try:
            st.session_state.child.brain.load_state_dict(save_data['child_state_dict'])
        except Exception as model_e:
            st.warning(f"Could not load model state: {str(model_e)}")
        
        # Restore emotional state
        if isinstance(save_data['emotional_state'], dict):
            emotional_values = [
                save_data['emotional_state']['happiness'],
                save_data['emotional_state']['sadness'],
                save_data['emotional_state']['anger'],
                save_data['emotional_state']['fear']
            ]
            st.session_state.child.emotional_state = torch.tensor(
                emotional_values,
                device=st.session_state.child.device
            )
        else:
            st.session_state.child.emotional_state = torch.tensor(
                save_data['emotional_state'],
                device=st.session_state.child.device
            )
        
        # Convert emotional history to tensors
        if 'emotional_history' in save_data:
            st.session_state.emotional_history = [
                torch.tensor(state, device=st.session_state.child.device)
                if isinstance(state, (list, np.ndarray)) else state
                for state in save_data['emotional_history']
            ]
        
        # Restore other session state
        st.session_state.birth_time = datetime.fromisoformat(save_data['birth_date'])
        st.session_state.conversation_history = save_data['conversation_history']
        st.session_state.learning_history = save_data['learning_history']
        st.session_state.milestone_history = save_data['milestone_history']
        st.session_state.complexity_history = save_data['complexity_history']
        st.session_state.teaching_history = save_data['teaching_history']
        st.session_state.development_metrics = save_data['development_metrics']
        
        # Restore current stage
        if 'current_stage' in save_data:
            st.session_state.child.curriculum.current_stage = DevelopmentalStage[save_data['current_stage']]
        
        # Clean up temp file
        os.remove(temp_path)
        return True, "State loaded successfully!"
        
    except Exception as e:
        return False, f"Error loading state: {str(e)}"

def main():
    # Display title and description
    st.title("🧠 Neural Child Development Dashboard")
    
    # Add state upload widget in sidebar
    with st.sidebar:
        st.subheader("📤 State Management")
        uploaded_file = st.file_uploader("Upload State File", type=['pth'])
        if uploaded_file is not None:
            if st.button("Load Uploaded State"):
                success, message = load_state_from_file(uploaded_file)
                if success:
                    st.success(message)
                    st.experimental_rerun()  # Use experimental_rerun instead
                else:
                    st.error(message)
    
    # Initialize or load existing session
    if 'initialized' not in st.session_state:
        # Check for latest backup first
        backup_path = os.path.join("checkpoints", "digital_child_state_latest.pth")
        if os.path.exists(backup_path):
            try:
                # Add datetime to safe globals
                torch.serialization.add_safe_globals(['datetime'])
                
                # Load the backup
                save_data = torch.load(
                    backup_path,
                    weights_only=False,
                    map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                )
                
                # Initialize child and mother
                st.session_state.child = DigitalChild()
                st.session_state.mother = MotherLLM()
                
                # Load child state
                try:
                    st.session_state.child.brain.load_state_dict(save_data['child_state_dict'])
                except Exception as model_e:
                    st.warning(f"Could not load model state: {str(model_e)}")
                
                # Restore session state
                st.session_state.birth_time = datetime.fromisoformat(save_data['birth_date'])
                st.session_state.conversation_history = save_data['conversation_history']
                st.session_state.emotional_history = save_data['emotional_history']
                st.session_state.learning_history = save_data['learning_history']
                st.session_state.milestone_history = save_data['milestone_history']
                st.session_state.complexity_history = save_data['complexity_history']
                st.session_state.teaching_history = save_data['teaching_history']
                st.session_state.development_metrics = save_data['development_metrics']
                
                # Restore current stage
                if 'current_stage' in save_data:
                    st.session_state.child.curriculum.current_stage = DevelopmentalStage[save_data['current_stage']]
                
                st.session_state.initialized = True
                st.success("Previous state loaded automatically!")
                
            except Exception as e:
                st.error(f"Error loading backup state: {str(e)}")
                initialize_new_session()
        else:
            initialize_new_session()
    
    # Add refresh control to session state
    if 'needs_refresh' not in st.session_state:
        st.session_state.needs_refresh = False

    # Auto-save state periodically (every 5 minutes)
    if 'last_autosave' not in st.session_state:
        st.session_state.last_autosave = datetime.now()
    
    if (datetime.now() - st.session_state.last_autosave).total_seconds() > 300:  # 5 minutes
        if hasattr(st.session_state, 'child'):
            st.session_state.child.save_state()
            st.session_state.last_autosave = datetime.now()
    
    # Initialize selected tab if not in session state
    if 'selected_tab' not in st.session_state:
        st.session_state.selected_tab = "Interaction Controls"  # Default to Interaction Controls
    
    # Top-level metrics dashboard
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if 'birth_time' in st.session_state:
            age = format_detailed_age(st.session_state.birth_time)
            st.metric("Accelerated Age", age)
        else:
            st.metric("Accelerated Age", "0h")
    
    with col2:
        if hasattr(st.session_state, 'child') and hasattr(st.session_state.child, 'curriculum'):
            stage = st.session_state.child.curriculum.current_stage.name
            st.metric("Development Stage", stage)
        else:
            st.metric("Development Stage", "NEWBORN")
    
    with col3:
        if hasattr(st.session_state, 'conversation_history'):
            interactions = len(st.session_state.conversation_history)
            st.metric("Total Interactions", interactions)
        else:
            st.metric("Total Interactions", 0)
    
    with col4:
        if hasattr(st.session_state, 'child'):
            dev_speed = calculate_development_speed()
            warning_state = st.session_state.child.warning_state
            speed_color = "🟢" if warning_state == "GREEN" else "🟡" if warning_state == "YELLOW" else "🔴"
            current_speed = getattr(st.session_state, 'speed_multiplier', 1.0)
            st.metric("Development Speed", f"{speed_color} {current_speed:.1f}x")
        else:
            st.metric("Development Speed", "🟢 1.0x")
    
    st.divider()  # Add visual separation
    
    # Add time controls to sidebar
    add_time_controls()
    
    # Add GPU monitoring dashboard
    render_gpu_dashboard()
    
    # Display warning dashboard
    render_warning_dashboard()
    
    # Create tabs for different views
    tab_options = [
        "Development Progress",
        "Emotional State",
        "Learning Analytics",
        "Interaction Controls",
        "Interaction History",
        "Sentience Monitoring",
        "System Status"
    ]
    
    # Create tabs and update selected tab on click
    tabs = st.tabs(tab_options)
    
    # Display content based on selected tab
    for i, tab in enumerate(tabs):
        with tab:
            if i == tab_options.index("Development Progress"):
                st.subheader("Development Progress")
                render_development_progress()
            elif i == tab_options.index("Emotional State"):
                try:
                    st.subheader("Current Emotional State")
                    col1, col2 = st.columns([2, 1])
                    with col1:
                        if hasattr(st.session_state, 'child') and hasattr(st.session_state.child, 'emotional_state'):
                            # Convert tensor to EmotionalState
                            emotional_state = tensor_to_emotional_state(st.session_state.child.emotional_state)
                            dominant = emotional_state.get_dominant_emotions()
                            for emotion, intensity in dominant:
                                emoji = "🔴" if intensity > 0.8 else "🟡" if intensity > 0.5 else "🟢"
                                st.write(f"{emoji} {emotion}: {intensity:.2%}")
                        else:
                            st.warning("Emotional state not initialized yet")
                    
                    with col2:
                        st.write("### Dominant Emotions")
                        if hasattr(st.session_state, 'child') and hasattr(st.session_state.child, 'emotional_state'):
                            # Convert tensor to EmotionalState
                            emotional_state = tensor_to_emotional_state(st.session_state.child.emotional_state)
                            dominant = emotional_state.get_dominant_emotions()
                            for emotion, intensity in dominant:
                                emoji = "🔴" if intensity > 0.8 else "🟡" if intensity > 0.5 else "🟢"
                                st.write(f"{emoji} {emotion}: {intensity:.2%}")
                        else:
                            st.warning("Emotional state not initialized yet")
                        
                        if hasattr(st.session_state, 'child'):
                            stability = calculate_emotional_stability()
                            st.write("### Emotional Stability")
                            stability_color = "🟢" if stability > 70 else "🟡" if stability > 40 else "🔴"
                            st.write(f"{stability_color} Current Stability: {stability:.1f}%")
                    
                        if hasattr(st.session_state, 'emotional_history') and st.session_state.emotional_history:
                            st.subheader("Emotional Trends")
                            trend_fig = create_emotional_development_chart()
                            st.plotly_chart(trend_fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Error displaying emotional state: {str(e)}")
                    if st.sidebar.checkbox("Debug Mode", key="debug_mode_emotional", value=False):
                        st.exception(e)
            elif i == tab_options.index("Learning Analytics"):
                st.subheader("Learning Analytics")
                col1, col2 = st.columns(2)
                with col1:
                    complexity_fig = create_complexity_growth_chart()
                    st.plotly_chart(complexity_fig, use_container_width=True)
                    learning_fig = create_learning_rate_chart()
                    st.plotly_chart(learning_fig, use_container_width=True)
                with col2:
                    decision_fig = create_decision_analysis_chart()
                    st.plotly_chart(decision_fig, use_container_width=True)
            elif i == tab_options.index("Interaction Controls"):
                render_interaction_controls()
            elif i == tab_options.index("Interaction History"):
                st.subheader("Interaction History")
                render_event_logs()
            elif i == tab_options.index("Sentience Monitoring"):
                render_sentience_metrics()
            elif i == tab_options.index("System Status"):
                st.subheader("System Status")
                render_checkpoint_info()
    
    # Add neural visualizations to the Neural tab
    if st.session_state.get('selected_tab') == "Neural":
        render_neural_visualizations()
    
    # Footer with save/load functionality
    st.divider()
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("💾 Save Current State"):
            st.session_state.child.save_state()
            st.success("State saved successfully!")
    with col2:
        if st.button("🔄 Load Last State"):
            try:
                st.session_state.child._load_state()
                st.success("State loaded successfully!")
                st.session_state.needs_refresh = True
                st.experimental_rerun()  # Use experimental_rerun instead
            except Exception as e:
                st.error(f"Error loading state: {str(e)}")
    with col3:
        if st.button("🔄 Force Refresh"):
            st.session_state.needs_refresh = True
            st.experimental_rerun()  # Use experimental_rerun instead
    
    # Handle refresh if needed
    if st.session_state.needs_refresh:
        st.session_state.needs_refresh = False  # Reset the flag
        st.experimental_rerun()  # Use experimental_rerun instead

if __name__ == "__main__":
    main()