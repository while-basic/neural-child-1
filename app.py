import streamlit as st
import torch
import time
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from developmental_stages import DevelopmentalStage, DevelopmentalSystem
import json
import torch.serialization
import requests
import sseclient
import threading
import queue
import os

try:
    from main import DigitalChild, MotherLLM
except ImportError:
    st.error("Failed to import DigitalChild and MotherLLM. Make sure main.py exists and is properly configured.")
    DigitalChild = None
    MotherLLM = None

def validate_stage_definitions():
    """Validate that all developmental stages have proper definitions"""
    missing_stages = []
    for stage in DevelopmentalStage:
        try:
            # Create a temporary system to test stage definitions
            system = DevelopmentalSystem()
            system.current_stage = stage
            # Try to get requirements for each stage
            system.get_stage_requirements()
        except Exception as e:
            missing_stages.append(stage.name)
    return missing_stages

# Initialize session state
if DigitalChild is not None:
    if 'initialized' not in st.session_state:
        try:
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
        except Exception as e:
            st.error(f"Error initializing digital child: {str(e)}")
            if st.sidebar.checkbox("Debug Mode", value=False):
                st.exception(e)

def ensure_tensor_device(tensor, target_device=None):
    """Ensure tensor is on the correct device"""
    if target_device is None:
        target_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if isinstance(tensor, torch.Tensor):
        return tensor.to(target_device)
    return tensor

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
    try:
        stage = st.session_state.child.curriculum.current_stage
        
        # Safety check for stage
        if not isinstance(stage, DevelopmentalStage):
            st.error("Invalid stage type")
            return
        
        # Get stage requirements using the new API
        stage_reqs = st.session_state.child.curriculum.get_stage_requirements()
        
        # Display current milestones
        st.write("Current Milestones:")
        current_milestones = stage_reqs.get('current_milestones', [])
        if current_milestones:
            for milestone in current_milestones:
                st.write(f"✓ {milestone}")
        else:
            st.write("No current milestones defined")
        
        # Display upcoming milestones
        st.write("\nUpcoming Milestones:")
        upcoming_milestones = stage_reqs.get('upcoming_milestones', [])
        if upcoming_milestones:
            for milestone in upcoming_milestones:
                st.write(f"○ {milestone}")
        else:
            st.write("No upcoming milestones defined")
        
        # Display next stage milestones if available
        if stage.value < len(DevelopmentalStage) - 1:
            try:
                # Create a temporary system for next stage
                next_stage = DevelopmentalStage(stage.value + 1)
                temp_system = DevelopmentalSystem()
                temp_system.current_stage = next_stage
                next_stage_reqs = temp_system.get_stage_requirements()
                
                next_milestones = next_stage_reqs.get('current_milestones', [])
                if next_milestones:
                    st.write(f"\nNext Stage ({next_stage.name}) Milestones:")
                    for milestone in next_milestones:
                        st.write(f"◇ {milestone}")
            except Exception as e:
                if st.sidebar.checkbox("Debug Mode", value=False):
                    st.error(f"Error loading next stage: {str(e)}")
    
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

def format_detailed_age(birth_time):
    """Format age with detailed breakdown"""
    now = datetime.now()
    delta = now - birth_time
    
    total_seconds = delta.total_seconds()
    
    # Calculate all time units
    weeks = int(total_seconds // (7 * 24 * 3600))
    days = int((total_seconds % (7 * 24 * 3600)) // (24 * 3600))
    hours = int((total_seconds % (24 * 3600)) // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = int(total_seconds % 60)
    
    parts = []
    if weeks > 0:
        parts.append(f"{weeks}w")
    if days > 0 or weeks > 0:
        parts.append(f"{days}d")
    if hours > 0 or days > 0 or weeks > 0:
        parts.append(f"{hours}h")
    if minutes > 0 or hours > 0 or days > 0 or weeks > 0:
        parts.append(f"{minutes}m")
    parts.append(f"{seconds}s")  # Always show seconds
    
    return " ".join(parts)

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

def main():
    # Define stage_templates at the start of main()
    stage_templates = {
        DevelopmentalStage.NEWBORN: [
            "👶 [FEED] Feed the baby",
            "😴 [SLEEP] Help sleep",
            "🤗 [COMFORT] Comfort",
            "🎵 [SOOTHE] Soothe with sounds",
            "👀 [STIMULATE] Visual stimulation"
        ],
        DevelopmentalStage.EARLY_INFANCY: [
            "😊 [SMILE] Social smile",
            "🎈 [PLAY] Play peek-a-boo",
            "🗣️ [TALK] Baby talk",
            "🤲 [TOUCH] Gentle touch",
            "🎵 [SING] Sing lullaby"
        ],
        DevelopmentalStage.LATE_INFANCY: [
            "🚶 [ENCOURAGE] Encourage movement",
            "🎯 [GUIDE] Guide exploration",
            "🛡️ [PROTECT] Ensure safety",
            "🎮 [PLAY] Interactive play",
            "👋 [TEACH] Wave bye-bye"
        ],
        DevelopmentalStage.EARLY_TODDLER: [
            "📚 [TEACH] Basic words",
            "🚶‍♂️ [GUIDE] Walking practice",
            "🌟 [ENCOURAGE] New skills",
            "🎨 [CREATE] Simple art",
            "🧩 [SOLVE] Simple puzzles"
        ],
        DevelopmentalStage.LATE_TODDLER: [
            "📝 [TEACH] New words",
            "🎮 [PLAY] Pretend play",
            "✨ [PRAISE] Good behavior",
            "🤝 [SHARE] Teaching sharing",
            "🎨 [CREATE] Drawing shapes"
        ],
        DevelopmentalStage.EARLY_PRESCHOOL: [
            "🎭 [PRETEND] Imaginative play",
            "📖 [STORY] Storytelling",
            "🎨 [CREATE] Art project",
            "🔢 [COUNT] Number learning",
            "🌈 [EXPLORE] Color learning"
        ],
        DevelopmentalStage.LATE_PRESCHOOL: [
            "📚 [READ] Reading practice",
            "✍️ [WRITE] Writing letters",
            "🧮 [MATH] Basic math",
            "🤝 [SOCIAL] Group play",
            "🎯 [SOLVE] Problem solving"
        ],
        DevelopmentalStage.EARLY_CHILDHOOD: [
            "📖 [READ] Reading together",
            "✏️ [WRITE] Writing practice",
            "🔢 [MATH] Number work",
            "🔍 [DISCOVER] Science exploration",
            "🎨 [CREATE] Creative projects"
        ],
        DevelopmentalStage.MIDDLE_CHILDHOOD: [
            "📚 [STUDY] Academic work",
            "🤝 [TEAM] Team projects",
            "🎯 [GOAL] Goal setting",
            "🧪 [EXPERIMENT] Science projects",
            "🎭 [EXPRESS] Self expression"
        ],
        DevelopmentalStage.LATE_CHILDHOOD: [
            "🔍 [RESEARCH] Independent research",
            "💭 [DISCUSS] Complex topics",
            "📝 [WRITE] Creative writing",
            "🤝 [MENTOR] Peer mentoring",
            "🌟 [ACHIEVE] Achievement focus"
        ],
        DevelopmentalStage.EARLY_ELEMENTARY: [
            "📊 [PROJECT] Project work",
            "👥 [COLLABORATE] Team collaboration",
            "🔬 [INVESTIGATE] Scientific method",
            "📝 [REPORT] Report writing",
            "🎯 [PLAN] Project planning"
        ],
        DevelopmentalStage.MIDDLE_ELEMENTARY: [
            "🧪 [ANALYZE] Data analysis",
            "👥 [LEAD] Team leadership",
            "💡 [INNOVATE] Creative solutions",
            "📊 [PRESENT] Presentations",
            "🎯 [ACHIEVE] Goal achievement"
        ],
        DevelopmentalStage.LATE_ELEMENTARY: [
            "🔬 [RESEARCH] Advanced research",
            "💭 [CRITIQUE] Critical analysis",
            "📚 [STUDY] Independent study",
            "🎯 [SOLVE] Complex problems",
            "👥 [MENTOR] Peer teaching"
        ],
        DevelopmentalStage.EARLY_ADOLESCENCE: [
            "🤔 [REFLECT] Self-reflection",
            "💭 [EXPLORE] Identity exploration",
            "🤝 [CONNECT] Social connections",
            "🎯 [GOAL] Personal goals",
            "💡 [EXPRESS] Self expression"
        ],
        DevelopmentalStage.MIDDLE_ADOLESCENCE: [
            "🧭 [GUIDE] Life guidance",
            "💭 [VALUES] Value discussion",
            "🎯 [PLAN] Future planning",
            "👥 [SOCIAL] Social skills",
            "📚 [LEARN] Advanced learning"
        ],
        DevelopmentalStage.LATE_ADOLESCENCE: [
            "🎓 [PREPARE] College prep",
            "💼 [CAREER] Career planning",
            "💰 [FINANCE] Financial planning",
            "🤝 [RELATE] Relationships",
            "🌟 [GROW] Personal growth"
        ],
        DevelopmentalStage.YOUNG_ADULT: [
            "💼 [CAREER] Career development",
            "💡 [LIFE] Life skills",
            "💰 [MANAGE] Financial management",
            "❤️ [RELATE] Relationships",
            "🎯 [ACHIEVE] Goal achievement"
        ],
        DevelopmentalStage.MATURE_ADULT: [
            "🌟 [WISDOM] Share wisdom",
            "👥 [MENTOR] Mentorship",
            "🌍 [IMPACT] Community impact",
            "💭 [REFLECT] Life reflection",
            "🎯 [LEGACY] Legacy building"
        ]
    }

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
        # Use birth_time from session state instead of child object
        st.metric("Age", format_detailed_age(st.session_state.birth_time))
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
            # Use birth_time from session state
            birth_time = format_detailed_age(st.session_state.birth_time)
            st.info(f"🎂 Time since birth: {birth_time}")
            
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
        # Add debug mode toggle
        debug_mode = st.sidebar.checkbox("Debug Mode", value=False, 
                                       help="Show raw LLM responses and processing details")
        
        # Add interaction guide toggle in sidebar
        show_interaction_guide = st.sidebar.checkbox("Show Interaction Guide", value=False,
                                                   help="Display a comprehensive guide of all possible interactions")
        
        if show_interaction_guide:
            st.subheader("📚 Interaction Guide")
            
            # Display categorized interactions
            for category, info in stage_templates.items():
                # Convert the stage enum to a readable string
                stage_name = category.name.replace('_', ' ').title()
                with st.expander(stage_name, expanded=False):
                    # Display the first template as a description
                    st.write(f"**Available Actions:**")
                    for template in info:
                        st.write(template)
            
            st.divider()
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Teaching & Nurturing Interface")
            
            # Interact with Child
            st.subheader("Interact with Child")
            current_behaviors = st.session_state.child.curriculum.get_stage_requirements()['behaviors']
            
            # Create a list of all interactions with their stage information
            all_interactions = []
            for stage, templates in stage_templates.items():
                for template in templates:
                    all_interactions.append({
                        "template": template,
                        "stage": stage.name,
                        "is_current": stage == st.session_state.child.curriculum.current_stage
                    })

            # Interaction selection method
            selection_method = st.radio(
                "Interaction Selection Method:",
                ["Stage-Appropriate", "All Interactions", "By Category"],
                horizontal=True
            )

            if selection_method == "Stage-Appropriate":
                template_options = stage_templates.get(st.session_state.child.curriculum.current_stage, ["Custom"])
                selected_template = st.selectbox(
                    "Current Stage Interactions:", 
                    ["Custom"] + template_options,
                    key="interaction_type"
                )
            
            elif selection_method == "All Interactions":
                # Group interactions by stage
                st.write("💡 Current stage interactions are highlighted in green")
                selected_template = st.selectbox(
                    "All Available Interactions:",
                    ["Custom"] + [
                        f"{interaction['template']} {'✨' if interaction['is_current'] else ''}"
                        for interaction in all_interactions
                    ],
                    key="interaction_type_all",
                    format_func=lambda x: x.replace("✨", " (Current Stage)") if "✨" in x else x
                )
                # Remove the ✨ if present
                if selected_template != "Custom":
                    selected_template = selected_template.replace(" ✨", "")
            
            else:  # By Category
                # Create category mapping
                interaction_categories = {
                    "Basic Care (0-6 months) 👶": [t for t in all_interactions 
                        if t["stage"] in ["NEWBORN", "EARLY_INFANCY"]],
                    "Early Development (6-24 months) 🚶": [t for t in all_interactions 
                        if t["stage"] in ["LATE_INFANCY", "EARLY_TODDLER", "LATE_TODDLER"]],
                    "Preschool Learning (2-4 years) 📚": [t for t in all_interactions 
                        if t["stage"] in ["EARLY_PRESCHOOL", "LATE_PRESCHOOL"]],
                    "Early Education (4-7 years) 🎓": [t for t in all_interactions 
                        if t["stage"] in ["EARLY_CHILDHOOD", "MIDDLE_CHILDHOOD", "LATE_CHILDHOOD"]],
                    "Elementary Development (7-11 years) 🏫": [t for t in all_interactions 
                        if t["stage"] in ["EARLY_ELEMENTARY", "MIDDLE_ELEMENTARY", "LATE_ELEMENTARY"]],
                    "Adolescent Growth (11-18 years) 🌱": [t for t in all_interactions 
                        if t["stage"] in ["EARLY_ADOLESCENCE", "MIDDLE_ADOLESCENCE", "LATE_ADOLESCENCE"]],
                    "Adult Development (18+ years) 🌟": [t for t in all_interactions 
                        if t["stage"] in ["YOUNG_ADULT", "MATURE_ADULT"]]
                }
                
                # Category selection
                selected_category = st.selectbox(
                    "Select Age Category:",
                    list(interaction_categories.keys()),
                    key="category_select"
                )
                
                # Show interactions for selected category
                st.write("💡 Current stage interactions are highlighted in green")
                selected_template = st.selectbox(
                    f"Interactions for {selected_category}:",
                    ["Custom"] + [
                        f"{interaction['template']} {'✨' if interaction['is_current'] else ''}"
                        for interaction in interaction_categories[selected_category]
                    ],
                    key="interaction_type_category",
                    format_func=lambda x: x.replace("✨", " (Current Stage)") if "✨" in x else x
                )
                # Remove the ✨ if present
                if selected_template != "Custom":
                    selected_template = selected_template.replace(" ✨", "")

            # Input field
            if selected_template == "Custom":
                user_input = st.text_input(
                    "Say something to the child:", 
                    key="user_input"
                )
            else:
                # Extract the action part from the template
                action = selected_template.split("] ")[0] + "]"
                user_input = st.text_input(
                    "Say something to the child:",
                    value=action,
                    key="user_input"
                )

            # Add a warning if using interaction from a different stage
            if selected_template != "Custom":
                template_stage = next(
                    (interaction["stage"] for interaction in all_interactions 
                     if interaction["template"] == selected_template),
                    None
                )
                if template_stage and template_stage != st.session_state.child.curriculum.current_stage.name:
                    st.warning(f"⚠️ This interaction is designed for the {template_stage.replace('_', ' ').title()} stage. Current stage is {st.session_state.child.curriculum.current_stage.name.replace('_', ' ').title()}. Adjust your approach accordingly.")

            col1, col2 = st.columns([3, 1])
            with col2:
                st.write("")
                st.write("")
                interact_button = st.button(
                    "Interact", 
                    use_container_width=True,
                    key="interact_button"
                )
            
            if interact_button and user_input:
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
                    
                    # Ensure emotional vector is on correct device
                    if 'emotional_vector' in stimulus:
                        stimulus['emotional_vector'] = ensure_tensor_device(
                            torch.tensor(stimulus['emotional_vector'])
                        )
                    
                    # Update child's emotional state
                    st.session_state.child.update_emotions(stimulus['emotional_vector'])
                    
                    # Process child's perception and response
                    perception = st.session_state.child.perceive(stimulus)
                    response = st.session_state.child.respond(perception)
                    
                    if debug_mode:
                        with st.expander("🔍 Debug: Child Processing", expanded=True):
                            st.write("Perception:", perception)
                            st.write("Emotional State:", st.session_state.child.emotional_state.cpu().numpy())
                            st.write("Current Stage:", st.session_state.child.curriculum.current_stage.name)
                    
                    # Add to conversation history
                    interaction_data = {
                        "timestamp": datetime.now(),
                        "user": user_input,
                        "mother": stimulus.get('text', 'No response'),
                        "child": response,
                        "emotion": st.session_state.child.express_feeling(),
                        "stage": st.session_state.child.curriculum.current_stage.name
                    }
                    st.session_state.conversation_history.append(interaction_data)
                    
                    # Store emotional state history
                    emotional_state = st.session_state.child.emotional_state.cpu().numpy()
                    st.session_state.emotional_history.append(emotional_state)
                    
                    # Update teaching history
                    teaching_data = {
                        "date": datetime.now(),
                        "topic": selected_template if selected_template != "Custom" else "Custom Interaction",
                        "method": stimulus.get('text', 'No response'),
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
                
                except Exception as e:
                    st.error(f"Error during interaction: {str(e)}")
                    if debug_mode:
                        st.exception(e)  # This will show the full traceback
                    if "device" in str(e).lower():
                        st.info("Attempting to fix device mismatch...")
                        try:
                            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                            st.session_state.child.to(device)
                            st.success("Device mismatch fixed. Please try the interaction again.")
                        except Exception as device_e:
                            st.error(f"Could not fix device mismatch: {str(device_e)}")
            
            # Add debug history panel
            if debug_mode and st.session_state.conversation_history:
                st.subheader("🔍 Debug: Interaction History")
                for idx, interaction in enumerate(reversed(st.session_state.conversation_history[-5:])):
                    with st.expander(f"Interaction {len(st.session_state.conversation_history)-idx}", expanded=False):
                        st.json(interaction)
                        if hasattr(st.session_state.child, 'decision_history'):
                            st.write("Decision History for this interaction:", 
                                   st.session_state.child.decision_history[-(idx+1):])
            
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
            try:
                # Create a timestamp for the filename
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                
                # Convert datetime objects to ISO format strings for saving
                conversation_history = [
                    {**item, 'timestamp': item['timestamp'].isoformat()}
                    for item in st.session_state.conversation_history
                ]
                
                teaching_history = [
                    {**item, 'date': item['date'].isoformat()}
                    for item in st.session_state.teaching_history
                ]
                
                # Save both child and session state
                save_data = {
                    'child_state_dict': st.session_state.child.brain.state_dict(),
                    'emotional_state': st.session_state.child.emotional_state.cpu().numpy().tolist(),
                    'birth_date': st.session_state.birth_time.isoformat(),
                    'conversation_history': conversation_history,
                    'emotional_history': [e.tolist() if isinstance(e, torch.Tensor) else e for e in st.session_state.emotional_history],
                    'learning_history': st.session_state.learning_history,
                    'milestone_history': st.session_state.milestone_history,
                    'complexity_history': st.session_state.complexity_history,
                    'teaching_history': teaching_history,
                    'development_metrics': st.session_state.development_metrics,
                    'current_stage': st.session_state.child.curriculum.current_stage.name
                }
                
                # Create save directory if it doesn't exist
                save_dir = "checkpoints"
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                
                save_path = os.path.join(save_dir, f"digital_child_state_{timestamp}.pth")
                torch.save(save_data, save_path)
                st.success(f"Full state saved to {save_path}")
                
                # Save a backup copy
                backup_path = os.path.join(save_dir, "digital_child_state_latest.pth")
                torch.save(save_data, backup_path)
                st.info("Backup saved as 'digital_child_state_latest.pth'")
                
            except Exception as e:
                st.error(f"Error saving state: {str(e)}")
                if st.sidebar.checkbox("Debug Mode", value=False):
                    st.exception(e)
    
    with save_load_cols[1]:
        uploaded_file = st.file_uploader("Load State", type="pth")
        if uploaded_file is not None:
            try:
                # Add datetime to safe globals
                torch.serialization.add_safe_globals(['datetime'])
                
                # Load with weights_only=False to handle datetime objects
                save_data = torch.load(
                    uploaded_file,
                    weights_only=False,
                    map_location=st.session_state.child.device
                )
                
                # Load child state
                st.session_state.child.brain.load_state_dict(save_data['child_state_dict'])
                
                # Restore emotional state
                st.session_state.child.emotional_state = torch.tensor(
                    save_data['emotional_state'],
                    device=st.session_state.child.device
                )
                
                # Restore birth time
                st.session_state.birth_time = datetime.fromisoformat(save_data['birth_date'])
                
                # Load session state with datetime conversion
                st.session_state.conversation_history = [
                    {**item, 'timestamp': datetime.fromisoformat(item['timestamp']) 
                     if isinstance(item['timestamp'], str) else item['timestamp']}
                    for item in save_data['conversation_history']
                ]
                
                # Convert emotional history tensors
                st.session_state.emotional_history = [
                    torch.tensor(e, device=st.session_state.child.device) 
                    if isinstance(e, list) else e 
                    for e in save_data['emotional_history']
                ]
                
                st.session_state.learning_history = save_data['learning_history']
                st.session_state.milestone_history = save_data['milestone_history']
                st.session_state.complexity_history = save_data['complexity_history']
                
                # Convert teaching history timestamps
                st.session_state.teaching_history = [
                    {**item, 'date': datetime.fromisoformat(item['date']) 
                     if isinstance(item['date'], str) else item['date']}
                    for item in save_data['teaching_history']
                ]
                
                st.session_state.development_metrics = save_data['development_metrics']
                
                # Restore current stage
                if 'current_stage' in save_data:
                    st.session_state.child.curriculum.current_stage = DevelopmentalStage[save_data['current_stage']]
                
                st.success("Full state loaded successfully!")
                st.info("Please refresh the page to see all restored state.")
            except Exception as e:
                st.error(f"Error loading state: {str(e)}")
                if st.sidebar.checkbox("Debug Mode", value=False):
                    st.exception(e)
                st.info("If you trust this file, try restarting the application and loading again.")
    
    with save_load_cols[2]:
        st.info("Save/Load functionality preserves all history and development progress.")

    # Add a new section for logs in the sidebar
    with st.sidebar:
        st.subheader("🔍 LM Studio Logs")
        if st.checkbox("Show Server Logs", value=False):
            log_placeholder = st.empty()
            
            # Create a container for scrollable logs
            with st.container():
                # Initialize or get log history from session state
                if 'log_history' not in st.session_state:
                    st.session_state.log_history = []
                
                # Add a clear logs button
                if st.button("Clear Logs"):
                    st.session_state.log_history = []
                
                # Stream logs
                try:
                    for log in stream_llm_logs():
                        # Add new log to history
                        st.session_state.log_history.append(log)
                        # Keep only last 100 logs
                        if len(st.session_state.log_history) > 100:
                            st.session_state.log_history.pop(0)
                        
                        # Display all logs in reverse order (newest first)
                        log_placeholder.code('\n'.join(reversed(st.session_state.log_history)))
                        
                except Exception as e:
                    st.error(f"Error streaming logs: {str(e)}")
                    st.info("Make sure LM Studio server is running on http://localhost:1234")

if __name__ == "__main__":
    main()