"""Development progress component for the Neural Child UI."""

import streamlit as st
import plotly.graph_objects as go
from backend.developmental_stages import DevelopmentalStage, STAGE_DEFINITIONS

def render_development_progress():
    """Render the development progress showing milestones and achievements."""
    if 'child' not in st.session_state:
        st.warning("No child instance available. Please initialize the session first.")
        return
        
    child = st.session_state.child
    current_stage = child.current_stage
    
    # Display current stage information
    st.subheader(f"Current Stage: {current_stage.name.replace('_', ' ').title()}")
    
    if current_stage in STAGE_DEFINITIONS:
        stage_info = STAGE_DEFINITIONS[current_stage]
        
        # Display age range
        min_age, max_age = stage_info['age_range']
        st.write(f"Age Range: {min_age} to {max_age} months")
        
        # Create columns for different aspects
        col1, col2 = st.columns(2)
        
        with col1:
            # Required Skills
            st.write("#### Required Skills")
            for skill in stage_info['required_skills']:
                progress = child.get_skill_progress(skill)
                st.progress(progress, text=f"{skill}: {progress*100:.1f}%")
            
            # Learning Focus
            st.write("#### Learning Focus")
            for focus in stage_info['learning_focus']:
                st.write(f"- {focus}")
        
        with col2:
            # Current Milestones
            st.write("#### Current Milestones")
            for milestone in stage_info['current_milestones']:
                progress = child.get_milestone_progress(milestone)
                st.progress(progress, text=f"{milestone}: {progress*100:.1f}%")
            
            # Upcoming Milestones
            st.write("#### Upcoming Milestones")
            for milestone in stage_info['upcoming_milestones']:
                st.write(f"- {milestone}")
    
    # Display overall development progress
    st.subheader("Overall Development Progress")
    
    # Create progress chart using numerical values for stages
    stages = [
        DevelopmentalStage.NEWBORN,  # 0
        DevelopmentalStage.EARLY_INFANCY,  # 1
        DevelopmentalStage.LATE_INFANCY,  # 2
        DevelopmentalStage.EARLY_TODDLER,  # 3
        DevelopmentalStage.LATE_TODDLER,  # 4
        DevelopmentalStage.EARLY_PRESCHOOL,  # 5
        DevelopmentalStage.LATE_PRESCHOOL,  # 6
        DevelopmentalStage.EARLY_CHILDHOOD,  # 7
        DevelopmentalStage.MIDDLE_CHILDHOOD,  # 8
        DevelopmentalStage.LATE_CHILDHOOD,  # 9
        DevelopmentalStage.EARLY_ELEMENTARY,  # 10
        DevelopmentalStage.MIDDLE_ELEMENTARY,  # 11
        DevelopmentalStage.LATE_ELEMENTARY,  # 12
        DevelopmentalStage.EARLY_ADOLESCENCE,  # 13
        DevelopmentalStage.MIDDLE_ADOLESCENCE,  # 14
        DevelopmentalStage.LATE_ADOLESCENCE,  # 15
        DevelopmentalStage.YOUNG_ADULT,  # 16
        DevelopmentalStage.MATURE_ADULT  # 17
    ]
    
    try:
        current_index = current_stage.value
        progress_data = []
        
        for i, stage in enumerate(stages):
            if i < current_index:
                progress = 1.0
            elif i == current_index:
                progress = child.get_stage_progress()
            else:
                progress = 0.0
            progress_data.append(progress)
        
        fig = go.Figure(data=[
            go.Bar(
                x=[stage.name.replace('_', ' ').title() for stage in stages],
                y=progress_data,
                marker_color=['#00ff00' if i < current_index else 
                             '#ffff00' if i == current_index else 
                             '#808080' for i in range(len(stages))]
            )
        ])
        
        fig.update_layout(
            title="Development Stage Progress",
            xaxis_title="Developmental Stages",
            yaxis_title="Progress",
            yaxis=dict(range=[0, 1]),
            xaxis=dict(tickangle=45)
        )
        
        st.plotly_chart(fig, use_container_width=True)
    except ValueError as e:
        st.error(f"Error displaying progress chart: Invalid stage {current_stage}")
    except Exception as e:
        st.error(f"Error displaying progress chart: {str(e)}") 