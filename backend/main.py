"""Main module for the Neural Child project.

This module implements the core functionality of the Neural Child system,
including the MotherLLM and DigitalChild classes that simulate the interaction
between a mother and a developing child.
"""

import torch
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import psutil
from dataclasses import dataclass
import logging
import os
import time

from llm_module import chat_completion
from child_model import DynamicNeuralChild
from developmental_stages import DevelopmentalStage, DevelopmentalSystem, get_stage_prompt
from memory_module import DifferentiableMemory
from moral_network import MoralPolicyNetwork
from metacognition import MetacognitionSystem
from self_supervised_trainer import AutonomousTrainer
from text_embed import get_embeddings
from autonomous_learner import AutonomousLearner
from sandbox_manager import SandboxManager
from training_system import DevelopmentalTrainer
from emotional_regulation import EmotionalRegulation
from config import config
from curriculum import DevelopmentalCurriculum

# Remove circular import
# from main import DigitalChild, MotherLLM

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette.sse import EventSourceResponse
import asyncio
import json
import random
from typing import Dict, List
import uvicorn

from .models import EmotionalState, WarningMetrics, WarningIndicators, InteractionRequest, DevelopmentState

app = FastAPI(title="Neural Child Development API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Simulated state
current_state = DevelopmentState(
    emotionalState=EmotionalState(
        happiness=0.7,
        sadness=0.2,
        anger=0.1,
        fear=0.1,
        surprise=0.3,
        disgust=0.1,
        trust=0.8,
        anticipation=0.6
    ),
    warnings=WarningIndicators(
        warning_state="normal",
        metrics=WarningMetrics(
            emotional_stability=0.8,
            learning_efficiency=0.75,
            attention_level=0.9,
            overstimulation_risk=0.2
        ),
        recent_warnings=[]
    ),
    developmentSpeed=1.0,
    currentStage="infant",
    ageMonths=6.0
)

@app.get("/api/development/state")
async def get_development_state() -> DevelopmentState:
    return current_state

async def state_generator():
    while True:
        # Simulate state changes
        current_state.emotionalState.happiness += random.uniform(-0.1, 0.1)
        current_state.emotionalState.happiness = max(0, min(1, current_state.emotionalState.happiness))
        
        # Update age based on development speed
        current_state.ageMonths += 0.1 * current_state.developmentSpeed
        
        # Convert to JSON string
        json_data = current_state.model_dump_json()
        yield {
            "event": "state_update",
            "data": json_data
        }
        await asyncio.sleep(2)  # Update every 2 seconds

@app.get("/api/development/stream")
async def stream_development_state():
    return EventSourceResponse(state_generator())

@app.post("/api/development/emotional-state")
async def update_emotional_state(new_state: EmotionalState):
    current_state.emotionalState = new_state
    return {"status": "success", "message": "Emotional state updated"}

@app.post("/api/development/speed")
async def update_speed(speed: float):
    if speed < 0:
        raise HTTPException(status_code=400, detail="Speed cannot be negative")
    current_state.developmentSpeed = speed
    return {"status": "success", "message": f"Development speed set to {speed}"}

@app.get("/api/development/warnings")
async def get_warnings() -> WarningIndicators:
    return current_state.warnings

@app.post("/api/development/interact")
async def interact(request: InteractionRequest):
    # Simulate interaction effects
    emotional_change = random.uniform(-0.2, 0.2)
    current_state.emotionalState.happiness = max(0, min(1, 
        current_state.emotionalState.happiness + emotional_change))
    
    return {
        "response": f"Interaction processed: {request.interaction}",
        "emotionalState": current_state.emotionalState
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000) 