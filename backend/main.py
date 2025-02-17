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
import shutil
import numpy as np
from pydantic import BaseModel

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

from fastapi import FastAPI, HTTPException, UploadFile, File
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

class NeuralActivityData(BaseModel):
    """Real-time neural network activity data."""
    timestamp: float
    activity_values: List[float]
    mean_activation: float
    spike_rate: float
    network_load: float

class NetworkTopologyData(BaseModel):
    """3D network topology data."""
    node_positions: List[List[float]]  # [[x, y, z], ...]
    edge_connections: List[List[int]]  # [[node1_idx, node2_idx], ...]
    node_activations: List[float]
    edge_weights: List[float]

# Global state for neural metrics
neural_activity_buffer = []
MAX_ACTIVITY_POINTS = 100

def generate_neural_activity() -> NeuralActivityData:
    """Generate simulated neural network activity data."""
    current_time = time.time()
    activity = np.random.normal(0.5, 0.15, 10).clip(0, 1).tolist()
    mean_act = np.mean(activity)
    spike_rate = np.random.normal(100, 10)  # Hz
    network_load = np.random.normal(0.7, 0.1).clip(0, 1)
    
    return NeuralActivityData(
        timestamp=current_time,
        activity_values=activity,
        mean_activation=mean_act,
        spike_rate=spike_rate,
        network_load=network_load
    )

def generate_network_topology() -> NetworkTopologyData:
    """Generate simulated 3D network topology data."""
    num_nodes = 50
    # Generate node positions in 3D space
    positions = np.random.normal(0, 1, (num_nodes, 3)).tolist()
    
    # Generate random connections (edges)
    num_edges = num_nodes * 2
    edges = []
    for _ in range(num_edges):
        node1 = np.random.randint(0, num_nodes)
        node2 = np.random.randint(0, num_nodes)
        if node1 != node2:
            edges.append([node1, node2])
    
    # Generate node activations and edge weights
    node_acts = np.random.normal(0.5, 0.15, num_nodes).clip(0, 1).tolist()
    edge_weights = np.random.normal(0.5, 0.15, len(edges)).clip(0, 1).tolist()
    
    return NetworkTopologyData(
        node_positions=positions,
        edge_connections=edges,
        node_activations=node_acts,
        edge_weights=edge_weights
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

@app.post("/api/models/upload")
async def upload_model(
    model_file: UploadFile = File(...),
    model_name: str = None
) -> Dict[str, str]:
    """Upload a specific model file to the system.
    
    Args:
        model_file (UploadFile): The model file to upload
        model_name (str, optional): Custom name for the model. Defaults to original filename.
        
    Returns:
        Dict[str, str]: Response containing upload status and model information
        
    Raises:
        HTTPException: If file upload fails or invalid file type
    """
    if not model_file.filename.endswith(('.pt', '.pth', '.ckpt')):
        raise HTTPException(
            status_code=400,
            detail="Invalid file type. Only PyTorch model files (.pt, .pth, .ckpt) are supported."
        )
    
    try:
        # Create models directory if it doesn't exist
        os.makedirs('models', exist_ok=True)
        
        # Use provided model name or original filename
        save_name = model_name or model_file.filename
        file_path = os.path.join('models', save_name)
        
        # Save uploaded file
        with open(file_path, 'wb') as buffer:
            shutil.copyfileobj(model_file.file, buffer)
            
        return {
            "status": "success",
            "message": f"Model uploaded successfully as {save_name}",
            "model_path": file_path
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to upload model: {str(e)}"
        )

@app.get("/api/neural/activity")
async def get_neural_activity() -> NeuralActivityData:
    """Get current neural network activity data."""
    return generate_neural_activity()

@app.get("/api/neural/topology")
async def get_network_topology() -> NetworkTopologyData:
    """Get current network topology data."""
    return generate_network_topology()

async def neural_metrics_generator():
    """Generate real-time neural metrics stream."""
    while True:
        activity_data = generate_neural_activity()
        topology_data = generate_network_topology()
        
        yield {
            "event": "neural_update",
            "data": {
                "activity": activity_data.model_dump(),
                "topology": topology_data.model_dump()
            }
        }
        await asyncio.sleep(1)  # Update every second

@app.get("/api/neural/stream")
async def stream_neural_metrics():
    """Stream real-time neural network metrics."""
    return EventSourceResponse(neural_metrics_generator())

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000) 