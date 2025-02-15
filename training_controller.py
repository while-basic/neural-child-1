import os
import json
import torch
import logging
from datetime import datetime
from pathlib import Path
import config

class TrainingController:
    def __init__(self, child):
        self.child = child
        self.current_iteration = 0
        self.setup_logging()
        self.create_directories()
        
    def setup_logging(self):
        logging.basicConfig(
            filename=os.path.join(config.LOG_DIR, f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        
    def create_directories(self):
        Path(config.CHECKPOINT_DIR).mkdir(exist_ok=True)
        Path(config.LOG_DIR).mkdir(exist_ok=True)
        
    def save_checkpoint(self):
        checkpoint = {
            'iteration': self.current_iteration,
            'child_state': self.child.brain.state_dict(),
            'memory_state': self.child.memory.state_dict(),
            'curriculum_state': self.child.curriculum.current_stage,
            'birth_date': self.child.birth_date.isoformat(),
            'emotional_state': self.child.emotional_state.cpu().tolist()
        }
        
        checkpoint_path = os.path.join(
            config.CHECKPOINT_DIR, 
            f'checkpoint_iter_{self.current_iteration}.pth'
        )
        torch.save(checkpoint, checkpoint_path)
        logging.info(f"Saved checkpoint at iteration {self.current_iteration}")
        
    def load_checkpoint(self, checkpoint_path):
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path)
            self.current_iteration = checkpoint['iteration']
            self.child.brain.load_state_dict(checkpoint['child_state'])
            self.child.memory.load_state_dict(checkpoint['memory_state'])
            self.child.curriculum.current_stage = checkpoint['curriculum_state']
            self.child.birth_date = datetime.fromisoformat(checkpoint['birth_date'])
            self.child.emotional_state = torch.tensor(
                checkpoint['emotional_state'], 
                device=config.DEVICE
            )
            logging.info(f"Loaded checkpoint from iteration {self.current_iteration}")
            return True
        return False
        
    def should_continue(self):
        return self.current_iteration < config.MAX_ITERATIONS
        
    def step(self):
        self.current_iteration += 1
        if self.current_iteration % config.SAVE_INTERVAL == 0:
            self.save_checkpoint()
            
    def log_metrics(self, metrics):
        logging.info(f"Iteration {self.current_iteration} metrics: {json.dumps(metrics)}")
