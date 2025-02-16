# training_system.py

import torch
import torch.nn as nn
import time
import numpy as np
from collections import deque, defaultdict
from pathlib import Path
from typing import Dict, Any, Optional
from config import config

class MovingAverageMonitor:
    def __init__(self, window_size: int = 50):
        self.window_size = window_size
        self.steps = 0
        self.error_log = []
        self.loss_buffer = deque(maxlen=window_size)
        self.grad_buffer = deque(maxlen=window_size)
        self.learning_rates = deque(maxlen=window_size)
        self.component_losses = defaultdict(lambda: deque(maxlen=window_size))
        
    def update_stats(self, total_loss: float, individual_losses: dict, gradient_norm: float, learning_rate: float) -> dict:
        self.steps += 1
        self.loss_buffer.append(total_loss)
        self.grad_buffer.append(gradient_norm)
        self.learning_rates.append(learning_rate)
        for key, value in individual_losses.items():
            self.component_losses[key].append(value)
        stats = {
            'step': self.steps,
            'total_loss': total_loss,
            'loss_ma': np.mean(self.loss_buffer),
            'loss_std': np.std(self.loss_buffer) if len(self.loss_buffer) > 1 else 0,
            'grad_norm': gradient_norm,
            'grad_ma': np.mean(self.grad_buffer),
            'learning_rate': learning_rate,
            'component_losses': {
                key: {
                    'current': value,
                    'mean': np.mean(list(self.component_losses[key])),
                    'std': np.std(list(self.component_losses[key])) if len(self.component_losses[key]) > 1 else 0
                }
                for key, value in individual_losses.items()
            }
        }
        return stats
        
    def check_loss_spike(self, current_loss: float, threshold: float) -> bool:
        if len(self.loss_buffer) < 2:
            return False
        loss_mean = np.mean(self.loss_buffer)
        loss_std = np.std(self.loss_buffer)
        if loss_std == 0:
            return current_loss > loss_mean * threshold
        z_score = (current_loss - loss_mean) / loss_std
        return z_score > threshold
        
    def check_gradient_issues(self, grad_norm: float) -> bool:
        if np.isnan(grad_norm) or np.isinf(grad_norm):
            return True
        if len(self.grad_buffer) < 2:
            return False
        grad_mean = np.mean(self.grad_buffer)
        grad_std = np.std(self.grad_buffer)
        if grad_norm < 1e-7:
            return True
        if grad_std > 0:
            z_score = (grad_norm - grad_mean) / grad_std
            return z_score > 3.0
        return False
        
    def log_error(self, error_type: str, error_message: str):
        self.error_log.append({
            'step': self.steps,
            'type': error_type,
            'message': error_message,
            'timestamp': time.time()
        })
        
    def summarize_episode(self, episode_stats: list) -> dict:
        if not episode_stats:
            return {'status': 'failed', 'error_log': self.error_log}
        summary = {
            'steps_completed': len(episode_stats),
            'final_loss': episode_stats[-1]['total_loss'],
            'mean_loss': np.mean([s['total_loss'] for s in episode_stats]),
            'loss_std': np.std([s['total_loss'] for s in episode_stats]),
            'mean_grad_norm': np.mean([s['grad_norm'] for s in episode_stats]),
            'component_trends': {},
            'error_log': self.error_log
        }
        for key in episode_stats[0]['component_losses'].keys():
            values = [s['component_losses'][key]['current'] for s in episode_stats]
            summary['component_trends'][key] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'trend': np.polyfit(range(len(values)), values, deg=1)[0]
            }
        return summary

class CheckpointManager:
    def __init__(self, model: nn.Module, save_dir: str, max_checkpoints: int = 5):
        self.model = model
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.max_checkpoints = max_checkpoints
        self.checkpoint_history = []
        self.stable_checkpoints = []
        
    def save_checkpoint(self, step: int, loss: float, stats: dict) -> str:
        checkpoint_path = self.save_dir / f"checkpoint_{step}.pt"
        checkpoint_data = {
            'step': step,
            'model_state': self.model.state_dict(),
            'loss': loss,
            'stats': stats,
            'timestamp': time.time()
        }
        torch.save(checkpoint_data, checkpoint_path)
        self.checkpoint_history.append({
            'path': checkpoint_path,
            'step': step,
            'loss': loss,
            'timestamp': time.time()
        })
        if self._is_stable_checkpoint(stats):
            self.stable_checkpoints.append(checkpoint_path)
        self._cleanup_old_checkpoints()
        return str(checkpoint_path)
        
    def _is_stable_checkpoint(self, stats: dict) -> bool:
        if 'loss_std' in stats and stats['loss_std'] > 0.5:
            return False
        if 'grad_ma' in stats and (stats['grad_ma'] < 1e-7 or stats['grad_ma'] > 10.0):
            return False
        if 'component_losses' in stats:
            for comp_stats in stats['component_losses'].values():
                if comp_stats['std'] > 0.5:
                    return False
        return True
        
    def _cleanup_old_checkpoints(self):
        protected = set(self.stable_checkpoints)
        remaining = sorted(
            [cp for cp in self.checkpoint_history if cp['path'] not in protected],
            key=lambda x: x['timestamp']
        )
        while len(remaining) + len(protected) > self.max_checkpoints:
            oldest = remaining.pop(0)
            if oldest['path'].exists():
                oldest['path'].unlink()
            self.checkpoint_history.remove(oldest)
            
    def load_last_stable(self) -> dict:
        if not self.stable_checkpoints:
            return None
        latest_stable = max(self.stable_checkpoints, key=lambda p: p.stat().st_mtime)
        if latest_stable.exists():
            checkpoint_data = torch.load(latest_stable)
            self.model.load_state_dict(checkpoint_data['model_state'])
            return checkpoint_data
        return None
        
    def load_best_checkpoint(self, metric: str = 'loss') -> dict:
        if not self.checkpoint_history:
            return None
        best_checkpoint = min(self.checkpoint_history, key=lambda x: x[metric])
        if best_checkpoint['path'].exists():
            checkpoint_data = torch.load(best_checkpoint['path'])
            self.model.load_state_dict(checkpoint_data['model_state'])
            return checkpoint_data
        return None

class EarlyStopping:
    def __init__(self, patience: int = 5, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.loss_history = deque(maxlen=100)
        
    def check(self, current_loss: float) -> bool:
        self.loss_history.append(current_loss)
        if len(self.loss_history) < self.patience:
            return False
        if self.best_loss is None:
            self.best_loss = current_loss
        elif current_loss < self.best_loss - self.min_delta:
            self.best_loss = current_loss
            self.counter = 0
        else:
            self.counter += 1
        if len(self.loss_history) >= 10:
            recent_mean = np.mean(list(self.loss_history)[-10:])
            recent_std = np.std(list(self.loss_history)[-10:])
            if recent_std > recent_mean * 0.5:
                return True
        return self.counter >= self.patience
        
    def reset(self):
        self.counter = 0
        self.best_loss = None
        self.loss_history.clear()

class DevelopmentalTrainer:
    def __init__(self,
                 child_model: nn.Module,
                 memory: nn.Module,
                 emotional_regulation: nn.Module,
                 curriculum_manager: Any,
                 mother_llm: Any,
                 metacognition_system: nn.Module,
                 config: Dict[str, Any]):
        self.child_model = child_model
        self.memory = memory
        self.emotional_regulation = emotional_regulation
        self.curriculum = curriculum_manager
        self.mother = mother_llm
        self.metacognition = metacognition_system
        
        # Training configuration
        self.device = config.get('device', torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        self.learning_rate = config.get('learning_rate', 3e-4)
        self.weight_decay = config.get('weight_decay', 0.01)
        self.gradient_clip_norm = config.get('gradient_clip_norm', 1.0)
        self.warmup_steps = config.get('warmup_steps', 1000)
        
        # Initialize optimizer
        self.optimizer = torch.optim.AdamW(
            self.child_model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        # Training state
        self.steps = 0
        self.best_performance = 0.0
        self.performance_history = []
        
        # Initialize monitoring with default window size if not provided
        window_size = config.get('moving_average_window', 50)
        self.monitoring = MovingAverageMonitor(window_size=window_size)
        self.checkpointing = CheckpointManager(model=self.child_model, save_dir='checkpoints', max_checkpoints=5)
        self._initialize_loss_weights()

    def _initialize_loss_weights(self):
        self.loss_weights = {'moral': 0.3, 'attachment': 0.3, 'emotional': 0.2, 'cognitive': 0.2}
        self.weight_momentum = 0.95
        self.weight_history = {k: deque(maxlen=100) for k in self.loss_weights}

    def update_loss_weights(self, stage_requirements: dict, current_losses: dict):
        base_weights = self._compute_stage_weights(stage_requirements)
        for key, loss in current_losses.items():
            self.weight_history[key].append(loss)
        loss_stats = {key: {'mean': np.mean(values), 'std': np.std(values) if len(values) > 1 else 0}
                      for key, values in self.weight_history.items()}
        total_inverse_mean = sum(1 / stats['mean'] for stats in loss_stats.values() if stats['mean'] > 0)
        adjusted_weights = {key: (1 / stats['mean']) / total_inverse_mean for key, stats in loss_stats.items() if stats['mean'] > 0}
        for key in self.loss_weights:
            target_weight = 0.5 * base_weights.get(key, 0.25) + 0.5 * adjusted_weights.get(key, 0.25)
            self.loss_weights[key] = self.weight_momentum * self.loss_weights[key] + (1 - self.weight_momentum) * target_weight

    def _compute_stage_weights(self, stage_requirements: dict) -> dict:
        return {'moral': 0.3, 'attachment': 0.3, 'emotional': 0.2, 'cognitive': 0.2}

    def training_step(self, input_data: torch.Tensor) -> float:
        """Execute one training step"""
        self.child_model.train()
        self.steps += 1
        
        try:
            # Get current stage characteristics
            stage = self.curriculum.get_stage_characteristics()
            
            # Forward pass
            output = self.child_model(input_data)
            
            # Get metacognitive assessment
            meta_output = self.metacognition(output)
            
            # Calculate losses
            reconstruction_loss = self._compute_reconstruction_loss(output, input_data)
            emotional_loss = self._compute_emotional_loss(output)
            complexity_loss = self._compute_complexity_loss(output, stage)
            
            # Combine losses with metacognitive weighting
            total_loss = (
                reconstruction_loss * meta_output['attention'].item() +
                emotional_loss * 0.3 +
                complexity_loss * 0.2
            )
            
            # Backward pass with gradient clipping
            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.child_model.parameters(),
                self.gradient_clip_norm
            )
            self.optimizer.step()
            
            # Update learning rate with warmup
            if self.steps < self.warmup_steps:
                lr_scale = min(1.0, float(self.steps) / self.warmup_steps)
                for pg in self.optimizer.param_groups:
                    pg['lr'] = self.learning_rate * lr_scale
            
            # Track performance
            performance = 1.0 - total_loss.item()
            self.performance_history.append(performance)
            
            if performance > self.best_performance:
                self.best_performance = performance
            
            return total_loss.item()
            
        except Exception as e:
            print(f"Error in training step: {str(e)}")
            return float('inf')
    
    def _compute_reconstruction_loss(self,
                                   output: torch.Tensor,
                                   target: torch.Tensor) -> torch.Tensor:
        """Compute reconstruction loss"""
        return nn.functional.mse_loss(output, target)
    
    def _compute_emotional_loss(self, output: torch.Tensor) -> torch.Tensor:
        """Compute emotional regulation loss"""
        try:
            emotional_state = self.emotional_regulation(output)
            target_state = torch.zeros_like(emotional_state)
            target_state[0] = 0.7  # Target for positive emotion
            return nn.functional.mse_loss(emotional_state, target_state)
        except Exception:
            return torch.tensor(0.0, device=self.device)
    
    def _compute_complexity_loss(self,
                               output: torch.Tensor,
                               stage_char: Any) -> torch.Tensor:
        """Compute complexity-based loss"""
        try:
            # Get complexity range for current stage
            min_complexity, max_complexity = stage_char.complexity_range
            
            # Calculate output complexity (using standard deviation as a proxy)
            complexity = torch.std(output)
            
            # Scale complexity to match stage requirements
            target_complexity = torch.tensor(
                (min_complexity + max_complexity) / 2,
                device=self.device
            )
            
            return nn.functional.mse_loss(complexity, target_complexity)
        except Exception:
            return torch.tensor(0.0, device=self.device)
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """Get current performance metrics"""
        if not self.performance_history:
            return {
                'current_performance': 0.0,
                'best_performance': 0.0,
                'average_performance': 0.0,
                'learning_rate': self.learning_rate
            }
            
        current = self.performance_history[-1]
        average = np.mean(self.performance_history[-100:])  # Last 100 steps
        
        return {
            'current_performance': current,
            'best_performance': self.best_performance,
            'average_performance': average,
            'learning_rate': self.optimizer.param_groups[0]['lr']
        }
    
    def save_checkpoint(self, path: str):
        """Save trainer state"""
        torch.save({
            'steps': self.steps,
            'best_performance': self.best_performance,
            'model_state': self.child_model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'performance_history': self.performance_history
        }, path)
    
    def load_checkpoint(self, path: str):
        """Load trainer state"""
        checkpoint = torch.load(path, map_location=self.device)
        self.steps = checkpoint['steps']
        self.best_performance = checkpoint['best_performance']
        self.child_model.load_state_dict(checkpoint['model_state'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        self.performance_history = checkpoint['performance_history']

    def train_episode(self, num_steps: int = 1000) -> dict:
        episode_stats = []
        early_stopping = EarlyStopping(patience=config['early_stopping_patience'])
        for step in range(num_steps):
            mother_stimulus = self.mother.generate_stimulus(self.curriculum.current_stage, self.child_model.express_feeling())
            step_stats = self.training_step(mother_stimulus['embedding'])
            if step_stats == float('inf'):
                print(f"Training failed at step {step}")
                break
            episode_stats.append({
                'step': step,
                'total_loss': step_stats,
                'performance': self.get_performance_metrics()['current_performance']
            })
            if early_stopping.check(step_stats):
                print(f"Early stopping triggered at step {step}")
                break
        return self.monitoring.summarize_episode(episode_stats)
