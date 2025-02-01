# training_system.py

import torch
import torch.nn as nn
import time
import numpy as np
from collections import deque, defaultdict
from pathlib import Path

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
    def __init__(self, child_model: nn.Module, memory: nn.Module, emotional_regulation: nn.Module,
                 curriculum_manager, mother_llm, metacognition_system, config: dict):
        self.child = child_model
        self.memory = memory
        self.emotional_regulation = emotional_regulation
        self.curriculum = curriculum_manager
        self.mother = mother_llm
        self.metacognition = metacognition_system
        self.device = config.get('device', 'cuda')
        self.config = {
            'learning_rate': config.get('learning_rate', 3e-4),
            'weight_decay': config.get('weight_decay', 0.01),
            'gradient_clip_norm': config.get('gradient_clip_norm', 1.0),
            'warmup_steps': config.get('warmup_steps', 1000),
            'checkpoint_interval': config.get('checkpoint_interval', 100),
            'moving_average_window': config.get('moving_average_window', 50),
            'early_stopping_patience': config.get('early_stopping_patience', 5),
            'loss_spike_threshold': config.get('loss_spike_threshold', 2.0),
        }
        self.optimizer = torch.optim.AdamW(
            self.child.parameters(),
            lr=self.config['learning_rate'],
            weight_decay=self.config['weight_decay']
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=self.config['warmup_steps'],
            T_mult=2
        )
        self.monitoring = MovingAverageMonitor(window_size=self.config['moving_average_window'])
        self.checkpointing = CheckpointManager(model=self.child, save_dir='checkpoints', max_checkpoints=5)
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

    def training_step(self, stimulus: torch.Tensor, mother_response: dict) -> dict:
        self.optimizer.zero_grad()
        stage_requirements = self.curriculum.get_stage_requirements()
        try:
            child_output = self.child(stimulus)
        except RuntimeError as e:
            self.monitoring.log_error('forward_pass', str(e))
            return self._handle_training_error('Forward pass failed')
        try:
            memory_context = self.memory.retrieve_memories(stimulus)
            regulated_emotion = self.emotional_regulation.regulate(
                emotional_state=self.child.emotional_state,
                stimulus=stimulus,
                memory_context=memory_context
            )
        except RuntimeError as e:
            self.monitoring.log_error('emotional_regulation', str(e))
            return self._handle_training_error('Emotional regulation failed')
        try:
            losses = self._compute_losses(child_output, regulated_emotion, mother_response, stage_requirements)
        except RuntimeError as e:
            self.monitoring.log_error('loss_computation', str(e))
            return self._handle_training_error('Loss computation failed')
        self.update_loss_weights(stage_requirements, losses)
        total_loss = sum(self.loss_weights[key] * value for key, value in losses.items())
        if self.monitoring.check_loss_spike(total_loss.item(), self.config['loss_spike_threshold']):
            return self._handle_training_error('Loss spike detected')
        try:
            total_loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(self.child.parameters(), self.config['gradient_clip_norm'])
            if self.monitoring.check_gradient_issues(grad_norm):
                return self._handle_training_error('Gradient issues detected')
        except RuntimeError as e:
            self.monitoring.log_error('backward_pass', str(e))
            return self._handle_training_error('Backward pass failed')
        self.optimizer.step()
        self.scheduler.step()
        self.memory.record_experience(stimulus, child_output, total_loss.item(), time.time(), self.child.emotional_state)
        stage_update = self.curriculum.update_stage({
            'success_rate': (1.0 - total_loss.item()),
            'emotional_stability': regulated_emotion.get('emotional_state').mean().item(),
            'cognitive_complexity': self.metacognition(child_output).get('complexity', torch.tensor(0.5, device=self.device)).item(),
            'social_awareness': self.child.last_attachment_trust.item()
        })
        if stage_update:
            self.monitoring.error_log.append({'step': self.monitoring.steps, 'transition': stage_update, 'timestamp': time.time()})
        stats = self.monitoring.update_stats(
            total_loss=total_loss.item(),
            individual_losses={k: v.item() if torch.is_tensor(v) else v for k, v in losses.items()},
            gradient_norm=grad_norm,
            learning_rate=self.scheduler.get_last_lr()[0]
        )
        if self.monitoring.steps % self.config['checkpoint_interval'] == 0:
            self.checkpointing.save_checkpoint(step=self.monitoring.steps, loss=total_loss.item(), stats=stats)
        return stats

    def _compute_losses(self, child_output, regulated_emotion, mother_response, stage_requirements) -> dict:
        moral_loss = nn.MSELoss()(self.child.morality(child_output)['moral_score'], torch.tensor(mother_response.get('reward_score', 0.5), device=self.device))
        target_trust = torch.tensor(mother_response.get('emotional_context', {}).get('trust', 0.5), device=self.device)
        attachment_loss = nn.MSELoss()(self.child.last_attachment_trust, target_trust)
        base_emotional_loss = nn.MSELoss()(regulated_emotion['emotional_state'], torch.tensor(mother_response.get('emotional_vector', [0.5]*4), device=self.device))
        allowed_range = stage_requirements['metrics'].get('emotional_stability', 0.5)
        range_penalty = torch.relu(torch.abs(regulated_emotion['emotional_state']) - allowed_range).mean()
        emotional_loss = base_emotional_loss + 0.5 * range_penalty
        target_complexity = stage_requirements['metrics'].get('language_complexity', 0.5)
        output_complexity = self.metacognition(child_output).get('complexity', torch.tensor(0.5, device=self.device))
        cognitive_loss = nn.MSELoss()(output_complexity, torch.tensor(target_complexity, device=self.device))
        return {
            'moral': moral_loss,
            'attachment': attachment_loss,
            'emotional': emotional_loss,
            'cognitive': cognitive_loss
        }

    def _handle_training_error(self, error_type: str) -> dict:
        last_stable = self.checkpointing.load_last_stable()
        if last_stable:
            print(f"Rolling back to step {last_stable['step']} due to {error_type}")
            return {'error': error_type, 'rollback_step': last_stable['step'], 'status': 'rolled_back'}
        return {'error': error_type, 'status': 'failed'}

    def train_episode(self, num_steps: int = 1000) -> dict:
        episode_stats = []
        early_stopping = EarlyStopping(patience=self.config['early_stopping_patience'])
        for step in range(num_steps):
            mother_stimulus = self.mother.generate_stimulus(self.curriculum.current_stage, self.child.express_feeling())
            step_stats = self.training_step(mother_stimulus['embedding'], mother_stimulus)
            if step_stats.get('error'):
                if step_stats['status'] == 'failed':
                    print(f"Training failed at step {step}: {step_stats['error']}")
                    break
                continue
            episode_stats.append(step_stats)
            if early_stopping.check(step_stats['total_loss']):
                print(f"Early stopping triggered at step {step}")
                break
        return self.monitoring.summarize_episode(episode_stats)
