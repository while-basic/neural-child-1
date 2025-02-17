# API Reference

Complete documentation of the Neural Child Development System's API.

## Core Modules

### Child Model
- [`child_model.ChildModel`](child_model.md): Main class for the neural child model
- [`developmental_stages.DevelopmentalStage`](developmental_stages.md): Stage management
- [`emotional_regulation.EmotionalRegulator`](emotional_regulation.md): Emotion processing

### Psychological Components
- [`attachment.AttachmentSystem`](attachment.md): Bonding and relationship modeling
- [`theory_of_mind.TheoryOfMind`](theory_of_mind.md): Social understanding
- [`moral_network.MoralNetwork`](moral_network.md): Ethical decision making
- [`defense_mechanisms.DefenseMechanism`](defense_mechanisms.md): Coping strategies

### Learning Systems
- [`training_system.TrainingSystem`](training_system.md): Core training functionality
- [`autonomous_learner.AutonomousLearner`](autonomous_learner.md): Self-directed learning
- [`curriculum_manager.CurriculumManager`](curriculum_manager.md): Learning progression
- [`replay_system.ReplaySystem`](replay_system.md): Experience replay

### Memory and Cognition
- [`memory_module.MemorySystem`](memory_module.md): Memory management
- [`metacognition.MetaCognition`](metacognition.md): Self-awareness
- [`symbol_grounding.SymbolGrounding`](symbol_grounding.md): Concept formation

### Environment and Safety
- [`sandbox_manager.SandboxManager`](sandbox_manager.md): Safe execution environment
- [`llm_module.LLMInterface`](llm_module.md): Language model integration

### Utilities
- [`config.Configuration`](config.md): System configuration
- [`utils.Utilities`](utils.md): Helper functions
- [`schemas.DataSchemas`](schemas.md): Data structures

## Configuration

### Environment Variables
Required environment variables:
```bash
NEURAL_CHILD_ENV=development|production
CUDA_VISIBLE_DEVICES=0  # GPU selection
MODEL_CHECKPOINT_DIR=/path/to/checkpoints
```

### Configuration File
Example `config.yaml` structure:
```yaml
model:
  architecture: transformer
  hidden_size: 768
  num_layers: 12
  
training:
  batch_size: 32
  learning_rate: 0.0001
  
development:
  start_age: 0
  max_age: 72  # months
```

## Data Structures

### State Object
```python
{
    'age': float,  # months
    'stage': DevelopmentalStage,
    'emotional_state': EmotionalState,
    'memory': MemoryState,
    'relationships': dict[str, float]
}
```

### Interaction Format
```python
{
    'input': str,
    'context': dict,
    'timestamp': float,
    'metadata': dict
}
```

## Error Handling

All modules use custom exceptions inheriting from `NeuralChildException`:
- `DevelopmentalError`
- `TrainingError`
- `ConfigurationError`
- `SafetyError`

Example error handling:
```python
try:
    model.process_interaction(input_data)
except DevelopmentalError as e:
    logger.error(f"Development stage error: {e}")
except SafetyError as e:
    logger.critical(f"Safety violation: {e}")
```

## Versioning

The API follows semantic versioning (MAJOR.MINOR.PATCH):
- MAJOR: Incompatible API changes
- MINOR: Backwards-compatible features
- PATCH: Backwards-compatible fixes

## Rate Limits

- Training: 100 requests/minute
- Inference: 1000 requests/minute
- Checkpoint saves: 10/hour

## Further Reading

- [Architecture Overview](../architecture/index.md)
- [Implementation Examples](../examples/index.md)
- [Contributing Guidelines](../guides/contributing.md) 