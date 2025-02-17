# Child Model API Reference

## ChildModel

```python
class ChildModel:
    """
    Core class for the Neural Child Development System.
    
    This class implements the main functionality for simulating child development
    through neural networks and psychological modeling.
    
    Attributes:
        age (float): Current age in months
        stage (DevelopmentalStage): Current developmental stage
        emotional_state (EmotionalState): Current emotional state
        memory (MemorySystem): Memory management system
        config (Configuration): Model configuration
    """
```

## Constructor

### `__init__`

```python
def __init__(
    self,
    config: Configuration,
    initial_age: Optional[float] = 0.0,
    checkpoint_path: Optional[str] = None
) -> None:
    """
    Initialize a new Child Model instance.
    
    Args:
        config (Configuration): Configuration object containing model parameters
        initial_age (float, optional): Starting age in months. Defaults to 0.0
        checkpoint_path (str, optional): Path to load initial state from
        
    Raises:
        ConfigurationError: If configuration is invalid
        CheckpointError: If checkpoint loading fails
    """
```

## Core Methods

### `process_interaction`

```python
def process_interaction(
    self,
    input: str,
    context: Optional[Dict[str, Any]] = None,
    timeout: Optional[float] = None
) -> InteractionResponse:
    """
    Process an interaction with the child model.
    
    Args:
        input (str): The input interaction text
        context (Dict[str, Any], optional): Additional context for the interaction
        timeout (float, optional): Maximum processing time in seconds
        
    Returns:
        InteractionResponse: Model's response including emotional state
        
    Raises:
        TimeoutError: If processing exceeds timeout
        SafetyError: If input violates safety constraints
    """
```

### `update_state`

```python
def update_state(self) -> None:
    """
    Update the model's internal state.
    
    This includes age progression, developmental stage updates,
    and psychological component updates.
    
    Raises:
        StateError: If state update fails
    """
```

### `save_checkpoint`

```python
def save_checkpoint(
    self,
    path: str,
    metadata: Optional[Dict[str, Any]] = None
) -> str:
    """
    Save the current model state to a checkpoint.
    
    Args:
        path (str): Path to save the checkpoint
        metadata (Dict[str, Any], optional): Additional metadata to save
        
    Returns:
        str: Path to the saved checkpoint
        
    Raises:
        IOError: If saving fails
    """
```

### `load_checkpoint`

```python
def load_checkpoint(
    self,
    path: str,
    validate: bool = True
) -> None:
    """
    Load model state from a checkpoint.
    
    Args:
        path (str): Path to the checkpoint
        validate (bool): Whether to validate state after loading
        
    Raises:
        CheckpointError: If loading or validation fails
    """
```

## State Management

### `get_state`

```python
def get_state(self) -> ModelState:
    """
    Get the current model state.
    
    Returns:
        ModelState: Current state including age, stage, and emotional state
    """
```

### `set_state`

```python
def set_state(
    self,
    state: ModelState,
    validate: bool = True
) -> None:
    """
    Set the model state.
    
    Args:
        state (ModelState): State to set
        validate (bool): Whether to validate the state
        
    Raises:
        StateError: If state is invalid
    """
```

## Component Management

### `register_component`

```python
def register_component(
    self,
    name: str,
    component: BaseComponent
) -> None:
    """
    Register a psychological component.
    
    Args:
        name (str): Component name
        component (BaseComponent): Component instance
        
    Raises:
        ComponentError: If registration fails
    """
```

### `get_component`

```python
def get_component(
    self,
    name: str
) -> BaseComponent:
    """
    Get a registered component.
    
    Args:
        name (str): Component name
        
    Returns:
        BaseComponent: The requested component
        
    Raises:
        ComponentError: If component not found
    """
```

## Learning Management

### `configure_learning`

```python
def configure_learning(
    self,
    learning_rate: float = 0.001,
    batch_size: int = 32,
    optimizer: str = 'adam'
) -> None:
    """
    Configure learning parameters.
    
    Args:
        learning_rate (float): Learning rate
        batch_size (int): Training batch size
        optimizer (str): Optimizer type
        
    Raises:
        ConfigurationError: If parameters are invalid
    """
```

### `train`

```python
def train(
    self,
    data: Dataset,
    epochs: int = 1,
    validation_data: Optional[Dataset] = None,
    callbacks: Optional[List[Callback]] = None
) -> TrainingHistory:
    """
    Train the model.
    
    Args:
        data (Dataset): Training data
        epochs (int): Number of training epochs
        validation_data (Dataset, optional): Validation data
        callbacks (List[Callback], optional): Training callbacks
        
    Returns:
        TrainingHistory: Training metrics history
        
    Raises:
        TrainingError: If training fails
    """
```

## Memory Management

### `configure_memory`

```python
def configure_memory(
    self,
    max_memory_size: int = 8000,
    cleanup_threshold: float = 0.9,
    cache_size: int = 1000
) -> None:
    """
    Configure memory management parameters.
    
    Args:
        max_memory_size (int): Maximum memory usage in MB
        cleanup_threshold (float): Memory cleanup threshold
        cache_size (int): Response cache size
    """
```

### `cleanup_memory`

```python
def cleanup_memory(self) -> None:
    """
    Clean up unused memory.
    
    Raises:
        MemoryError: If cleanup fails
    """
```

## Monitoring

### `get_metrics`

```python
def get_metrics(self) -> Dict[str, Any]:
    """
    Get performance metrics.
    
    Returns:
        Dict[str, Any]: Dictionary of metrics
    """
```

### `get_development_progress`

```python
def get_development_progress(self) -> DevelopmentProgress:
    """
    Get development progress information.
    
    Returns:
        DevelopmentProgress: Current development progress
    """
```

## Events and Callbacks

### `add_event_handler`

```python
def add_event_handler(
    self,
    event: str,
    handler: Callable
) -> None:
    """
    Add an event handler.
    
    Args:
        event (str): Event name
        handler (Callable): Event handler function
    """
```

### `remove_event_handler`

```python
def remove_event_handler(
    self,
    event: str,
    handler: Callable
) -> None:
    """
    Remove an event handler.
    
    Args:
        event (str): Event name
        handler (Callable): Event handler function
        
    Raises:
        ValueError: If handler not found
    """
```

## Data Types

### ModelState

```python
@dataclass
class ModelState:
    """Model state data structure."""
    age: float
    stage: DevelopmentalStage
    emotional_state: EmotionalState
    memory: MemoryState
    relationships: Dict[str, float]
```

### InteractionResponse

```python
@dataclass
class InteractionResponse:
    """Interaction response data structure."""
    text: str
    emotional_state: EmotionalState
    confidence: float
    metadata: Dict[str, Any]
```

### DevelopmentProgress

```python
@dataclass
class DevelopmentProgress:
    """Development progress data structure."""
    stage: DevelopmentalStage
    age: float
    milestones: List[str]
    next_milestone: str
    progress_metrics: Dict[str, float]
```

## Exceptions

```python
class ModelError(Exception):
    """Base exception for model errors."""
    pass

class StateError(ModelError):
    """State-related errors."""
    pass

class ComponentError(ModelError):
    """Component-related errors."""
    pass

class CheckpointError(ModelError):
    """Checkpoint-related errors."""
    pass

class SafetyError(ModelError):
    """Safety-related errors."""
    pass
```

## Usage Examples

### Basic Interaction

```python
# Initialize model
config = Configuration.from_yaml('config.yaml')
model = ChildModel(config, initial_age=6)

# Process interaction
response = model.process_interaction(
    "Hello! Let's play with blocks!",
    context={'activity': 'play', 'objects': ['blocks']}
)

print(f"Response: {response.text}")
print(f"Emotional state: {response.emotional_state}")
```

### State Management

```python
# Save state
model.save_checkpoint(
    'checkpoints/milestone_1.pt',
    metadata={'milestone': 'first_words'}
)

# Load state
model.load_checkpoint('checkpoints/milestone_1.pt')

# Get current state
state = model.get_state()
print(f"Age: {state.age} months")
print(f"Stage: {state.stage}")
```

### Component Integration

```python
# Register custom component
class CustomComponent(BaseComponent):
    def process(self, input_data):
        return processed_result

model.register_component('custom', CustomComponent())

# Use component
component = model.get_component('custom')
result = component.process(data)
```

## See Also

- [Child Model Component](../components/child-model.md)
- [Architecture Overview](../architecture/index.md)
- [Examples](../examples/basic-usage.md)
- [Development Guide](../guides/development.md) 