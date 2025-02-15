from enum import Enum, auto

class DevelopmentalStage(Enum):
    NEWBORN = 0
    EARLY_INFANCY = auto()
    LATE_INFANCY = auto()
    EARLY_TODDLER = auto()
    LATE_TODDLER = auto()
    EARLY_PRESCHOOL = auto()
    LATE_PRESCHOOL = auto()
    EARLY_CHILDHOOD = auto()
    MIDDLE_CHILDHOOD = auto()
    LATE_CHILDHOOD = auto()
    EARLY_ELEMENTARY = auto()
    MIDDLE_ELEMENTARY = auto()
    LATE_ELEMENTARY = auto()
    EARLY_ADOLESCENCE = auto()
    MIDDLE_ADOLESCENCE = auto()
    LATE_ADOLESCENCE = auto()
    YOUNG_ADULT = auto()
    MATURE_ADULT = auto()

class DevelopmentalSystem:
    def __init__(self):
        self.current_stage = DevelopmentalStage.NEWBORN
        self.stage_metrics = {
            'success_rate': 0.0,
            'abstraction': 0.0,
            'self_awareness': 0.0
        }
        self.consecutive_successes = 0
        
    def update_stage(self, metrics):
        # Update internal metrics
        for key, value in metrics.items():
            self.stage_metrics[key] = value
            
        # Check if ready to progress
        if self.can_progress():
            self.consecutive_successes += 1
            if self.consecutive_successes >= 2:  # Need 2 consecutive successful evaluations
                self._advance_stage()
                self.consecutive_successes = 0
        else:
            self.consecutive_successes = 0
            
    def can_progress(self):
        # More lenient progression criteria
        return (
            self.stage_metrics['success_rate'] > config.MIN_STAGE_SUCCESS_RATE or
            (self.stage_metrics['abstraction'] + 
             self.stage_metrics['self_awareness']) / 2 > config.STAGE_PROGRESSION_THRESHOLD
        )
        
    def _advance_stage(self):
        current_value = self.current_stage.value
        if current_value < len(DevelopmentalStage) - 1:
            self.current_stage = DevelopmentalStage(current_value + 1)
            # Reset metrics for new stage
            for key in self.stage_metrics:
                self.stage_metrics[key] = 0.0
            print(f"\nAdvancing to stage: {self.current_stage.name}")
