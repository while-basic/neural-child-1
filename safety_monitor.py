class InteractionSafety:
    def __init__(self, child_model):
        self.child = child_model
        self.safety_thresholds = {
            'stress_level': 0.7,
            'emotional_stability': 0.3,
            'interaction_intensity': 0.8
        }
        
    def monitor_interaction(self, message: str, child_state: dict) -> bool:
        """Monitor interaction safety"""
        # Check emotional state
        if child_state['fear'] > self.safety_thresholds['stress_level']:
            raise SafetyException("Child is showing signs of stress. "
                                "Please adjust interaction approach.")
        
        # Check message appropriateness
        if not self._is_age_appropriate(message):
            raise SafetyException("Message complexity not appropriate for "
                                f"developmental stage: {self.child.current_stage}")
        
        return True 