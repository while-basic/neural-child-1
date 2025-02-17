from dataclasses import dataclass
from typing import List, Dict, Tuple

@dataclass
class EmotionalState:
    """Represents a complex emotional state with multiple dimensions and derived emotions.
    
    This class handles both primary and secondary emotions, allowing for more nuanced
    emotional expression and analysis.
    """
    # Primary emotions (based on Plutchik's wheel)
    happiness: float  # joy/ecstasy
    sadness: float
    anger: float
    fear: float
    surprise: float = 0.0
    disgust: float = 0.0
    trust: float = 0.5
    anticipation: float = 0.5

    def to_vector(self) -> List[float]:
        """Convert emotional state to a vector representation."""
        return [
            self.happiness, self.sadness, self.anger, self.fear,
            self.surprise, self.disgust, self.trust, self.anticipation
        ]

    @classmethod
    def from_vector(cls, vector: List[float]) -> 'EmotionalState':
        """Create an EmotionalState instance from a vector."""
        return cls(
            happiness=vector[0],
            sadness=vector[1],
            anger=vector[2],
            fear=vector[3],
            surprise=vector[4] if len(vector) > 4 else 0.0,
            disgust=vector[5] if len(vector) > 5 else 0.0,
            trust=vector[6] if len(vector) > 6 else 0.5,
            anticipation=vector[7] if len(vector) > 7 else 0.5
        )

    def get_complex_emotions(self) -> Dict[str, float]:
        """Calculate complex emotions based on combinations of primary emotions.
        
        Returns:
            Dictionary mapping complex emotion names to their intensities (0-1).
        """
        return {
            'love': min(1.0, (self.happiness + self.trust) / 2),
            'submission': min(1.0, (self.trust + self.fear) / 2),
            'awe': min(1.0, (self.fear + self.surprise) / 2),
            'disappointment': min(1.0, (self.surprise + self.sadness) / 2),
            'remorse': min(1.0, (self.sadness + self.disgust) / 2),
            'contempt': min(1.0, (self.disgust + self.anger) / 2),
            'aggressiveness': min(1.0, (self.anger + self.anticipation) / 2),
            'optimism': min(1.0, (self.anticipation + self.happiness) / 2),
            'guilt': min(1.0, (self.fear + self.sadness) / 2),
            'curiosity': min(1.0, (self.anticipation + self.trust) / 2),
            'pride': min(1.0, (self.happiness + self.anticipation) / 2),
            'shame': min(1.0, (self.sadness + self.fear) / 2),
            'anxiety': min(1.0, (self.fear + self.anticipation) / 2),
            'contentment': min(1.0, (self.happiness + self.trust) / 2)
        }

    def get_dominant_emotions(self, threshold: float = 0.5) -> List[Tuple[str, float]]:
        """Get the dominant emotions above a certain threshold.
        
        Args:
            threshold: Minimum intensity for an emotion to be considered dominant.
            
        Returns:
            List of (emotion_name, intensity) tuples, sorted by intensity.
        """
        primary = {
            'happiness': self.happiness,
            'sadness': self.sadness,
            'anger': self.anger,
            'fear': self.fear,
            'surprise': self.surprise,
            'disgust': self.disgust,
            'trust': self.trust,
            'anticipation': self.anticipation
        }
        
        complex = self.get_complex_emotions()
        all_emotions = {**primary, **complex}
        
        dominant = [
            (emotion, intensity) 
            for emotion, intensity in all_emotions.items() 
            if intensity >= threshold
        ]
        return sorted(dominant, key=lambda x: x[1], reverse=True)

    def get_emotional_description(self) -> str:
        """Generate a natural language description of the emotional state.
        
        Returns:
            A string describing the current emotional state in natural language.
        """
        dominant = self.get_dominant_emotions(threshold=0.6)
        if not dominant:
            return "feeling neutral"
            
        if len(dominant) == 1:
            emotion, intensity = dominant[0]
            intensity_word = "slightly " if intensity < 0.7 else "very " if intensity > 0.8 else ""
            return f"feeling {intensity_word}{emotion}"
            
        emotions = [emotion for emotion, _ in dominant[:2]]
        return f"feeling a mix of {emotions[0]} and {emotions[1]}"
