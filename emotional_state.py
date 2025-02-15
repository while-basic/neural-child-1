from dataclasses import dataclass

@dataclass
class EmotionalState:
    happiness: float  # Previously 'joy'
    sadness: float
    anger: float
    fear: float
    surprise: float = 0.0
    disgust: float = 0.0

    def to_vector(self):
        return [self.happiness, self.sadness, self.anger, self.fear, 
                self.surprise, self.disgust]

    @classmethod
    def from_vector(cls, vector):
        return cls(
            happiness=vector[0],
            sadness=vector[1],
            anger=vector[2],
            fear=vector[3],
            surprise=vector[4] if len(vector) > 4 else 0.0,
            disgust=vector[5] if len(vector) > 5 else 0.0
        )
