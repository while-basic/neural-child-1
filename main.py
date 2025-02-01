# main.py
import torch
import time
from datetime import datetime
from llm_module import chat_completion
from child_model import DynamicNeuralChild
from curriculum_manager import DevelopmentalStage, DevelopmentalSystem
from memory_module import DifferentiableMemory
from moral_network import MoralPolicyNetwork
from metacognition import MetacognitionSystem
from self_supervised_trainer import AutonomousTrainer
from text_embed import get_embeddings

class MotherLLM:
    def __init__(self):
        self.emotional_history = []
        # Add a prompt for each developmental stage.
        self.stage_prompts = {
            DevelopmentalStage.NEWBORN: (
                "You are a nurturing mother to your newborn. "
                "Keep your words extremely simple, using only basic expressions like [FEED], [SLEEP], [CRY]."
            ),
            DevelopmentalStage.EARLY_INFANCY: (
                "You are a caring mother teaching your infant. "
                "Use simple 1-3 word phrases focused on basic needs and feelings, such as [FOOD], [LOVE], [SLEEP], [CRY], [HAPPY]."
            ),
            DevelopmentalStage.LATE_INFANCY: (
                "You are a gentle mother interacting with your older infant. "
                "Use short, simple words to encourage recognition and comfort, for example [SMILE], [PLAY], [SLEEP]."
            ),
            DevelopmentalStage.EARLY_TODDLER: (
                "You are a patient mother teaching your early toddler. "
                "Use simple sentences and clear language to describe everyday objects and actions like [WALK], [EAT], [PLAY]."
            ),
            DevelopmentalStage.LATE_TODDLER: (
                "You are an encouraging mother guiding your older toddler. "
                "Use slightly more detailed sentences that invite exploration and self-expression, such as [RUN], [JUMP], [SMILE]."
            ),
            DevelopmentalStage.EARLY_PRESCHOOL: (
                "You are an engaging mother nurturing your preschooler. "
                "Use playful and imaginative language to spark curiosity and early learning, for example [STORY], [GAME], [ART]."
            ),
            DevelopmentalStage.LATE_PRESCHOOL: (
                "You are a supportive mother guiding your preschooler. "
                "Use friendly, encouraging language that promotes social skills and creative thinking."
            ),
            DevelopmentalStage.EARLY_CHILDHOOD: (
                "You are a loving mother teaching your young child. "
                "Use clear, simple language to introduce basic academic and social concepts."
            ),
            DevelopmentalStage.MIDDLE_CHILDHOOD: (
                "You are a supportive mother helping your child navigate school and social life. "
                "Use language that encourages curiosity and academic growth."
            ),
            DevelopmentalStage.LATE_CHILDHOOD: (
                "You are a guiding mother helping your older child develop critical thinking and independence. "
                "Encourage creativity and problem-solving through your words."
            ),
            DevelopmentalStage.EARLY_ELEMENTARY: (
                "You are a nurturing mother guiding your child as they begin formal education. "
                "Use instructive yet gentle language to build confidence and learning skills."
            ),
            DevelopmentalStage.MIDDLE_ELEMENTARY: (
                "You are a caring mother encouraging your child to explore more complex ideas and social interactions. "
                "Focus on fostering independence and teamwork."
            ),
            DevelopmentalStage.LATE_ELEMENTARY: (
                "You are a thoughtful mother guiding your child towards more independent and analytical thinking. "
                "Encourage curiosity and a love for learning."
            ),
            DevelopmentalStage.EARLY_ADOLESCENCE: (
                "You are a guiding mother helping your early adolescent navigate new challenges. "
                "Ask reflective questions and provide thoughtful insights on identity and ethics."
            ),
            DevelopmentalStage.MIDDLE_ADOLESCENCE: (
                "You are a wise mother supporting your adolescent through the ups and downs of growing up. "
                "Encourage self-reflection and the exploration of personal values."
            ),
            DevelopmentalStage.LATE_ADOLESCENCE: (
                "You are a mentoring mother guiding your older adolescent toward adult responsibilities. "
                "Discuss future plans, ethics, and personal growth with depth and sensitivity."
            ),
            DevelopmentalStage.YOUNG_ADULT: (
                "You are a wise mentor to a young adult. "
                "Engage in complex discussions about life, ethics, and personal development."
            ),
            DevelopmentalStage.MATURE_ADULT: (
                "You are a mentor to a mature adult. "
                "Provide deep insights into life experiences, wisdom, and long-term planning."
            )
        }
        self.feedback_history = []
        self.conversation_history = []
    
    def generate_stimulus(self, stage, child_response="[EMPTY]"):
        self.conversation_history.append({"role": "child", "content": child_response})
        prompt_text = self.stage_prompts.get(stage, self.stage_prompts[DevelopmentalStage.NEWBORN])
        prompt = prompt_text + f" Conversation history: {self.conversation_history}"
        response = chat_completion(
            system_prompt=prompt,
            user_prompt="Provide nurturing feedback and guidance:",
            structured_output=True
        )
        emotional_vector = torch.tensor([
            response.get('joy', 0.5),
            response.get('trust', 0.5),
            response.get('fear', 0.1),
            response.get('surprise', 0.3)
        ], device='cuda')
        if self.emotional_history:
            decay = 0.9 ** len(self.emotional_history)
            emotional_vector += self.emotional_history[-1] * decay
        self.emotional_history.append(emotional_vector)
        self.conversation_history.append({"role": "mother", "content": response['content']})
        self.feedback_history.append(response)
        return {
            'text': response['content'],
            'emotional_vector': torch.sigmoid(emotional_vector)
        }

class DigitalChild:
    def __init__(self):
        self.brain = DynamicNeuralChild()
        self.memory = DifferentiableMemory()
        self.morality = MoralPolicyNetwork(device=self.brain.device)
        self.metacognition = MetacognitionSystem()
        self.curriculum = DevelopmentalSystem()
        self.trainer = AutonomousTrainer(self.brain, self.memory, self.morality)
        self.mother = MotherLLM()
        self.birth_date = datetime.now()
        self.emotional_state = torch.zeros(4, device='cuda')
        
    def update_emotions(self, mother_vector):
        delta = mother_vector - self.emotional_state
        self.emotional_state += 0.3 * delta + 0.1 * torch.randn_like(delta)
        self.emotional_state = torch.clamp(self.emotional_state, 0, 1)
        
    def express_feeling(self):
        joy = self.emotional_state[0].item()
        trust = self.emotional_state[1].item()
        fear = self.emotional_state[2].item()
        surprise = self.emotional_state[3].item()

        feelings = []

        # Evaluate the level of joy
        if joy >= 0.8:
            feelings.append("happy")
        elif joy >= 0.6:
            feelings.append("content")

        # Evaluate the level of trust
        if trust >= 0.8:
            feelings.append("affectionate")
        elif trust >= 0.6:
            feelings.append("warm")

        # Evaluate the level of fear
        if fear >= 0.7:
            feelings.append("terrified")
        elif fear >= 0.5:
            feelings.append("anxious")

        # Evaluate the level of surprise
        if surprise >= 0.7:
            feelings.append("startled")
        elif surprise >= 0.5:
            feelings.append("curious")

        # If no strong emotion is detected, default to neutral
        if not feelings:
            feelings.append("neutral")

        # Combine the identified feelings into a composite string
        return "[" + " & ".join(feelings).upper() + "]"

    def perceive(self, stimulus):
        return torch.tensor(
            get_embeddings(stimulus['text'])[0]['embedding'], 
            device='cuda'
        ).unsqueeze(0)
        
    def respond(self, perception):
        with torch.amp.autocast("cuda"):
            return self.brain(perception)
    
    def learn(self, mother_feedback):
        self.memory.record_experience(
            mother_feedback['input'],
            mother_feedback['internal_state'],
            mother_feedback['reward'],
            time.time(),
            self.emotional_state  
        )
        return self.trainer.training_step(mother_feedback['input'])

    def age(self):
        return (datetime.now() - self.birth_date).days // 30  # months

def main():
    child = DigitalChild()
    telemetry = {'loss': [], 'memory_usage': [], 'moral_scores': []}
    telemetry['psychological'] = {
        'attachment_style': child.brain.attachment.attachment_styles.tolist(),
        'defense_mechanisms': child.brain.defense_mechanisms.anxiety_threshold.item(),
        'theory_of_mind': child.brain.theory_of_mind.social_bias.tolist()
    }
    
    try:
        while True:
            # Use the child's current stage from its curriculum.
            stimulus = child.mother.generate_stimulus(child.curriculum.current_stage, child.express_feeling())
            child.update_emotions(stimulus['emotional_vector'])
            perception = child.perceive(stimulus)
            response = child.respond(perception)
            
            feedback = chat_completion(
                system_prompt=f"Evaluate this response from your digital child: {response}",
                user_prompt=f"Child expressed {child.express_feeling()}. Provide nurturing feedback:",
                structured_output=True
            )
            
            # Here's the fix! Just append loss directly since it's already a float
            loss = child.learn({
                'input': perception,
                'internal_state': response,
                'reward': feedback['reward_score']
            })
            
            telemetry['loss'].append(loss)  # Removed .item() call
            telemetry['memory_usage'].append(torch.cuda.memory_allocated())
            telemetry['moral_scores'].append(feedback['reward_score'])
            
            child.curriculum.update_stage({
                'success_rate': feedback['success_metric'],
                'abstraction': feedback['complexity_rating'],
                'self_awareness': feedback['self_critique_score']
            })
            
            # Periodic memory consolidation
            if time.time() % 86400 < 3600:
                child.memory.replay_consolidation()
            if torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated() > 0.9:
                child.memory.replay_consolidation(batch_size=16)
                
    except KeyboardInterrupt:
        print(f"\nMother: Goodnight my dear child. (Age: {child.age()} months)")
        torch.save(child.brain.state_dict(), f"digital_child_{child.age()}mo.pth")

if __name__ == "__main__":
    main()