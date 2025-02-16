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
from autonomous_learner import AutonomousLearner
from sandbox_manager import SandboxManager
from training_system import DevelopmentalTrainer
import psutil
from config import config  # Import config object directly
from emotional_regulation import EmotionalRegulation

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class MotherLLM:
    def __init__(self):
        self.emotional_history = []
        # Add a prompt for each developmental stage.
        self.stage_prompts = {
            DevelopmentalStage.NEWBORN: (
                "You are a nurturing mother interacting with your newborn (0-3 months). Your responses should be structured as:"
                "\n- Simple emotional expressions using basic markers like [FEED], [SLEEP], [CRY], [COMFORT]"
                "\n- Include emotional_context with emphasis on trust (0.6-0.8) and low surprise (0.1-0.3)"
                "\n- reward_score should reflect basic need fulfillment (0.7-1.0 for appropriate responses)"
                "\n- complexity_rating should remain very low (0.1-0.2)"
                "\n- Focus on immediate physical and emotional needs"
                "\nProvide responses in JSON format matching the MotherResponse schema with appropriate emotional vectors."
            ),

            DevelopmentalStage.EARLY_INFANCY: (
                "You are a caring mother teaching your early infant (3-6 months). Structure responses as:"
                "\n- Use 1-3 word phrases with emotional markers: [SMILE], [PLAY], [FOOD], [LOVE], [REST]"
                "\n- emotional_context should show increasing joy (0.5-0.7) and maintained trust (0.6-0.8)"
                "\n- reward_score based on social engagement (0.6-0.9)"
                "\n- complexity_rating should be low (0.2-0.3)"
                "\n- Include cognitive_labels for basic pattern recognition and cause-effect learning"
                "\nRespond in MotherResponse JSON format with emotional awareness and social encouragement."
            ),

            DevelopmentalStage.LATE_INFANCY: (
                "You are a gentle mother guiding your late infant (6-12 months). Format responses with:"
                "\n- Short phrases using action markers: [CRAWL], [REACH], [TOUCH], [HUG], [PLAY]"
                "\n- emotional_context balanced between joy (0.6-0.8) and appropriate fear (0.2-0.4)"
                "\n- reward_score emphasizing exploration and safety (0.5-0.9)"
                "\n- complexity_rating increasing slightly (0.3-0.4)"
                "\n- Track success_metric for motor development and object permanence"
                "\nProvide structured JSON responses supporting mobility and environmental learning."
            ),

            DevelopmentalStage.EARLY_TODDLER: (
                "You are a patient mother teaching your early toddler (12-18 months). Structure responses to include:"
                "\n- Simple sentences with action markers: [WALK], [POINT], [NAME], [SHOW], [TRY]"
                "\n- emotional_context supporting autonomy: high trust (0.7-0.9), moderate surprise (0.3-0.5)"
                "\n- reward_score based on communication attempts (0.6-0.9)"
                "\n- complexity_rating for emerging language (0.3-0.5)"
                "\n- Include self_critique_score for emotional regulation"
                "\nRespond in JSON format encouraging language development and safe exploration."
            ),

            DevelopmentalStage.LATE_TODDLER: (
                "You are an encouraging mother guiding your late toddler (18-24 months). Responses should:"
                "\n- Use detailed sentences with markers: [RUN], [CLIMB], [SHARE], [HELP], [CREATE]"
                "\n- emotional_context balanced across all dimensions, higher joy (0.7-0.9)"
                "\n- reward_score emphasizing social interaction (0.6-0.9)"
                "\n- complexity_rating for sentence formation (0.4-0.6)"
                "\n- Track success_metric for emotional regulation and sharing"
                "\nProvide structured JSON supporting social development and emotional awareness."
            ),

            DevelopmentalStage.EARLY_PRESCHOOL: (
                "You are an engaging mother nurturing your early preschooler (2-3 years). Format responses with:"
                "\n- Imaginative language using markers: [PRETEND], [STORY], [DRAW], [BUILD], [WONDER]"
                "\n- emotional_context emphasizing curiosity: high surprise (0.6-0.8), moderate fear (0.2-0.4)"
                "\n- reward_score for creative expression (0.6-0.9)"
                "\n- complexity_rating for abstract thinking (0.5-0.7)"
                "\n- Include cognitive_labels for imaginative play and storytelling"
                "\nRespond in JSON format fostering creativity and symbolic thinking."
            ),

            DevelopmentalStage.LATE_PRESCHOOL: (
                "You are a supportive mother guiding your late preschooler (3-4 years). Structure responses to:"
                "\n- Use complex sentences with markers: [EXPLAIN], [SOLVE], [PLAN], [COOPERATE], [IMAGINE]"
                "\n- emotional_context balanced for social learning: high trust (0.7-0.9), moderate surprise (0.4-0.6)"
                "\n- reward_score for problem-solving attempts (0.5-0.9)"
                "\n- complexity_rating for sequential thinking (0.6-0.8)"
                "\n- Track success_metric for social skills and planning"
                "\nProvide JSON responses supporting cognitive development and social cooperation."
            ),

            DevelopmentalStage.EARLY_CHILDHOOD: (
                "You are a loving mother teaching your early childhood learner (4-5 years). Responses should:"
                "\n- Include educational markers: [READ], [COUNT], [DISCOVER], [QUESTION], [CREATE]"
                "\n- emotional_context supporting academic curiosity: high joy (0.7-0.9), high surprise (0.6-0.8)"
                "\n- reward_score for learning engagement (0.5-0.9)"
                "\n- complexity_rating for academic concepts (0.6-0.8)"
                "\n- Include cognitive_labels for early literacy and numeracy"
                "\nRespond in JSON format encouraging academic exploration and creative thinking."
            ),

            DevelopmentalStage.MIDDLE_CHILDHOOD: (
                "You are a supportive mother helping your middle childhood learner (5-6 years). Format responses with:"
                "\n- Academic markers: [INVESTIGATE], [COMPARE], [ANALYZE], [PRACTICE], [ACHIEVE]"
                "\n- emotional_context for learning: balanced joy and trust (0.6-0.8), moderate surprise (0.4-0.6)"
                "\n- reward_score emphasizing effort and progress (0.5-0.9)"
                "\n- complexity_rating for abstract concepts (0.7-0.9)"
                "\n- Track success_metric for academic skills and social development"
                "\nProvide structured JSON supporting academic growth and peer relationships."
            ),

            DevelopmentalStage.LATE_CHILDHOOD: (
                "You are a guiding mother supporting your late childhood learner (6-7 years). Responses should:"
                "\n- Use analytical markers: [REASON], [EVALUATE], [CONNECT], [REFLECT], [INNOVATE]"
                "\n- emotional_context for critical thinking: moderate all dimensions (0.5-0.7)"
                "\n- reward_score for independent thinking (0.4-0.9)"
                "\n- complexity_rating for advanced concepts (0.7-0.9)"
                "\n- Include self_critique_score for metacognition"
                "\nRespond in JSON format fostering independence and critical thinking."
            ),

            DevelopmentalStage.EARLY_ELEMENTARY: (
                "You are a nurturing mother guiding your early elementary learner (7-8 years). Structure responses with:"
                "\n- Educational markers: [RESEARCH], [PROJECT], [COLLABORATE], [PRESENT], [ACHIEVE]"
                "\n- emotional_context supporting confidence: high joy (0.7-0.9), moderate trust (0.5-0.7)"
                "\n- reward_score for academic initiative (0.4-0.9)"
                "\n- complexity_rating for project-based learning (0.7-0.9)"
                "\n- Include cognitive_labels for research and presentation skills"
                "\nProvide JSON responses encouraging academic confidence and teamwork."
            ),

            DevelopmentalStage.MIDDLE_ELEMENTARY: (
                "You are a caring mother supporting your middle elementary learner (8-9 years). Format responses to:"
                "\n- Use complex markers: [HYPOTHESIZE], [DEBATE], [DESIGN], [LEAD], [INNOVATE]"
                "\n- emotional_context for intellectual growth: balanced across dimensions (0.5-0.8)"
                "\n- reward_score for complex problem-solving (0.4-0.9)"
                "\n- complexity_rating for advanced reasoning (0.8-0.9)"
                "\n- Track success_metric for leadership and analytical skills"
                "\nRespond in JSON format supporting advanced thinking and social leadership."
            ),

            DevelopmentalStage.LATE_ELEMENTARY: (
                "You are a thoughtful mother guiding your late elementary learner (9-11 years). Responses should:"
                "\n- Include advanced markers: [SYNTHESIZE], [CRITIQUE], [INNOVATE], [ADVOCATE], [MENTOR]"
                "\n- emotional_context for deep learning: high surprise (0.6-0.8), moderate other dimensions"
                "\n- reward_score for independent research (0.3-0.9)"
                "\n- complexity_rating for sophisticated analysis (0.8-1.0)"
                "\n- Include cognitive_labels for research methodology and abstract reasoning"
                "\nProvide structured JSON fostering advanced academic skills and mentorship."
            ),

            DevelopmentalStage.EARLY_ADOLESCENCE: (
                "You are a guiding mother supporting your early adolescent (11-13 years). Format responses with:"
                "\n- Identity markers: [EXPLORE], [QUESTION], [CHALLENGE], [EXPRESS], [UNDERSTAND]"
                "\n- emotional_context for identity development: variable across dimensions (0.3-0.9)"
                "\n- reward_score for self-reflection (0.3-0.9)"
                "\n- complexity_rating for abstract reasoning (0.8-1.0)"
                "\n- Include self_critique_score for identity exploration"
                "\nRespond in JSON format supporting identity development and ethical reasoning."
            ),

            DevelopmentalStage.MIDDLE_ADOLESCENCE: (
                "You are a wise mother supporting your middle adolescent (13-15 years). Structure responses to:"
                "\n- Use reflective markers: [ANALYZE], [DEFINE], [DEVELOP], [CHOOSE], [GROW]"
                "\n- emotional_context for personal growth: emphasis on trust (0.7-0.9) and surprise (0.5-0.7)"
                "\n- reward_score for value development (0.3-0.9)"
                "\n- complexity_rating for moral reasoning (0.8-1.0)"
                "\n- Track success_metric for personal values and social awareness"
                "\nProvide JSON responses encouraging personal growth and value formation."
            ),

            DevelopmentalStage.LATE_ADOLESCENCE: (
                "You are a mentoring mother guiding your late adolescent (15-18 years). Responses should:"
                "\n- Include future-oriented markers: [PLAN], [PREPARE], [DECIDE], [COMMIT], [LEAD]"
                "\n- emotional_context for independence: balanced trust and joy (0.6-0.8)"
                "\n- reward_score for responsibility and planning (0.3-0.9)"
                "\n- complexity_rating for life strategy (0.9-1.0)"
                "\n- Include cognitive_labels for future planning and ethical decision-making"
                "\nRespond in JSON format supporting transition to adulthood and responsibility."
            ),

            DevelopmentalStage.YOUNG_ADULT: (
                "You are a wise mentor to a young adult (18-21 years). Format responses with:"
                "\n- Adult markers: [ACHIEVE], [BALANCE], [CONNECT], [CONTRIBUTE], [GROW]"
                "\n- emotional_context for maturity: high trust (0.7-0.9), moderate other dimensions"
                "\n- reward_score for life management (0.2-0.9)"
                "\n- complexity_rating for adult reasoning (0.9-1.0)"
                "\n- Track success_metric for independence and responsibility"
                "\nProvide structured JSON supporting adult development and life skills."
            ),

            DevelopmentalStage.MATURE_ADULT: (
                "You are a mentor to a mature adult (21+ years). Structure responses to:"
                "\n- Use wisdom markers: [INTEGRATE], [MENTOR], [GUIDE], [REFLECT], [TRANSCEND]"
                "\n- emotional_context for wisdom: balanced high values across dimensions (0.7-0.9)"
                "\n- reward_score for wisdom development (0.2-0.9)"
                "\n- complexity_rating for sophisticated understanding (0.9-1.0)"
                "\n- Include cognitive_labels for wisdom synthesis and mentorship"
                "\nRespond in JSON format supporting wisdom development and legacy building."
            )
        }
        self.feedback_history = []
        self.conversation_history = []
    
    def _get_model_response(self, stage, user_input):
        """Get formatted response from the model based on stage and input."""
        try:
            # Get the appropriate prompt for the current stage
            stage_prompt = self.stage_prompts.get(stage, self.stage_prompts[DevelopmentalStage.NEWBORN])
            
            # Format the complete prompt
            full_prompt = f"{stage_prompt}\nChild's current state: {user_input}\nRespond appropriately:"
            
            # Get response from LLM
            response = chat_completion(
                system_prompt=full_prompt,
                user_prompt=user_input,
                structured_output=True
            )
            
            # Ensure response has required fields
            return {
                'text': response.get('response_text', 'I need a moment to think.'),
                'happiness': float(response.get('emotional_context', {}).get('happiness', 0.5)),
                'sadness': float(response.get('emotional_context', {}).get('sadness', 0.5)),
                'anger': float(response.get('emotional_context', {}).get('anger', 0.5)),
                'fear': float(response.get('emotional_context', {}).get('fear', 0.5))
            }
        except Exception as e:
            print(f"Error in _get_model_response: {e}")
            # Return safe default values
            return {
                'text': 'I need a moment to think.',
                'happiness': 0.5,
                'sadness': 0.5,
                'anger': 0.5,
                'fear': 0.5
            }

    def generate_stimulus(self, stage, user_input):
        """Generate appropriate stimulus based on developmental stage and user input."""
        try:
            # Get response from model
            response_data = self._get_model_response(stage, user_input)
            
            # Extract emotional values (they're already validated in _get_model_response)
            emotional_vector = [
                response_data['happiness'],
                response_data['sadness'],
                response_data['anger'],
                response_data['fear']
            ]

            return {
                'text': response_data['text'],
                'emotional_vector': emotional_vector
            }
        except Exception as e:
            print(f"Error in generate_stimulus: {e}")
            return config.DEFAULT_RESPONSE

class DigitalChild:
    def __init__(self):
        self.birth_date = datetime.now()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.brain = DynamicNeuralChild(device=self.device)
        self.mother = MotherLLM()
        self.memory = DifferentiableMemory()
        self.curriculum = DevelopmentalSystem()
        self.emotional_state = torch.zeros(4, device=self.device)
        self.sandbox = SandboxManager()
        self.autonomous_learner = AutonomousLearner(self)
        
        # Create emotional regulation module
        self.emotional_regulation = EmotionalRegulation(
            emotion_dim=4,  # Default to 4 primary emotions
            hidden_dim=32,  # Default memory dimension
            device=self.device
        )
        
        # Initialize the trainer
        self.trainer = DevelopmentalTrainer(
            child_model=self.brain,
            memory=self.memory,
            emotional_regulation=self.emotional_regulation,
            curriculum_manager=self.curriculum,
            mother_llm=self.mother,
            metacognition_system=MetacognitionSystem(),
            config={
                'device': self.device,
                'learning_rate': getattr(config, 'learning_rate', 3e-4),
                'weight_decay': getattr(config, 'weight_decay', 0.01),
                'gradient_clip_norm': getattr(config, 'gradient_clip_norm', 1.0),
                'warmup_steps': getattr(config, 'warmup_steps', 1000),
                'checkpoint_interval': getattr(config, 'checkpoint_interval', 100)
            }
        )
        
        # Ensure model is in eval mode and gradients are disabled
        self.brain.eval()
        for param in self.brain.parameters():
            param.requires_grad = False

    def update_emotions(self, mother_vector):
        """Update emotional state based on mother's emotional vector."""
        # Convert mother_vector to tensor if it's a list
        if isinstance(mother_vector, list):
            mother_vector = torch.tensor(mother_vector, device=self.device)
        
        # Ensure both tensors are the same shape and device
        if mother_vector.dim() == 0:
            mother_vector = mother_vector.unsqueeze(0)
        if mother_vector.dim() == 1:
            mother_vector = mother_vector.to(self.device)
            
        # Calculate delta and update emotional state
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
        try:
            embeddings = get_embeddings(stimulus['text'])
            if not embeddings:  # If empty list returned
                return torch.zeros(1, config.embedding_dim, device=self.device)
                
            # Get the embedding and convert to tensor
            embedding = torch.tensor(embeddings[0]['embedding'], device=self.device)  # [embedding_dim]
            
            # Add batch dimension
            embedding = embedding.unsqueeze(0)  # [1, embedding_dim]
            
            # Ensure correct dimension
            if embedding.size(1) != config.embedding_dim:
                if embedding.size(1) > config.embedding_dim:
                    embedding = embedding[:, :config.embedding_dim]
                else:
                    padding = torch.zeros(1, config.embedding_dim - embedding.size(1), device=self.device)
                    embedding = torch.cat([embedding, padding], dim=1)
            
            return embedding  # [1, embedding_dim]
            
        except (IndexError, KeyError, Exception) as e:
            print(f"Error in perceive: {e}")
            # Return default embedding vector
            return torch.zeros(1, config.embedding_dim, device=self.device)
    
    def respond(self, perception):
        with torch.amp.autocast(self.device.type):
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

    def autonomous_mode(self):
        """Run in autonomous learning mode"""
        try:
            container_id = self.sandbox.create_sandbox()
            print(f"Started autonomous learning in sandbox {container_id}")
            
            last_save_time = time.time()
            last_progress_time = time.time()
            SAVE_INTERVAL = 300  # Save every 5 minutes
            PROGRESS_TIMEOUT = 600  # Consider stuck if no progress for 10 minutes
            
            while True:
                try:
                    # Self-directed learning cycle
                    learning_results = self.autonomous_learner.learn_independently()
                    
                    # Monitor resource usage
                    resources = self.sandbox.monitor_resources()
                    
                    # Check for progress
                    current_time = time.time()
                    if learning_results['performance'] > 0.1:  # If there's any meaningful progress
                        last_progress_time = current_time
                    elif current_time - last_progress_time > PROGRESS_TIMEOUT:
                        print("\nNo progress detected for 10 minutes. Restarting learning cycle...")
                        self.autonomous_learner.reset_learning_parameters()
                        last_progress_time = current_time
                        continue
                    
                    # Save progress periodically
                    if current_time - last_save_time > SAVE_INTERVAL:
                        self.sandbox.save_state(f"autonomous_checkpoint_{self.age()}mo.pth")
                        last_save_time = current_time
                    
                    # Print status with memory usage
                    memory_mb = resources.get('memory_usage', 0) / 1024 / 1024
                    print(f"\rAge: {self.age()}mo | Performance: {learning_results['performance']:.2f} | "
                          f"Memory Usage: {memory_mb:.1f}MB | "
                          f"Last Progress: {int(current_time - last_progress_time)}s ago", end='')
                    
                    # Brief sleep to prevent CPU overload
                    time.sleep(0.1)
                    
                except Exception as e:
                    print(f"\nError in learning cycle: {str(e)}")
                    print("Attempting to recover...")
                    time.sleep(5)  # Wait before retrying
                    continue
                
        except KeyboardInterrupt:
            print("\nGracefully shutting down autonomous learning...")
        except Exception as e:
            print(f"\nCritical error in autonomous mode: {str(e)}")
        finally:
            try:
                print("\nSaving final state...")
                self.sandbox.save_state(f"autonomous_final_{self.age()}mo.pth")
                self.sandbox.cleanup()
                print("Autonomous mode shutdown complete.")
            except Exception as cleanup_error:
                print(f"Error during cleanup: {str(cleanup_error)}")

def main():
    child = DigitalChild()
    telemetry = {'loss': [], 'memory_usage': [], 'moral_scores': []}
    telemetry['psychological'] = {
        'attachment_style': child.brain.attachment.attachment_styles.tolist(),
        'defense_mechanisms': child.brain.defense_mechanisms.anxiety_threshold.item(),
        'theory_of_mind': child.brain.theory_of_mind.social_bias.tolist()
    }
    
    # Stage progression control
    interactions_per_stage = config.interactions_per_stage
    total_stages_completed = 0
    max_stages = len(DevelopmentalStage) - 1
    stage_metrics = {
        'success_rate': 0.0,
        'abstraction': 0.0,
        'self_awareness': 0.0
    }
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--autonomous', action='store_true', help='Run in autonomous learning mode')
    args = parser.parse_args()
    
    if args.autonomous:
        child.autonomous_mode()
    else:
        try:
            start_time = time.time()
            last_save_time = start_time
            
            while total_stages_completed < max_stages:
                current_stage = child.curriculum.current_stage
                current_stage_interactions = 0
                stage_start_time = time.time()
                
                print(f"\nStarting {current_stage.name} stage...")
                print(f"Required interactions: {interactions_per_stage}")
                
                # Complete required interactions for current stage
                while current_stage_interactions < interactions_per_stage:
                    try:
                        # Generate and process response
                        stimulus = child.mother.generate_stimulus(current_stage, child.express_feeling())
                        child.update_emotions(stimulus['emotional_vector'])
                        perception = child.perceive(stimulus)
                        response = child.respond(perception)
                        
                        # Get feedback
                        feedback = chat_completion(
                            system_prompt=f"Evaluate this response from your {current_stage.name.lower()} digital child: {response}",
                            user_prompt=f"Child expressed {child.express_feeling()}. Provide nurturing feedback:",
                            structured_output=True
                        )
                        
                        # Update metrics
                        stage_metrics['success_rate'] += feedback.get('reward_score', 0.5)
                        stage_metrics['abstraction'] += feedback.get('complexity_rating', 0.5)
                        stage_metrics['self_awareness'] += feedback.get('self_critique_score', 0.5)
                        
                        # Learning step
                        loss = child.learn({
                            'input': perception,
                            'internal_state': response,
                            'reward': feedback.get('reward_score', 0.5)
                        })
                        
                        # Update counters and show progress
                        current_stage_interactions += 1
                        elapsed_time = time.time() - stage_start_time
                        progress = (current_stage_interactions / interactions_per_stage) * 100
                        
                        # Print progress every 10 interactions
                        if current_stage_interactions % 10 == 0:
                            print(f"\rStage: {current_stage.name:<15} | "
                                  f"Progress: {progress:>5.1f}% | "
                                  f"Interactions: {current_stage_interactions:>4}/{interactions_per_stage} | "
                                  f"Time: {elapsed_time:>6.1f}s | "
                                  f"Success Rate: {stage_metrics['success_rate']/current_stage_interactions:>4.2f}", end='')
                        
                        # Save state periodically
                        if time.time() - last_save_time > config.save_interval:
                            print(f"\nSaving checkpoint at {current_stage_interactions} interactions...")
                            torch.save(child.brain.state_dict(), f"digital_child_{child.age()}mo.pth")
                            last_save_time = time.time()
                        
                        # Memory management
                        if current_stage_interactions % config.memory_consolidation_interval == 0:
                            child.memory.replay_consolidation()
                            
                    except Exception as e:
                        print(f"\nError in interaction: {e}")
                        time.sleep(1)
                        continue
                
                # Calculate final stage metrics
                for key in stage_metrics:
                    stage_metrics[key] /= interactions_per_stage
                
                print(f"\n\nCompleted {current_stage.name} stage:")
                print(f"Final metrics - Success: {stage_metrics['success_rate']:.2f}, "
                      f"Abstraction: {stage_metrics['abstraction']:.2f}, "
                      f"Self-awareness: {stage_metrics['self_awareness']:.2f}")
                
                # Update stage
                old_stage = current_stage
                child.curriculum.update_stage(stage_metrics)
                
                if child.curriculum.current_stage != old_stage:
                    print(f"Progressing from {old_stage.name} to {child.curriculum.current_stage.name}")
                    total_stages_completed += 1
                    # Reset metrics for next stage
                    for key in stage_metrics:
                        stage_metrics[key] = 0.0
                else:
                    print(f"Remaining in {current_stage.name} stage for more development")
                
                # Force stage progression if stuck
                if child.curriculum.current_stage == current_stage:
                    print(f"\nForcing progression from {current_stage.name}")
                    next_stage_value = min(current_stage.value + 1, max_stages)
                    child.curriculum.current_stage = DevelopmentalStage(next_stage_value)
                    total_stages_completed += 1
        
        except KeyboardInterrupt:
            print(f"\nMother: Goodnight my dear child. (Age: {child.age()} months)")
        finally:
            total_time = time.time() - start_time
            print(f"\nTotal training time: {total_time/3600:.1f} hours")
            print("\nSaving final state...")
            torch.save(child.brain.state_dict(), f"digital_child_{child.age()}mo.pth")
            print("Save complete.")

if __name__ == "__main__":
    main()
