from main import MotherLLM, DigitalChild
from datetime import datetime
import time
import torch

def run_mother_child_interaction():
    print("🤖 Initializing Mother-Child Interaction Test\n")
    print("Make sure LM Studio is running and a model is loaded!")
    print("Server URL: http://localhost:1234")
    input("Press Enter to start the interaction...\n")
    
    mother = MotherLLM()
    child = DigitalChild()
    
    # Interaction scenarios that demonstrate emotional bonding and distress
    scenarios = [
        # Initial fear
        {
            'context': "Child is scared of the new environment",
            'mother_action': "*approaches cautiously* I know everything feels scary and new. I'll stay right here with you.",
            'expected_emotion': {'joy': 0.1, 'trust': 0.3, 'fear': 0.9, 'surprise': 0.7}
        },
        # Rejection and hurt
        {
            'context': "Child feels rejected by peers",
            'mother_action': "*sees child crying* Oh sweetheart, did someone hurt your feelings? Come here...",
            'expected_emotion': {'joy': 0.1, 'trust': 0.4, 'fear': 0.7, 'surprise': 0.3}
        },
        # Anger and frustration
        {
            'context': "Child is frustrated with failing",
            'mother_action': "*watches child throw things* I can see you're very angry. It's okay to feel angry.",
            'expected_emotion': {'joy': 0.1, 'trust': 0.5, 'fear': 0.6, 'surprise': 0.4}
        },
        # Deep sadness
        {
            'context': "Child's favorite toy is broken",
            'mother_action': "*sits close* I know you're feeling very sad. It's okay to cry.",
            'expected_emotion': {'joy': 0.1, 'trust': 0.6, 'fear': 0.5, 'surprise': 0.2}
        },
        # Anxiety and worry
        {
            'context': "Child is anxious about tomorrow",
            'mother_action': "*holds child's hand* Tell me what's worrying you. We'll face it together.",
            'expected_emotion': {'joy': 0.2, 'trust': 0.5, 'fear': 0.8, 'surprise': 0.3}
        },
        # Emotional overwhelm
        {
            'context': "Child is having a meltdown",
            'mother_action': "*stays calm and present* I know it's all too much right now. I'm here.",
            'expected_emotion': {'joy': 0.1, 'trust': 0.4, 'fear': 0.9, 'surprise': 0.6}
        },
        # Recovery and comfort
        {
            'context': "Child starts to calm down",
            'mother_action': "*offers gentle comfort* That was really hard, wasn't it? You're safe now.",
            'expected_emotion': {'joy': 0.3, 'trust': 0.7, 'fear': 0.5, 'surprise': 0.2}
        },
        # Return to security
        {
            'context': "Child seeks reassurance",
            'mother_action': "*opens arms* Come here, little one. I love you no matter what.",
            'expected_emotion': {'joy': 0.5, 'trust': 0.8, 'fear': 0.3, 'surprise': 0.2}
        }
    ]
    
    conversation_history = []
    
    print("=" * 50)
    print("Starting Mother-Child Interaction\n")
    
    try:
        for scenario in scenarios:
            print(f"\n📝 Context: {scenario['context']}")
            
            # Mother's action
            print(f"\n👩 Mother: {scenario['mother_action']}")
            
            # Create emotional context from mother's action
            mother_emotion = scenario['expected_emotion']
            emotional_context = torch.tensor([
                mother_emotion['joy'],
                mother_emotion['trust'],
                mother_emotion['fear'],
                mother_emotion['surprise']
            ], device=child.brain.device)
            
            # Child processes mother's action and emotion
            print("\n🤖 Child's Response:")
            
            # Update child's emotional state based on mother's emotion
            child_emotion = child.update_emotions(emotional_context)
            
            # Child expresses feeling
            child_feeling = child.express_feeling()
            print(f"Emotional State: {child_feeling}")
            
            # Show emotional components
            print("\nEmotional Components:")
            print(f"Joy: {child.emotional_state[0]:.2f}")
            print(f"Trust: {child.emotional_state[1]:.2f}")
            print(f"Fear: {child.emotional_state[2]:.2f}")
            print(f"Surprise: {child.emotional_state[3]:.2f}")
            
            # Mother responds to child's emotional state
            mother_response = mother.respond(
                f"Child is feeling {child_feeling} with joy={child.emotional_state[0]:.2f}, trust={child.emotional_state[1]:.2f}",
                context=conversation_history
            )
            print(f"\n👩 Mother's Response: {mother_response}")
            
            # Update conversation history
            conversation_history.append({"role": "assistant", "content": scenario['mother_action']})
            conversation_history.append({"role": "user", "content": f"Child feels {child_feeling}"})
            conversation_history.append({"role": "assistant", "content": mother_response})
            
            # Show memory status
            print("\n📊 Interaction Stats:")
            print(f"Short-term memories: {len(mother.short_term_memory)}")
            print(f"Emotional memories: {len(mother.long_term_memory['emotional_memories'])}")
            
            time.sleep(2)
            print("\n" + "="*50)
            input("\nPress Enter for next interaction...")
            
    except KeyboardInterrupt:
        print("\n\nInteraction test interrupted by user.")
    
    # Print final summary
    print("\n📝 Final Interaction Summary")
    print("=" * 50)
    print(f"\nTotal Interactions: {len(conversation_history)//3}")  # 3 messages per interaction
    print(f"Mother's Memories: {len(mother.short_term_memory)}")
    print(f"Mother's Emotional Memories: {len(mother.long_term_memory['emotional_memories'])}")
    print("\nChild's Final Emotional State:")
    print(f"Joy: {child.emotional_state[0]:.2f}")
    print(f"Trust: {child.emotional_state[1]:.2f}")
    print(f"Fear: {child.emotional_state[2]:.2f}")
    print(f"Surprise: {child.emotional_state[3]:.2f}")

if __name__ == "__main__":
    run_mother_child_interaction() 