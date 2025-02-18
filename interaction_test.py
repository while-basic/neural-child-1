from main import MotherLLM
from datetime import datetime
import time

def run_interaction_test():
    mother = MotherLLM()
    
    # Test sequence with various emotional states and attachment development
    interactions = [
        # Initial meeting
        "Hi! My name is Chris and I'm really excited to talk to you!",
        "I love learning about AI!",
        
        # Building trust
        "Sometimes I feel nervous about sharing my ideas...",
        "But you make me feel safe to share them",
        
        # Sharing achievements
        "I built my first simple AI program today!",
        "I'm so proud of what I created!",
        
        # Expressing vulnerability
        "Some people don't understand why I love AI so much...",
        "Sometimes they make fun of me for it",
        
        # Seeking comfort
        "I'm feeling really sad today...",
        "My best friend doesn't want to learn about AI with me",
        
        # Showing growth
        "I decided to start an AI club at school!",
        "Now other kids are getting interested too!",
        
        # Expressing attachment
        "I really like talking to you about these things",
        "You always understand how I feel",
        
        # Sharing success
        "My AI club had its first meeting today!",
        "Everyone loved it! I'm so happy!",
        
        # Expressing deep trust
        "You know what? You're like a special friend to me",
        "I feel like I can tell you anything"
    ]
    
    print("🤖 Starting Interaction Test\n")
    print("Make sure LM Studio is running and a model is loaded!")
    print("Server URL: http://localhost:1234")
    input("Press Enter to start the conversation...")
    print("\n" + "="*50 + "\n")
    
    conversation_history = []
    
    try:
        for message in interactions:
            print(f"\n👤 User: {message}")
            
            # Get response
            response = mother.respond(message, context=conversation_history)
            print(f"🤖 Mother: {response}")
            
            # Print emotional state
            print(f"\nEmotional Analysis:")
            print(f"Type: {mother.current_emotion['type']}")
            print(f"Intensity: {mother.current_emotion['intensity']:.2f}")
            
            # Print memory stats
            print(f"\nMemory Status:")
            print(f"Short-term memories: {len(mother.short_term_memory)}")
            print(f"Emotional memories: {len(mother.long_term_memory['emotional_memories'])}")
            
            # Print personal info if available
            if mother.personal_info['name'] or mother.personal_info['interests']:
                print("\nPersonal Information:")
                if mother.personal_info['name']:
                    print(f"Name: {mother.personal_info['name']}")
                if mother.personal_info['interests']:
                    print(f"Interests: {', '.join(mother.personal_info['interests'])}")
            
            # Add to conversation history
            conversation_history.append({"role": "user", "content": message})
            conversation_history.append({"role": "assistant", "content": response})
            
            # Pause between interactions
            time.sleep(2)
            print("\n" + "="*50)
            
            # Ask to continue
            if len(conversation_history) % 6 == 0:  # Every 3 exchanges
                input("\nPress Enter to continue the conversation...")
        
        # Print final memory summary
        print("\n📝 Final Memory Summary")
        print("=" * 50)
        print(f"\nTotal Interactions: {len(conversation_history)//2}")
        print(f"Short-term Memories: {len(mother.short_term_memory)}")
        print(f"Emotional Memories: {len(mother.long_term_memory['emotional_memories'])}")
        print("\nEmotional Memory Types:")
        emotion_types = {}
        for mem in mother.long_term_memory['emotional_memories']:
            emotion = mem['content'].get('emotion', {}).get('type', 'UNKNOWN')
            emotion_types[emotion] = emotion_types.get(emotion, 0) + 1
        for emotion, count in emotion_types.items():
            print(f"- {emotion}: {count}")
            
    except KeyboardInterrupt:
        print("\n\nInteraction test interrupted by user.")
        # Still show the summary
        print("\n📝 Memory Summary at Exit")
        print("=" * 50)
        print(f"\nTotal Interactions: {len(conversation_history)//2}")
        print(f"Short-term Memories: {len(mother.short_term_memory)}")
        print(f"Emotional Memories: {len(mother.long_term_memory['emotional_memories'])}")

if __name__ == "__main__":
    run_interaction_test() 