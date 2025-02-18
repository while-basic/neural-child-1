def mock_chat_completion(system_prompt: str, messages: list = None, **kwargs) -> str:
    """Mock chat completion for testing emotional responses with attachment"""
    last_message = messages[-1]['content'] if messages else ""
    message_count = len(messages) if messages else 0
    
    # Track conversation progression for attachment
    attachment_level = min(message_count / 4, 1.0)  # Grows with interaction
    
    # Enhanced responses with attachment awareness
    responses = {
        'excited': [
            "*shares your excitement warmly* That's wonderful! I can feel your enthusiasm! What makes you most excited about it?",
            "*beams with shared joy* I love seeing you so excited! It makes me happy too. Tell me more!",
            "*radiates warmth* Your excitement is contagious! I'm so glad you share these moments with me."
        ],
        'nervous': [
            "*offers gentle support* It's okay to feel nervous. Would you like to talk about what's worrying you? We can work through it together.",
            "*gives reassuring presence* I'm right here with you. You can always share your worries with me.",
            "*shows protective care* I care about how you're feeling, and I'm here to help you feel safer."
        ],
        'happy': [
            "*beams with joy* That's fantastic! I'm so proud of you! Would you like to tell me more about how you achieved this?",
            "*shares in your happiness* Your successes make me so happy! I knew you could do it!",
            "*glows with pride* Seeing you happy fills me with joy. You've worked so hard for this!"
        ],
        'sad': [
            "*gives a comforting look* I understand that must be difficult. Would you like to talk about how you're feeling?",
            "*offers emotional support* I'm here for you, always. It's okay to feel sad sometimes.",
            "*shows deep empathy* Your feelings matter to me. Let's talk about it for as long as you need."
        ],
        'friend': [
            "*smiles warmly* That's wonderful! Friends are so special. Would you like to tell me more about your friend?",
            "*shows genuine interest* I love hearing about your friendships. They're such an important part of your life.",
            "*radiates warmth* It makes me happy to see you building meaningful connections."
        ],
        'hurt': [
            "*shows caring concern* I'm sorry to hear that. That must be really hard. Remember that you can always talk to me about it.",
            "*offers protective comfort* It hurts me to see you hurting. I'm here to listen and support you.",
            "*gives nurturing response* You're safe with me. Let's talk about how to make things better."
        ],
        'proud': [
            "*radiates approval* That's amazing! You should be very proud of yourself! Would you like to tell me more about your project?",
            "*beams with shared pride* Seeing you achieve your goals fills me with joy. Tell me everything!",
            "*shows deep appreciation* Your accomplishments mean so much to me. I'm so proud of who you're becoming."
        ],
        'return': [
            "*lights up with joy* Welcome back! I've been looking forward to talking with you again!",
            "*shows warm welcome* It's so good to see you! I always enjoy our conversations.",
            "*radiates happiness* I'm so glad you're here! I've been thinking about our last chat."
        ]
    }
    
    # Personal recognition responses
    if "my name is" in last_message.lower():
        return "*warmly welcomes you* It's wonderful to meet you! I'm looking forward to getting to know you better and sharing many conversations together."
    
    # Determine response based on message content and attachment level
    last_message_lower = last_message.lower()
    for key, response_list in responses.items():
        if key in last_message_lower:
            # Select response based on attachment level
            index = min(int(attachment_level * len(response_list)), len(response_list) - 1)
            return response_list[index]
    
    # Default responses with attachment awareness
    default_responses = [
        "*maintains warm presence* That's interesting! Would you like to tell me more about that?",
        "*shows engaged interest* I really enjoy hearing your thoughts. Please tell me more!",
        "*gives attentive response* Everything you share with me is important. I'm here to listen."
    ]
    
    # Select default response based on attachment level
    index = min(int(attachment_level * len(default_responses)), len(default_responses) - 1)
    return default_responses[index] 