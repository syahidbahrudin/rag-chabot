import re
from typing import Tuple, Optional


# Common greeting patterns (case-insensitive)
GREETING_PATTERNS = [
    r'^(hi|hello|hey|greetings|howdy|hiya|sup|yo)[\s!.,]*$',
    r'^(good\s+(morning|afternoon|evening|day|night))[\s!.,]*$',
    r'^(what\'?s\s+up|whats\s+up|wassup|how\s+(are\s+you|do\s+you\s+do|goes\s+it))[\s?.,]*$',
    r'^(nice\s+to\s+meet\s+you|pleased\s+to\s+meet\s+you)[\s!.,]*$',
]


def is_greeting(text: str) -> bool:
    """
    Check if the input text is a greeting.
    
    Args:
        text: The user input text to check
        
    Returns:
        True if the text is identified as a greeting, False otherwise
    """
    if not text or not text.strip():
        return False
    
    # Normalize text: lowercase and strip whitespace/punctuation
    normalized = text.strip().lower()
    
    # Check against greeting patterns
    for pattern in GREETING_PATTERNS:
        if re.match(pattern, normalized, re.IGNORECASE):
            return True
    
    # Check for very short messages that are just greetings
    words = normalized.split()
    if len(words) <= 3:
        # Common single-word greetings
        single_word_greetings = ['hi', 'hello', 'hey', 'hiya', 'howdy', 'sup', 'yo', 'hola', 'ciao']
        if words[0] in single_word_greetings:
            return True
    
    return False


def get_greeting_response() -> str:
    """
    Get a friendly response for greeting messages.
    
    Returns:
        A friendly greeting response that encourages the user to ask policy questions
    """
    return """Hello! 👋 I'm your company policy and product assistant. I'm here to help you find information about:

- Company policies
- Product details
- Shipping information
- Return policies
- And much more!

Feel free to ask me any questions about our policies or products. How can I help you today?"""
