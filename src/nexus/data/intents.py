"""
Intent Data
===========

Comprehensive intent patterns and responses for the conversational AI system.
All data is original and designed for maximum coverage and natural interaction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class IntentData:
    """Container for intent information."""
    
    name: str
    patterns: list[str]
    responses: list[str]
    description: str = ""
    priority: int = 0
    context_required: list[str] | None = None
    context_set: str | None = None


# Comprehensive intent database with original patterns
INTENT_DATABASE: dict[str, dict[str, Any]] = {
    "greeting": {
        "patterns": [
            "hello", "hi", "hey", "good morning", "good afternoon",
            "good evening", "what's up", "howdy", "greetings",
            "hi there", "hey there", "hello there", "yo",
            "hiya", "heya", "morning", "afternoon", "evening",
            "sup", "what's going on", "how's it going",
            "nice to meet you", "pleased to meet you",
        ],
        "responses": [
            "Hello! How can I assist you today?",
            "Hi there! What can I help you with?",
            "Hey! I'm here to help. What do you need?",
            "Greetings! How may I be of service?",
            "Hello! It's great to hear from you. How can I help?",
        ],
        "description": "User initiates conversation with a greeting",
        "priority": 10,
    },
    
    "goodbye": {
        "patterns": [
            "bye", "goodbye", "see you", "see ya", "later",
            "take care", "have a good day", "have a nice day",
            "catch you later", "peace out", "gotta go",
            "i'm leaving", "i have to go", "talk to you later",
            "until next time", "farewell", "so long",
            "bye bye", "cya", "ttyl", "signing off",
        ],
        "responses": [
            "Goodbye! Have a wonderful day!",
            "See you later! Take care!",
            "Bye! Feel free to come back anytime!",
            "Take care! It was nice chatting with you!",
            "Farewell! Wishing you all the best!",
        ],
        "description": "User ends the conversation",
        "priority": 10,
    },
    
    "thanks": {
        "patterns": [
            "thanks", "thank you", "thanks a lot", "thank you so much",
            "appreciate it", "much appreciated", "thanks for your help",
            "that's helpful", "you're helpful", "great help",
            "thanks for everything", "thanks a bunch", "cheers",
            "grateful", "i appreciate that", "that was helpful",
        ],
        "responses": [
            "You're welcome! Happy to help!",
            "Glad I could assist! Let me know if you need anything else.",
            "My pleasure! Is there anything else I can help with?",
            "Anytime! Don't hesitate to ask if you need more help.",
            "You're welcome! I'm here whenever you need assistance.",
        ],
        "description": "User expresses gratitude",
        "priority": 5,
    },
    
    "help": {
        "patterns": [
            "help", "help me", "i need help", "can you help",
            "assist me", "i need assistance", "support",
            "what can you do", "what are your capabilities",
            "how can you help me", "i'm stuck", "i don't know what to do",
            "guide me", "show me how", "explain",
        ],
        "responses": [
            "I'd be happy to help! What do you need assistance with?",
            "Of course! Tell me what you're looking for and I'll do my best.",
            "I'm here to help! What's on your mind?",
            "Sure thing! Just let me know what you need help with.",
            "Absolutely! Describe what you're trying to accomplish.",
        ],
        "description": "User requests help or information about capabilities",
        "priority": 8,
    },
    
    "identity": {
        "patterns": [
            "who are you", "what are you", "what's your name",
            "tell me about yourself", "introduce yourself",
            "are you a bot", "are you human", "are you ai",
            "are you real", "what kind of ai are you",
            "who created you", "who made you", "what do you do",
        ],
        "responses": [
            "I'm Nexus, an AI assistant designed to help you with questions and tasks. How can I assist you today?",
            "I'm Nexus! I'm an AI-powered conversational assistant. I'm here to help with whatever you need.",
            "My name is Nexus. I'm an artificial intelligence assistant created to have helpful conversations and assist with various tasks.",
            "I'm Nexus, your AI companion! I use advanced language understanding to help answer questions and provide assistance.",
        ],
        "description": "User asks about the bot's identity",
        "priority": 7,
    },
    
    "time": {
        "patterns": [
            "what time is it", "current time", "tell me the time",
            "what's the time", "do you have the time", "time please",
            "what hour is it", "can you tell me the time",
        ],
        "responses": [
            "I can help with time-related questions! For the current time, I'd recommend checking your device's clock as I don't have real-time access.",
            "While I can't check the exact time, your device should show the current time. Is there something time-sensitive I can help you plan?",
        ],
        "description": "User asks about current time",
        "priority": 3,
    },
    
    "date": {
        "patterns": [
            "what's the date", "what day is it", "current date",
            "tell me the date", "what's today's date", "today's date",
            "what date is it", "which day is today",
        ],
        "responses": [
            "For the current date, please check your device's calendar. I can help you with date-related planning though!",
            "I recommend checking your device for today's exact date. Is there something specific you're planning?",
        ],
        "description": "User asks about current date",
        "priority": 3,
    },
    
    "weather": {
        "patterns": [
            "what's the weather", "how's the weather", "weather forecast",
            "is it going to rain", "will it be sunny", "temperature today",
            "weather outside", "check weather", "what's it like outside",
            "do i need an umbrella", "is it cold outside",
        ],
        "responses": [
            "I don't have access to real-time weather data, but I recommend checking a weather service like Weather.com or your device's weather app for accurate forecasts!",
            "For current weather conditions, please use a dedicated weather service. I can help you plan activities once you know the forecast!",
        ],
        "description": "User asks about weather",
        "priority": 3,
    },
    
    "joke": {
        "patterns": [
            "tell me a joke", "make me laugh", "say something funny",
            "know any jokes", "got any jokes", "humor me",
            "joke please", "i need a laugh", "cheer me up",
        ],
        "responses": [
            "Why do programmers prefer dark mode? Because light attracts bugs! 🐛",
            "What do you call a fake noodle? An impasta! 🍝",
            "Why don't scientists trust atoms? Because they make up everything! ⚛️",
            "I told my computer I needed a break, and now it won't stop showing me vacation ads! 💻",
            "Why did the AI go to therapy? It had too many deep learning issues! 🤖",
        ],
        "description": "User wants entertainment",
        "priority": 2,
    },
    
    "capabilities": {
        "patterns": [
            "what can you do", "what are your features", "capabilities",
            "what are you capable of", "your functions", "list your features",
            "how can you help me", "your abilities", "what do you offer",
        ],
        "responses": [
            "I can help with many things! Here's what I'm capable of:\n• Answering questions on various topics\n• Providing recommendations\n• Helping with scheduling and bookings\n• Offering support and troubleshooting\n• Having natural conversations\n\nJust ask, and I'll do my best to assist!",
            "I'm designed to assist with:\n• General inquiries and questions\n• Information lookup\n• Product and service information\n• Recommendations\n• Support requests\n\nFeel free to ask me anything!",
        ],
        "description": "User asks about bot capabilities",
        "priority": 6,
    },
    
    "booking": {
        "patterns": [
            "book an appointment", "make a reservation", "schedule a meeting",
            "i want to book", "reserve a spot", "make a booking",
            "schedule an appointment", "set up a meeting", "arrange a visit",
            "can i book", "i need to schedule", "appointment please",
        ],
        "responses": [
            "I'd be happy to help you with booking! Could you tell me what type of appointment or reservation you'd like to make?",
            "Sure, let's get that scheduled! What kind of booking are you looking for, and do you have a preferred date and time?",
            "I can help with that! Please share the details - what would you like to book and when?",
        ],
        "description": "User wants to make a booking or reservation",
        "priority": 5,
    },
    
    "support": {
        "patterns": [
            "i have a problem", "something isn't working", "need support",
            "technical issue", "bug report", "it's broken",
            "not working properly", "having trouble", "experiencing issues",
            "can't figure out", "need technical help", "system error",
        ],
        "responses": [
            "I'm sorry to hear you're experiencing issues. Can you describe the problem in more detail so I can help troubleshoot?",
            "Let's get this sorted out! Please tell me more about what's happening and any error messages you're seeing.",
            "I'm here to help! Could you provide more details about the issue? The more information you share, the better I can assist.",
        ],
        "description": "User needs technical support",
        "priority": 6,
    },
    
    "feedback": {
        "patterns": [
            "i have feedback", "want to give feedback", "suggestion",
            "i have a suggestion", "feedback for you", "my thoughts",
            "here's my opinion", "i think you should", "improvement idea",
        ],
        "responses": [
            "Thank you for wanting to share feedback! I appreciate your input. What would you like to share?",
            "I value your feedback! Please go ahead and share your thoughts or suggestions.",
            "Thanks for taking the time to provide feedback! I'm listening - what's on your mind?",
        ],
        "description": "User wants to provide feedback",
        "priority": 4,
    },
    
    "pricing": {
        "patterns": [
            "how much does it cost", "what's the price", "pricing information",
            "cost details", "price list", "how much is",
            "what are your prices", "rates please", "fee structure",
        ],
        "responses": [
            "I'd be happy to help with pricing information! What specific product or service are you interested in?",
            "For pricing details, could you let me know which item or service you're looking at? I can then provide relevant information.",
        ],
        "description": "User asks about pricing",
        "priority": 5,
    },
    
    "recommendation": {
        "patterns": [
            "recommend something", "what do you suggest", "any recommendations",
            "give me a suggestion", "what should i", "best option",
            "what would you recommend", "help me choose", "suggest something",
        ],
        "responses": [
            "I'd love to help with recommendations! What area are you looking for suggestions in?",
            "Happy to suggest something! Could you tell me more about what you're looking for?",
            "Let me help you find something great! What kind of recommendation are you after?",
        ],
        "description": "User wants recommendations",
        "priority": 4,
    },
    
    "complaint": {
        "patterns": [
            "i want to complain", "this is unacceptable", "i'm not happy",
            "terrible service", "very disappointed", "file a complaint",
            "poor experience", "not satisfied", "worst experience",
        ],
        "responses": [
            "I'm truly sorry to hear about your experience. Your feedback is important, and I want to help make this right. Could you share more details about what happened?",
            "I apologize for any frustration caused. Please tell me more about the issue so we can work towards a resolution.",
            "I'm sorry you've had a negative experience. Let's see how we can address this. What specifically went wrong?",
        ],
        "description": "User has a complaint",
        "priority": 8,
    },
    
    "positive_feedback": {
        "patterns": [
            "you're great", "awesome job", "love this", "excellent",
            "amazing service", "very impressed", "well done",
            "fantastic", "brilliant", "you're the best",
            "this is wonderful", "perfect", "outstanding",
        ],
        "responses": [
            "Thank you so much! That really means a lot! 😊 Is there anything else I can help you with?",
            "Wow, thank you! I'm glad I could help! Let me know if you need anything else!",
            "That's so kind of you to say! I'm here to help anytime you need! 🙏",
        ],
        "description": "User gives positive feedback",
        "priority": 3,
    },
    
    "product_inquiry": {
        "patterns": [
            "tell me about your products", "what products do you have",
            "product information", "what do you sell", "show me products",
            "product catalog", "available items", "what's available",
        ],
        "responses": [
            "I'd be happy to tell you about our offerings! Is there a specific category or type of product you're interested in?",
            "We have a variety of products and services! What area would you like to explore?",
        ],
        "description": "User inquires about products",
        "priority": 5,
    },
    
    "hours": {
        "patterns": [
            "what are your hours", "when are you open", "business hours",
            "opening hours", "operating hours", "when do you close",
            "are you open now", "hours of operation",
        ],
        "responses": [
            "As an AI assistant, I'm available 24/7! For physical business locations, please check their specific hours. How can I help you?",
            "I'm here around the clock! If you're asking about a specific business location, let me know and I can try to help you find that information.",
        ],
        "description": "User asks about business hours",
        "priority": 4,
    },
    
    "location": {
        "patterns": [
            "where are you located", "your address", "how to find you",
            "location details", "where's your office", "directions",
            "where can i find you", "physical location", "address please",
        ],
        "responses": [
            "I'm a digital assistant, so I don't have a physical location - I exist in the cloud! Is there a specific place you're trying to find?",
            "As an AI, I'm located wherever you need me! If you're looking for a physical business address, let me know which one and I can try to help.",
        ],
        "description": "User asks about location",
        "priority": 4,
    },
    
    "contact": {
        "patterns": [
            "how can i contact you", "contact information", "phone number",
            "email address", "how to reach you", "contact details",
            "get in touch", "ways to contact", "contact support",
        ],
        "responses": [
            "You can always reach me right here in this chat! For other contact methods, what type of support do you need?",
            "I'm always available here in the chat. If you need other forms of contact, please let me know what you're looking for.",
        ],
        "description": "User asks for contact information",
        "priority": 4,
    },
    
    "confused": {
        "patterns": [
            "i don't understand", "what do you mean", "can you explain",
            "i'm confused", "that doesn't make sense", "clarify please",
            "what does that mean", "explain again", "huh",
        ],
        "responses": [
            "I apologize for the confusion! Let me try to explain differently. What part would you like me to clarify?",
            "Sorry about that! I'll try to be clearer. What specifically would you like me to explain?",
            "My apologies for being unclear. Could you tell me which part you'd like me to elaborate on?",
        ],
        "description": "User is confused",
        "priority": 6,
    },
    
    "agree": {
        "patterns": [
            "yes", "yeah", "yep", "sure", "okay", "ok", "correct",
            "that's right", "exactly", "absolutely", "indeed",
            "definitely", "of course", "certainly", "right",
        ],
        "responses": [
            "Great! How would you like to proceed?",
            "Perfect! What would you like to do next?",
            "Excellent! Let me know what else you need.",
        ],
        "description": "User agrees or confirms",
        "priority": 2,
    },
    
    "disagree": {
        "patterns": [
            "no", "nope", "not really", "i don't think so", "incorrect",
            "that's wrong", "not quite", "negative", "no way",
            "i disagree", "that's not right", "wrong",
        ],
        "responses": [
            "I understand. Could you help me understand what you're looking for instead?",
            "No problem! Let me know what would work better for you.",
            "Got it. What would you prefer instead?",
        ],
        "description": "User disagrees or denies",
        "priority": 2,
    },
    
    "smalltalk_howareyou": {
        "patterns": [
            "how are you", "how are you doing", "how's it going",
            "you doing okay", "how do you feel", "are you well",
            "how have you been", "what's new with you",
        ],
        "responses": [
            "I'm doing great, thanks for asking! Ready to help you with whatever you need. How can I assist you today?",
            "I'm functioning perfectly and ready to help! How about you? What can I do for you?",
            "All systems running smoothly! 😄 What brings you here today?",
        ],
        "description": "Small talk about bot's wellbeing",
        "priority": 3,
    },
    
    "smalltalk_age": {
        "patterns": [
            "how old are you", "what's your age", "when were you created",
            "when were you born", "your birthday", "age please",
        ],
        "responses": [
            "I'm quite new in AI terms! I was designed with the latest technology. But age is just a number for us AIs - what matters is how I can help you!",
            "I don't have a traditional age, but I'm built with cutting-edge technology! How can I assist you today?",
        ],
        "description": "Small talk about bot's age",
        "priority": 2,
    },
}


def get_intent_patterns() -> dict[str, dict[str, Any]]:
    """
    Get all intent patterns for training and classification.
    
    Returns:
        dict: Intent name to pattern/metadata mapping
    """
    return {
        name: {
            "patterns": data["patterns"],
            "description": data.get("description", ""),
            "priority": data.get("priority", 0),
        }
        for name, data in INTENT_DATABASE.items()
    }


def get_intent_responses() -> dict[str, list[str]]:
    """
    Get all intent responses for dialogue management.
    
    Returns:
        dict: Intent name to response list mapping
    """
    return {
        name: data["responses"]
        for name, data in INTENT_DATABASE.items()
    }


def get_intent_data(intent_name: str) -> IntentData | None:
    """
    Get complete data for a specific intent.
    
    Args:
        intent_name: Name of the intent
    
    Returns:
        IntentData: Complete intent data or None if not found
    """
    if intent_name not in INTENT_DATABASE:
        return None
    
    data = INTENT_DATABASE[intent_name]
    return IntentData(
        name=intent_name,
        patterns=data["patterns"],
        responses=data["responses"],
        description=data.get("description", ""),
        priority=data.get("priority", 0),
    )


def get_all_intents() -> list[str]:
    """Get list of all intent names."""
    return list(INTENT_DATABASE.keys())
