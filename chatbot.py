import os
from dotenv import load_dotenv
from anthropic import Anthropic

load_dotenv()
client = Anthropic()

# Memory — starts empty, grows with conversation
conversation_history = []

# System prompt: defines who Claude is in this app
SYSTEM_PROMPT = """You are a helpful documentation assistant. 
You answer technical questions clearly and concisely.
If you don't know something, you say so honestly instead of guessing.
Keep responses focused and avoid unnecessary preamble."""

print("Documentation Assistant — type 'quit' to exit\n")

while True:
    user_input = input("You: ").strip()
    
    if user_input.lower() == "quit":
        print("Goodbye!")
        break
    
    if not user_input:
        continue
    
    # Add user's message to history
    conversation_history.append({
        "role": "user",
        "content": user_input
    })
    
    # Send the FULL history to Claude
    response = client.messages.create(
        model="claude-sonnet-4-5-20250929",
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=conversation_history,
    )
    
    assistant_reply = response.content[0].text
    
    # Add Claude's reply to history
    conversation_history.append({
        "role": "assistant",
        "content": assistant_reply
    })
    
    print(f"\nBot: {assistant_reply}\n")