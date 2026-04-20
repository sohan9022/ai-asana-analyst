from google import genai
import os
from dotenv import load_dotenv

load_dotenv()
# Configure your API Key (Set this in your environment variables)
client = genai.Client()

def get_yoga_wisdom(pose_name, score, top_mistakes):
    """
    Generates a conversational summary using Gemini based on session data.
    """
    
    # Extract just the human-readable mistake strings
    mistake_list = ", ".join([v["mistake"] for v in top_mistakes])
    
    # Extract the name of the first joint to use in the prompt
    first_joint = top_mistakes[0]["joint"].replace("_", " ") if top_mistakes else "posture"
    
    prompt = f"""
    You are a supportive and expert Yoga Coach. A student just finished a session for {pose_name}.
    Their overall alignment score was {score}/100.
    The main issues detected were: {mistake_list}.

    Write a 3-sentence 'Yoga Wisdom' summary for their report:
    1. An encouraging opening sentence acknowledging their effort.
    2. A technical but gentle tip on why correcting those specific mistakes (like the {first_joint}) is important for their body's energy or safety.
    3. A motivating closing sentence to keep them practicing.
    
    Keep the tone professional, mindful, and concise.
    """

    try:
        # Note: Ensure 'gemini-1.5-flash' or your intended model name is correct here
        response = client.models.generate_content(
            model='gemini-1.5-flash',
            contents=prompt
        )
        return response.text.strip()
    except Exception as e:
        print(f"API Error: {e}") # Added this so you can see if the API call fails in your terminal
        return "Keep practicing with mindfulness; every breath brings you closer to balance."