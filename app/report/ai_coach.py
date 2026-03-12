import google.generativeai as genai
import os

# Configure your API Key (Set this in your environment variables)
genai.configure(api_key="AIzaSyBe7I3mCae9lNV-Snhw1lgwfOlNrfJhdV0")

def get_yoga_wisdom(pose_name, score, top_mistakes):
    """
    Generates a conversational summary using Gemini based on session data.
    """
    model = genai.GenerativeModel('gemini-1.5-flash')
    
    # Prepare the context for the AI
    mistake_list = ", ".join([f"{m[0]} (detected {m[1]} times)" for m in top_mistakes])
    
    prompt = f"""
    You are a supportive and expert Yoga Coach. A student just finished a session for {pose_name}.
    Their overall alignment score was {score}/100.
    The main issues detected were: {mistake_list}.

    Write a 3-sentence 'Yoga Wisdom' summary for their report:
    1. An encouraging opening sentence acknowledging their effort.
    2. A technical but gentle tip on why correcting those specific mistakes (like the {top_mistakes[0][0] if top_mistakes else 'posture'}) is important for their body's energy or safety.
    3. A motivating closing sentence to keep them practicing.
    
    Keep the tone professional, mindful, and concise.
    """

    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return "Keep practicing with mindfulness; every breath brings you closer to balance."