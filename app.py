import streamlit as st
from transformers import pipeline
from PIL import Image
from gtts import gTTS
import io
import traceback
import requests
from huggingface_hub import InferenceClient

# ==========================================
# 1. IMAGE-TO-TEXT FUNCTION (ADVANCED VQA)
# ==========================================
@st.cache_resource
def load_vision_model():
    return pipeline("visual-question-answering", model="Salesforce/blip-vqa-base")

def process_image_to_text(image):
    """Actively interrogates the image for specific storytelling elements."""
    vision_model = load_vision_model()
    
    q_subject = "Who or what is the main character or subject?"
    subject = vision_model(image, question=q_subject)[0]["answer"]
    
    q_action = "What is the main action taking place?"
    action = vision_model(image, question=q_action)[0]["answer"]
    
    q_setting = "Describe the environment, setting, or background."
    setting = vision_model(image, question=q_setting)[0]["answer"]
    
    q_flavor = "What are the most prominent colors or objects in the scene?"
    flavor = vision_model(image, question=q_flavor)[0]["answer"]
    
    scenario = f"The scene features {subject} actively {action}. It takes place in a setting with {setting}, highlighted by {flavor}."
    
    unsafe_words = ['smoke', 'smoking', 'cigarette', 'cigar', 'weed', 'drunk', 'blood', 'gun', 'kill', 'die']
    if any(bad_word in scenario.lower() for bad_word in unsafe_words):
        return "a brave and friendly magical puppy exploring a beautiful, colorful forest"
        
    return scenario

# ==========================================
# 2. STORY GENERATION FUNCTION (BULLETPROOF)
# ==========================================
def generate_story(scenario):
    """Generates a vivid story with aggressive trimming for a clean finish."""
    client = InferenceClient(token=st.secrets["HF_TOKEN"])
    model_id = "mistralai/Mistral-7B-Instruct-v0.2"
    
    messages = [
        {
            "role": "system", 
            "content": (
                "You are a master children's storyteller. "
                "Write a short, vivid bedtime story in 1 or 2 paragraphs. "
                "STRICT RULES: "
                "1. Total length must be under 150 words. "
                "2. You MUST reach a clear, satisfying conclusion. "
                "3. End with a complete, impactful sentence. "
                "4. Do not include 'The End' or any meta-commentary."
            )
        },
        {
            "role": "user", 
            "content": f"Write a complete bedtime story about: {scenario}"
        }
    ]
    
    try:
        response = client.chat_completion(
            messages=messages,
            model=model_id,
            max_tokens=350,
            temperature=0.7  
        )
        
        story = response.choices[0].message.content.strip()
        
        # --- POST-PROCESSING SCRUBBER ---
        unwanted = ["The end.", "The End.", "THE END.", "(I know", "Note:"]
        for item in unwanted:
            if item in story:
                story = story.split(item)[0].strip()
        
        # Ensure it ends on a valid punctuation mark
        valid_endings = ('.', '!', '?', '"', "”")
        if not story.endswith(valid_endings):
            last_punc = max(story.rfind('.'), story.rfind('!'), story.rfind('?'))
            if last_punc != -1:
                story = story[:last_punc + 1]

        # Final check for dangling words (like "as a")
        words = story.split()
        if words:
            conjunctions = ['as', 'and', 'with', 'but', 'or', 'a', 'the', 'of']
            while words and (words[-1].lower() in conjunctions or words[-1][-1] not in valid_endings):
                words.pop()
            story = " ".join(words)
                
        return story
    except Exception:
        return "Once upon a time, a magic cloud covered the forest, and everyone lived happily ever after."

# ==========================================
# 3. STORY-TO-AUDIO FUNCTION
# ==========================================
def convert_story_to_audio(story_text):
    tts = gTTS(text=story_text, lang='en', slow=False)
    audio_buffer = io.BytesIO()
    tts.write_to_fp(audio_buffer)
    audio_buffer.seek(0)
    return audio_buffer

# ==========================================
# 4. MAIN APPLICATION (KID-FRIENDLY UI)
# ==========================================
def main():
    st.set_page_config(page_title="Magic Story Machine", page_icon="🪄")

    # --- CUSTOM STYLING (FIXED unsafe_allow_html) ---
    st.markdown("""
        <style>
        .stApp {
            background-color: #f0faff;
        }
        h1 {
            color: #ff4b4b;
            font-family: 'Comic Sans MS', cursive, sans-serif;
            text-align: center;
        }
        .stMarkdown p {
            font-size: 1.2rem;
            color: #333;
        }
        div.stButton > button:first-child {
            background-color: #ff4b4b;
            color: white;
            border-radius: 20px;
            border: 2px solid #ff4b4b;
            font-weight: bold;
            width: 100%;
        }
        .story-box {
            background-color: white;
            padding: 25px;
            border-radius: 15px;
            border-left: 10px solid #ff4b4b;
            box-shadow: 2px 2px 10px rgba(0,0,0,0.1);
            margin-top: 20px;
            margin-bottom: 20px;
        }
        </style>
        """, unsafe_allow_html=True)

    st.title("🪄 Magic Story Machine")
    st.write("🌈 **Upload a picture and watch it turn into a
