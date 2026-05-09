import streamlit as st
from transformers import pipeline
from PIL import Image
from gtts import gTTS
import io
import os
from huggingface_hub import InferenceClient

# ==========================================
# 1. VISION ENGINE (Local VQA with Memory Protection)
# ==========================================
@st.cache_resource
def load_vision_model():
    """Loads and caches the VQA model. Handled inside a cache to prevent RAM leakage."""
    try:
        return pipeline("visual-question-answering", model="Salesforce/blip-vqa-base")
    except Exception as e:
        st.error(f"Failed to load local Vision Model: {e}")
        return None

def process_image_to_text(image):
    """
    Interrogates the image while protecting against high-resolution memory crashes.
    """
    # Defensive Measure: Resize image if it's too large (Downsampling)
    # This prevents the local VQA model from exceeding Streamlit's 1GB RAM limit.
    MAX_SIZE = (1024, 1024)
    if image.size[0] > MAX_SIZE[0] or image.size[1] > MAX_SIZE[1]:
        image.thumbnail(MAX_SIZE, Image.Resampling.LANCZOS)

    vision_model = load_vision_model()
    if vision_model is None:
        return "a peaceful forest clearing" # Graceful fallback

    try:
        # Step-by-step interrogation for storytelling attributes
        q_subject = "Who or what is the main character?"
        subject = vision_model(image, question=q_subject)[0]["answer"]
        
        q_action = "What is the main action taking place?"
        action = vision_model(image, question=q_action)[0]["answer"]
        
        q_setting = "Describe the environment or background."
        setting = vision_model(image, question=q_setting)[0]["answer"]
        
        q_flavor = "What are the most prominent colors or objects?"
        flavor = vision_model(image, question=q_flavor)[0]["answer"]
        
        scenario = f"The scene features {subject} actively {action}. It takes place in a setting with {setting}, highlighted by {flavor}."
        
        # English "Bad-Word Net" Safety Filter
        unsafe_words = ['smoke', 'smoking', 'cigarette', 'cigar', 'weed', 'drunk', 'blood', 'gun', 'kill', 'die']
        if any(word in scenario.lower() for word in unsafe_words):
            return "a brave and friendly magical puppy exploring a beautiful, colorful forest"
            
        return scenario
    except Exception as e:
        # Catching unexpected model errors (e.g. jumbled tensors)
        return "a magical garden filled with mystery"

# ==========================================
# 2. STORY ENGINE (API with Timeout Protection)
# ==========================================
def generate_story(scenario):
    """Generates the story with strict API error handling."""
    
    # Defensive Measure: Check for Secret Token before calling API
    if "HF_TOKEN" not in st.secrets:
        return "Oops! The magic key is missing. Please add HF_TOKEN to your secrets."

    client = InferenceClient(token=st.secrets["HF_TOKEN"])
    model_id = "mistralai/Mistral-7B-Instruct-v0.2"
    
    messages = [
        {"role": "system", "content": "You are a master children's storyteller for ages 3-10. Write a 1-paragraph bedtime story. Max 150 words. No meta-talk."},
        {"role": "user", "content": f"Write a complete story about: {scenario}"}
    ]
    
    try:
        # Set a timeout (30s) to prevent the app from hanging forever if the API is slow
        response = client.chat_completion(
            messages=messages, 
            model=model_id, 
            max_tokens=300, 
            temperature=0.7,
            timeout=30 
        )
        story = response.choices[0].message.content.strip()
        
        # The Scrubber logic for clean endings
        unwanted = ["The end.", "The End.", "THE END.", "(I know", "Note:"]
        for item in unwanted:
            if item in story: story = story.split(item)[0].strip()
        
        valid_endings = ('.', '!', '?', '"', "”")
        if not story.endswith(valid_endings):
            last_punc = max(story.rfind('.'), story.rfind('!'), story.rfind('?'))
            if last_punc != -1: story = story[:last_punc + 1]

        return story
    except Exception as e:
        # Handling Rate Limits, Overloaded Servers, or Connectivity issues
        return "The storyteller is a bit tired right now. Let's try again in a moment!"

# ==========================================
# 3. VOICE ENGINE (gTTS)
# ==========================================
def convert_story_to_audio(story_text):
    """Converts story to MP3 with empty-string protection."""
    if not story_text or len(story_text) < 5:
        story_text = "The magic story is currently being written."
        
    try:
        tts = gTTS(text=story_text, lang='en', slow=False)
        audio_buffer = io.BytesIO()
        tts.write_to_fp(audio_buffer)
        audio_buffer.seek(0)
        return audio_buffer
    except Exception:
        return None

# ==========================================
# 4. FRONT-END (Kid-Friendly UI)
# ==========================================
def main():
    st.set_page_config(page_title="Magic Story Hub", page_icon="🪄")

    st.markdown("""
        <style>
        .stApp { background-color: #f0faff; }
        h1 { color: #ff4b4b; font-family: 'Comic Sans MS', cursive; text-align: center; }
        div.stButton > button:first-child { background-color: #ff4b4b; color: white; border-radius: 20px; width: 100%; font-weight: bold; }
        .story-box { background-color: white; padding: 25px; border-radius: 15px; border-left: 10px solid #ff4b4b; box-shadow: 2px 2px 10px rgba(0,0,0,0.1); }
        </style>
        """, unsafe_allow_html=True)

    st.title("🪄 Magic Story Hub")
    
    uploaded_file = st.file_uploader("Upload a photo for your story!", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        try:
            # Defensive Measure: Validation of the image file integrity
            image = Image.open(uploaded_file)
            st.image(image, caption="Your Magic Image", use_container_width=True)

            if st.button("✨ Start the Magic"):
                with st.spinner("🌟 Gathering magic dust..."):
                    
                    # 1. Vision Interrogation
                    scenario = process_image_to_text(image)
                    st.info(f"👀 **AI Observation:** {scenario}")
                    
                    # 2. Story Generation
                    story = generate_story(scenario)
                    st.markdown(f'<div class="story-box"><h3>📖 Your Story</h3>{story}</div>', unsafe_allow_html=True)
                    
                    # 3. Audio Narration
                    audio_data = convert_story_to_audio(story)
                    if audio_data:
                        st.audio(audio_data, format="audio/mp3")
                        st.balloons()
                    else:
                        st.warning("🎧 The storyteller's voice is a bit raspy. You can still read the story above!")

        except Exception as e:
            st.error("Wait! That file doesn't look like a valid picture. Try another one!")

if __name__ == "__main__":
    main()
