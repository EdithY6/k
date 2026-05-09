import streamlit as st
from transformers import pipeline
from PIL import Image
from gtts import gTTS
import io
from huggingface_hub import InferenceClient

# ==========================================
# 1. VISION ENGINE (Local VQA Interrogation)
# ==========================================
@st.cache_resource
def load_vision_model():
    """Loads and caches the VQA model for local image interrogation."""
    return pipeline("visual-question-answering", model="Salesforce/blip-vqa-base")

def process_image_to_text(image):
    """Interrogates the image to build a rich scenario for the storyteller."""
    vision_model = load_vision_model()
    
    # Extracting core storytelling elements
    q_subject = "Who or what is the main character?"
    subject = vision_model(image, question=q_subject)[0]["answer"]
    
    q_action = "What is the main action taking place?"
    action = vision_model(image, question=q_action)[0]["answer"]
    
    q_setting = "Describe the environment or background."
    setting = vision_model(image, question=q_setting)[0]["answer"]
    
    q_flavor = "What are the most prominent colors or objects?"
    flavor = vision_model(image, question=q_flavor)[0]["answer"]
    
    scenario = f"The scene features {subject} actively {action}. It takes place in a setting with {setting}, highlighted by {flavor}."
    
    # Safety Net: Filter out mature or unsafe content for children
    unsafe_words = ['smoke', 'smoking', 'cigarette', 'cigar', 'weed', 'drunk', 'blood', 'gun', 'kill', 'die']
    if any(word in scenario.lower() for word in unsafe_words):
        return "a brave and friendly magical puppy exploring a beautiful, colorful forest"
        
    return scenario

# ==========================================
# 2. STORY ENGINE (Mistral API with Scrubber)
# ==========================================
def generate_story(scenario):
    """Generates a child-friendly story and ensures a clean, complete ending."""
    client = InferenceClient(token=st.secrets["HF_TOKEN"])
    model_id = "mistralai/Mistral-7B-Instruct-v0.2"
    
    messages = [
        {
            "role": "system", 
            "content": (
                "You are a master children's storyteller. Write a short, vivid bedtime story in 1-2 paragraphs. "
                "STRICT RULES: 1. Max 150 words. 2. Clear conclusion. 3. No meta-talk or 'The End'."
            )
        },
        {
            "role": "user", 
            "content": f"Write a complete bedtime story about: {scenario}"
        }
    ]
    
    try:
        response = client.chat_completion(messages=messages, model=model_id, max_tokens=350, temperature=0.7)
        story = response.choices[0].message.content.strip()
        
        # --- SCRUBBER: Cleaning AI artifacts ---
        unwanted = ["The end.", "The End.", "THE END.", "(I know", "Note:"]
        for item in unwanted:
            if item in story: story = story.split(item)[0].strip()
        
        valid_endings = ('.', '!', '?', '"', "”")
        if not story.endswith(valid_endings):
            last_punc = max(story.rfind('.'), story.rfind('!'), story.rfind('?'))
            if last_punc != -1: story = story[:last_punc + 1]

        words = story.split()
        if words:
            conjunctions = ['as', 'and', 'with', 'but', 'or', 'a', 'the', 'of']
            while words and (words[-1].lower() in conjunctions or words[-1][-1] not in valid_endings):
                words.pop()
            story = " ".join(words)
                
        return story
    except Exception:
        return "Once upon a time, a magical star twinkled, and everyone lived happily ever after."

# ==========================================
# 3. VOICE ENGINE (gTTS)
# ==========================================
def convert_story_to_audio(story_text):
    """Converts the text into a playable MP3 buffer."""
    tts = gTTS(text=story_text, lang='en', slow=False)
    audio_buffer = io.BytesIO()
    tts.write_to_fp(audio_buffer)
    audio_buffer.seek(0)
    return audio_buffer

# ==========================================
# 4. FRONT-END (Adaptive Dark/Light UI)
# ==========================================
def main():
    st.set_page_config(page_title="Magic Story Hub", page_icon="🪄")

    # CUSTOM CSS: Adaptive Light and Dark Mode logic
    st.markdown("""
        <style>
        /* Default: Light Mode Aesthetic */
        .stApp { background-color: #f0faff; }
        h1 { color: #ff4b4b; font-family: 'Comic Sans MS', cursive; text-align: center; }
        div.stButton > button:first-child {
            background-color: #ff4b4b; color: white; border-radius: 20px; font-weight: bold; width: 100%;
        }
        .vision-box {
            background-color: #e8f4f8; padding: 15px; border-radius: 10px;
            border-left: 5px solid #2e86de; margin-bottom: 20px; color: #2e86de;
        }
        .story-box {
            background-color: white; padding: 25px; border-radius: 15px;
            border-left: 10px solid #ff4b4b; box-shadow: 2px 2px 10px rgba(0,0,0,0.1);
            color: #333;
        }

        /* Dark Mode Overrides */
        @media (prefers-color-scheme: dark) {
            .stApp { background-color: #0e1117; }
            .story-box { 
                background-color: #262730; 
                color: #fafafa; 
                box-shadow: 2px 2px 15px rgba(0,0,0,0.5);
            }
            .vision-box { 
                background-color: #1a1c23; 
                color: #70a1ff; 
                border-left-color: #70a1ff;
            }
            .stMarkdown p, .stMarkdown h3 { color: #fafafa; }
            h1 { color: #ff4b4b !important; } /* Keep the magic red title */
        }
        </style>
        """, unsafe_allow_html=True)

    st.title("🪄 Magic Story Hub")
    st.write("🌈 **Upload a picture to start your adventure!**")
    
    uploaded_file = st.file_uploader("Pick a picture!", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Your Magic Image", use_container_width=True)

        if st.button("✨ Make the Magic Happen!"):
            with st.spinner("🌟 The forest spirits are reading your picture..."):
                
                # Run the Pipeline
                scenario = process_image_to_text(image)
                story = generate_story(scenario)
                audio_file = convert_story_to_audio(story)

                # Show the AI's "Eyes"
                st.markdown(f'<div class="vision-box"><b>👀 The AI Sees:</b><br>{scenario}</div>', unsafe_allow_html=True)
                
                # Show the Final Narrative
                st.markdown(f'<div class="story-box"><h3>📖 Your Tale:</h3>{story}</div>', unsafe_allow_html=True)
                
                st.write("### 🎧 Listen to your story:")
                st.audio(audio_file, format="audio/mp3")
                st.balloons()

if __name__ == "__main__":
    main()    if any(word in scenario.lower() for word in unsafe_words):
        return "a brave and friendly magical puppy exploring a beautiful, colorful forest"
        
    return scenario

# ==========================================
# 2. STORY ENGINE (Mistral API with Scrubber)
# ==========================================
def generate_story(scenario):
    """Generates a child-friendly story and ensures a clean, complete ending."""
    client = InferenceClient(token=st.secrets["HF_TOKEN"])
    model_id = "mistralai/Mistral-7B-Instruct-v0.2"
    
    messages = [
        {
            "role": "system", 
            "content": (
                "You are a master children's storyteller. Write a short, vivid bedtime story in 1-2 paragraphs. "
                "STRICT RULES: 1. Max 150 words. 2. Clear conclusion. 3. No meta-talk or 'The End'."
            )
        },
        {
            "role": "user", 
            "content": f"Write a complete bedtime story about: {scenario}"
        }
    ]
    
    try:
        response = client.chat_completion(messages=messages, model=model_id, max_tokens=350, temperature=0.7)
        story = response.choices[0].message.content.strip()
        
        # --- SCRUBBER: Cleaning AI artifacts ---
        unwanted = ["The end.", "The End.", "THE END.", "(I know", "Note:"]
        for item in unwanted:
            if item in story: story = story.split(item)[0].strip()
        
        valid_endings = ('.', '!', '?', '"', "”")
        if not story.endswith(valid_endings):
            last_punc = max(story.rfind('.'), story.rfind('!'), story.rfind('?'))
            if last_punc != -1: story = story[:last_punc + 1]

        words = story.split()
        if words:
            conjunctions = ['as', 'and', 'with', 'but', 'or', 'a', 'the', 'of']
            while words and (words[-1].lower() in conjunctions or words[-1][-1] not in valid_endings):
                words.pop()
            story = " ".join(words)
                
        return story
    except Exception:
        return "Once upon a time, a magical star twinkled, and everyone lived happily ever after."

# ==========================================
# 3. VOICE ENGINE (gTTS)
# ==========================================
def convert_story_to_audio(story_text):
    """Converts the text into a playable MP3 buffer."""
    tts = gTTS(text=story_text, lang='en', slow=False)
    audio_buffer = io.BytesIO()
    tts.write_to_fp(audio_buffer)
    audio_buffer.seek(0)
    return audio_buffer

# ==========================================
# 4. FRONT-END (Adaptive Dark/Light UI)
# ==========================================
def main():
    st.set_page_config(page_title="Magic Story Hub", page_icon="🪄")

    # CUSTOM CSS: Adaptive Light and Dark Mode logic
    st.markdown("""
        <style>
        /* Default: Light Mode Aesthetic */
        .stApp { background-color: #f0faff; }
        h1 { color: #ff4b4b; font-family: 'Comic Sans MS', cursive; text-align: center; }
        div.stButton > button:first-child {
            background-color: #ff4b4b; color: white; border-radius: 20px; font-weight: bold; width: 100%;
        }
        .vision-box {
            background-color: #e8f4f8; padding: 15px; border-radius: 10px;
            border-left: 5px solid #2e86de; margin-bottom: 20px; color: #2e86de;
        }
        .story-box {
            background-color: white; padding: 25px; border-radius: 15px;
            border-left: 10px solid #ff4b4b; box-shadow: 2px 2px 10px rgba(0,0,0,0.1);
            color: #333;
        }

        /* Dark Mode Overrides */
        @media (prefers-color-scheme: dark) {
            .stApp { background-color: #0e1117; }
            .story-box { 
                background-color: #262730; 
                color: #fafafa; 
                box-shadow: 2px 2px 15px rgba(0,0,0,0.5);
            }
            .vision-box { 
                background-color: #1a1c23; 
                color: #70a1ff; 
                border-left-color: #70a1ff;
            }
            .stMarkdown p, .stMarkdown h3 { color: #fafafa; }
            h1 { color: #ff4b4b !important; } /* Keep the magic red title */
        }
        </style>
        """, unsafe_allow_html=True)

    st.title("🪄 Magic Story Hub")
    st.write("🌈 **Upload a picture to start your adventure!**")
    
    uploaded_file = st.file_uploader("Pick a picture!", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Your Magic Image", use_container_width=True)

        if st.button("✨ Make the Magic Happen!"):
            with st.spinner("🌟 The forest spirits are reading your picture..."):
                
                # Run the Pipeline
                scenario = process_image_to_text(image)
                story = generate_story(scenario)
                audio_file = convert_story_to_audio(story)

                # Show the AI's "Eyes"
                st.markdown(f'<div class="vision-box"><b>👀 The AI Sees:</b><br>{scenario}</div>', unsafe_allow_html=True)
                
                # Show the Final Narrative
                st.markdown(f'<div class="story-box"><h3>📖 Your Tale:</h3>{story}</div>', unsafe_allow_html=True)
                
                st.write("### 🎧 Listen to your story:")
                st.audio(audio_file, format="audio/mp3")
                st.balloons()

if __name__ == "__main__":
    main()    unsafe_words = ['smoke', 'smoking', 'cigarette', 'cigar', 'weed', 'drunk', 'blood', 'gun', 'kill', 'die']
    if any(word in scenario.lower() for word in unsafe_words):
        return "a brave and friendly magical puppy exploring a beautiful, colorful forest"
        
    return scenario

# ==========================================
# 2. STORY ENGINE (Mistral API)
# ==========================================
def generate_story(scenario):
    """Generates the story and scrubs it for a clean ending."""
    client = InferenceClient(token=st.secrets["HF_TOKEN"])
    model_id = "mistralai/Mistral-7B-Instruct-v0.2"
    
    messages = [
        {
            "role": "system", 
            "content": (
                "You are a master children's storyteller. Write a short, vivid bedtime story in 1-2 paragraphs. "
                "STRICT RULES: 1. Max 150 words. 2. Clear conclusion. 3. No meta-talk or 'The End'."
            )
        },
        {
            "role": "user", 
            "content": f"Write a complete bedtime story about: {scenario}"
        }
    ]
    
    try:
        response = client.chat_completion(messages=messages, model=model_id, max_tokens=350, temperature=0.7)
        story = response.choices[0].message.content.strip()
        
        # --- TEXT CLEANING ---
        # Removing unwanted AI commentary or "The End" headers
        unwanted = ["The end.", "The End.", "THE END.", "(I know", "Note:"]
        for item in unwanted:
            if item in story: story = story.split(item)[0].strip()
        
        # Punctuation check to avoid hanging sentences
        valid_endings = ('.', '!', '?', '"', "”")
        if not story.endswith(valid_endings):
            last_punc = max(story.rfind('.'), story.rfind('!'), story.rfind('?'))
            if last_punc != -1: story = story[:last_punc + 1]

        # Trimming dangling conjunctions (e.g., "...and then a")
        words = story.split()
        if words:
            conjunctions = ['as', 'and', 'with', 'but', 'or', 'a', 'the', 'of']
            while words and (words[-1].lower() in conjunctions or words[-1][-1] not in valid_endings):
                words.pop()
            story = " ".join(words)
                
        return story
    except Exception:
        return "Once upon a time, a magical wind blew and everyone lived happily ever after."

# ==========================================
# 3. VOICE ENGINE (gTTS)
# ==========================================
def convert_story_to_audio(story_text):
    """Converts the story text into an MP3 file buffer."""
    tts = gTTS(text=story_text, lang='en', slow=False)
    audio_buffer = io.BytesIO()
    tts.write_to_fp(audio_buffer)
    audio_buffer.seek(0)
    return audio_buffer

# ==========================================
# 4. FRONT-END (Streamlit UI)
# ==========================================
def main():
    st.set_page_config(page_title="Magic Story Hub", page_icon="🪄")

    # Injecting Custom CSS for kid-friendly aesthetics
    st.markdown("""
        <style>
        .stApp { background-color: #f0faff; }
        h1 { color: #ff4b4b; font-family: 'Comic Sans MS', cursive; text-align: center; }
        div.stButton > button:first-child {
            background-color: #ff4b4b; color: white; border-radius: 20px; font-weight: bold; width: 100%;
        }
        .vision-box {
            background-color: #e8f4f8; padding: 15px; border-radius: 10px;
            border-left: 5px solid #2e86de; margin-bottom: 20px; font-size: 0.9rem;
        }
        .story-box {
            background-color: white; padding: 25px; border-radius: 15px;
            border-left: 10px solid #ff4b4b; box-shadow: 2px 2px 10px rgba(0,0,0,0.1);
        }
        </style>
        """, unsafe_allow_html=True)

    st.title("🪄 Magic Story Hub")
    st.write("🌈 **Upload a picture to start your adventure!**")
    
    uploaded_file = st.file_uploader("Pick a picture!", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Your Magic Image", use_container_width=True)

        if st.button("✨ Make the Magic Happen!"):
            with st.spinner("🌟 The forest spirits are reading your picture..."):
                
                # Execution Pipeline
                scenario = process_image_to_text(image)
                story = generate_story(scenario)
                audio_file = convert_story_to_audio(story)

                # --- DISPLAY ---
                # Show the AI's internal scenario observation
                st.markdown(f'<div class="vision-box"><b>👀 The AI Sees:</b><br>{scenario}</div>', unsafe_allow_html=True)
                
                # Show the final polished story
                st.markdown(f'<div class="story-box"><h3>📖 Your Tale:</h3>{story}</div>', unsafe_allow_html=True)
                
                st.write("### 🎧 Listen to your story:")
                st.audio(audio_file, format="audio/mp3")
                st.balloons()

if __name__ == "__main__":
    main()
