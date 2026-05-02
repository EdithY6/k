import streamlit as st
from PIL import Image
import time
# Uncomment these when you are ready to plug in the models:
# from transformers import pipeline
# from gtts import gTTS 
# import os

# ==========================================
# PAGE CONFIGURATION (Must be the first Streamlit command)
# ==========================================
st.set_page_config(
    page_title="Magic Story Machine",
    page_icon="🪄",
    layout="centered"
)

# ==========================================
# BACKEND FUNCTIONS (Placeholders for your models)
# ==========================================
# Implement the solution using functions for modularity[cite: 1]

def get_image_caption(image):
    """Uses a pre-trained model to generate a caption from the uploaded image[cite: 1]."""
    # TODO: Initialize Salesforce/blip-image-captioning-base pipeline here
    time.sleep(2) # Simulating model loading time
    return "a brave dog wearing a red cape flying through the sky"

def generate_story(caption):
    """Uses a text-generation model to expand the caption into a full story (50-100 words)[cite: 1]."""
    # TODO: Initialize text-generation pipeline here (e.g., GPT-2 or similar)
    time.sleep(2) # Simulating model loading time
    return (
        "Once upon a time, there was a brave dog named Barnaby. "
        "He wasn't an ordinary dog; he had a magical red cape! "
        "One sunny afternoon, Barnaby put on his cape and suddenly, he was flying through the sky. "
        "He soared above the clouds, chasing birds and waving at airplanes. "
        "It was the best adventure ever, and he made it home just in time for dinner."
    )

def create_voiceover(text):
    """Converts the generated text into speech using a TTS model[cite: 1]."""
    # TODO: Implement pyttsx3, gTTS, or a Hugging Face TTS model here
    time.sleep(2) # Simulating processing time
    # For now, we will just return a dummy audio path. 
    # In reality, save your audio to a file like 'story.mp3' and return that filename.
    return None 

# ==========================================
# USER INTERFACE
# ==========================================

# 1. The Welcome Banner
st.title("🪄 The Magic Story Machine 🐉")
st.info("Put a picture in the machine and watch it turn into a magical story!")

# 2. The Magic Portal (Image Input)
st.markdown("### 📸 How to add your picture:")

# Create two columns to show the options clearly
col1, col2 = st.columns(2)
with col1:
    st.info("**Option 1: Upload a file** \n\nClick the 'Browse files' button below to pick a saved drawing.")
with col2:
    # Explicitly mentioning Mac and Windows shortcuts for easy pasting
    st.success("**Option 2: Paste a Screenshot!** \n\nTake a screenshot, click the dashed box below, and press **Cmd+V** (or **Ctrl+V**) to paste it like magic! ✨")

# The uploader will automatically accept the pasted image[cite: 1]
uploaded_file = st.file_uploader("Drop, upload, or PASTE your picture here!", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

if uploaded_file is not None:
    # Display the uploaded/pasted image back to the user
    image = Image.open(uploaded_file)
    st.image(image, caption="Wow! What a great picture!", use_container_width=True)
    
    # 3. The "Make My Story" Button
    if st.button("✨ Make My Story! ✨", use_container_width=True):
        
        # 4. The "Thinking" Phase (Processing States)
        with st.status("The Machine is working its magic...", expanded=True) as status:
            
            st.write("👀 Looking closely at your picture...")
            caption = get_image_caption(image)
            
            st.write("💭 Dreaming up a fun adventure...")
            story_text = generate_story(caption)
            
            st.write("🗣️ Waking up the storyteller...")
            audio_file = create_voiceover(story_text)
            
            status.update(label="✨ Your story is ready!", state="complete", expanded=False)
        
        # 5. The Storybook Output
        st.success("Tada! Here is your adventure:")
        
        # Display the narrative[cite: 1]
        st.markdown(f"### {story_text}")
        
        # Display the audio format[cite: 1]
        if audio_file:
            st.audio(audio_file, format="audio/mp3")
        else:
            st.warning("(Audio player will appear here once TTS is connected!)")
            
        # The "Wow" Factor
        st.balloons()
