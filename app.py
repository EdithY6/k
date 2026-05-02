import streamlit as st
from PIL import Image
import time
# Uncomment these when you are ready to plug in the models:
# from transformers import pipeline
# from gtts import gTTS 
# import os

# ==========================================
# PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="Magic Story Chat",
    page_icon="🪄",
    layout="centered"
)

# ==========================================
# BACKEND FUNCTIONS (Placeholders)
# ==========================================
def get_image_caption(image):
    time.sleep(2) 
    return "a brave dog wearing a red cape flying through the sky"

def generate_story(caption):
    time.sleep(2) 
    return (
        "Once upon a time, there was a brave dog named Barnaby. "
        "He wasn't an ordinary dog; he had a magical red cape! "
        "One sunny afternoon, Barnaby put on his cape and suddenly, he was flying through the sky. "
        "He soared above the clouds, chasing birds and waving at airplanes. "
        "It was the best adventure ever, and he made it home just in time for dinner."
    )

def create_voiceover(text):
    time.sleep(2) 
    return None 

# ==========================================
# USER INTERFACE: CHAT LAYOUT
# ==========================================

st.title("🪄 Magic Story Chat 🐉")

# 1. The Assistant's Welcome Message
with st.chat_message("assistant"):
    st.write("Hello! I am the Magic Story Machine.")
    st.info("💡 **Pro-Tip:** Just click ANYWHERE on the blank background of this page and press **Cmd+V** to paste your screenshot!")

# 2. Hidden Uploader (Catches the pasted image behind the scenes)
uploaded_file = st.file_uploader("Upload", type=["jpg", "jpeg", "png"], label_visibility="hidden")

# 3. The Chat Input (Provides the visual chatbox feel)
user_text = st.chat_input("Or type a message to the machine here...")

if user_text:
    # If they type text instead of pasting an image, remind them how it works
    with st.chat_message("user"):
        st.write(user_text)
    with st.chat_message("assistant"):
        st.write("I love chatting, but I'm best at telling stories! Please paste a picture (Cmd+V) anywhere on the screen so I can get to work.")

# 4. Processing the Pasted Image
if uploaded_file is not None:
    # Display the user's action
    with st.chat_message("user"):
        image = Image.open(uploaded_file)
        st.image(image, width=300, caption="Can you make a story out of this?")
        
    # Display the Assistant's response
    with st.chat_message("assistant"):
        with st.status("Working my magic...", expanded=True) as status:
            
            st.write("👀 Looking closely at your picture...")
            caption = get_image_caption(image)
            
            st.write("💭 Dreaming up a fun adventure...")
            story_text = generate_story(caption)
            
            st.write("🗣️ Waking up the storyteller...")
            audio_file = create_voiceover(story_text)
            
            status.update(label="✨ Your story is ready!", state="complete", expanded=False)
        
        # Output the final generated content
        st.markdown(f"### {story_text}")
        
        if audio_file:
            st.audio(audio_file, format="audio/mp3")
        else:
            st.warning("(Audio player will appear here once TTS is connected!)")
            
        st.balloons()
