import streamlit as st
from PIL import Image
from gtts import gTTS
import io
from huggingface_hub import InferenceClient

# ==========================================
# 1. IMAGE-TO-TEXT FUNCTION (API VERSION)
# ==========================================
def process_image_to_text(image):
    """Extracts the scenario from the uploaded picture using the HF API."""
    
    client = InferenceClient(token=st.secrets["HF_TOKEN"])
    
    # Upgrading to the 'large' model for better, more accurate descriptions
    model_id = "Salesforce/blip-image-captioning-large"
    
    try:
        # Converting the PIL Image to raw bytes. The API handles bytes much better!
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format=image.format if image.format else 'JPEG')
        img_bytes = img_byte_arr.getvalue()

        # Send the image bytes to the server
        scenario = client.image_to_text(img_bytes, model=model_id).lower()
        
        # Scenario Quarantine (Safety Check)
        unsafe_words = ['smoke', 'smoking', 'cigarette', 'cigar', 'weed', 'drunk', 'blood', 'gun', 'kill', 'die']
        if any(bad_word in scenario for bad_word in unsafe_words):
            return "a brave and friendly magical puppy exploring a beautiful, colorful forest"
            
        return scenario
        
    except Exception as e:
        # Instead of failing silently, we now print the actual error to Streamlit so you know what's wrong
        st.error(f"Vision API Error: {str(e)}")
        return "a quiet little garden" # Shorter fallback

# ==========================================
# 2. STORY GENERATION FUNCTION (API VERSION)
# ==========================================
def generate_story(scenario):
    """Generates a vivid story using the Hugging Face Server API."""
    
    client = InferenceClient(token=st.secrets["HF_TOKEN"])
    model_id = "mistralai/Mistral-7B-Instruct-v0.2"
    
    # I added a strict rule to the system prompt to ensure grammatical endings
    messages = [
        {
            "role": "system", 
            "content": "You are a wonderful, creative children's storyteller. Write an exciting, vivid bedtime story (about 50 to 100 words). You MUST ensure the story reaches a natural, complete, and grammatically correct ending."
        },
        {
            "role": "user", 
            "content": f"Write a complete story based strictly on this scenario: {scenario}"
        }
    ]
    
    try:
        response = client.chat_completion(
            messages=messages,
            model=model_id,
            max_tokens=250,  # INCREASED from 150 to prevent the "guillotine" cut-off
            temperature=0.7  
        )
        return response.choices[0].message.content.strip()
        
    except Exception as e:
        return f"Oops! The storyteller is taking a nap. (API Error: {str(e)})"

# ==========================================
# 3. STORY-TO-AUDIO FUNCTION
# ==========================================
def convert_story_to_audio(story_text):
    """Uses gTTS for lightning-fast text-to-speech conversion."""
    
    tts = gTTS(text=story_text, lang='en', slow=False)
    audio_buffer = io.BytesIO()
    tts.write_to_fp(audio_buffer)
    audio_buffer.seek(0)
    
    return audio_buffer

# ==========================================
# 4. MAIN APPLICATION FUNCTION
# ==========================================
def main():
    """Handles the user-friendly Streamlit UI and executes the pipelines."""
    
    st.set_page_config(page_title="Magic Story Machine", page_icon="🪄")
    st.header("Turn Your Image into a Magic Story!")
    
    uploaded_file = st.file_uploader("Select an Image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)

        with st.spinner("Processing your magic story... Please wait!"):
            
            # --- Stage 1: Extract Scenario from Picture ---
            st.text('👀 Looking closely at the image...')
            scenario = process_image_to_text(image)
            st.info(f"**What the AI sees:** {scenario.capitalize()}")
            
            # --- Stage 2: Write Story based on Scenario ---
            st.text('✍️ Writing a vivid adventure...')
            story = generate_story(scenario)
            st.write(f"**📖 The Story:**\n\n{story}")

            # --- Stage 3: Convert to Audio ---
            st.text('🗣️ Recording the storyteller...')
            audio_file = convert_story_to_audio(story)

        st.success("Your story is ready! Hit play to listen.")
        st.audio(audio_file, format="audio/mp3")

# ==========================================
# EXECUTE THE APP
# ==========================================
if __name__ == "__main__":
    main()
