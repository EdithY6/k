import streamlit as st
from transformers import pipeline
from PIL import Image
from gtts import gTTS
import io
from huggingface_hub import InferenceClient # <--- NEW IMPORT

# ==========================================
# 1. IMAGE-TO-TEXT FUNCTION
# ==========================================
def process_image_to_text(image):
    if 'image_model' not in st.session_state:
        st.session_state.image_model = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")
    
    scenario = st.session_state.image_model(image)[0]["generated_text"].lower()
    
    unsafe_words = ['smoke', 'smoking', 'cigarette', 'cigar', 'weed', 'drunk', 'blood', 'gun', 'kill', 'die']
    if any(bad_word in scenario for bad_word in unsafe_words):
        return "a brave and friendly magical puppy exploring a beautiful, colorful forest"
        
    return scenario

# ==========================================
# 2. STORY GENERATION FUNCTION (API VERSION)
# ==========================================
def generate_story(scenario):
    """Generates a vivid story using the Hugging Face Server API."""
    
    # Initialize the API client using your Streamlit Secret
    client = InferenceClient(token=st.secrets["HF_TOKEN"])
    
    # We use the official Mistral Instruct model, as it is always active on the free API
    model_id = "mistralai/Mistral-7B-Instruct-v0.2"
    
    # We use Mistral's [INST] tags to force it into the storyteller persona
    prompt = f"<s>[INST] You are a wonderful, creative children's storyteller. Write an exciting, vivid bedtime story (about 50 to 100 words) based strictly on this scenario: {scenario}. [/INST]"
    
    # Call the Hugging Face API instead of running it locally
    try:
        response = client.text_generation(
            prompt,
            model=model_id,
            max_new_tokens=150,
            temperature=0.7
        )
        return response.strip()
    except Exception as e:
        return f"Oops! The storyteller is taking a nap. (API Error: {str(e)})"

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
# 4. MAIN APPLICATION FUNCTION
# ==========================================
def main():
    st.set_page_config(page_title="Magic Story Machine", page_icon="🪄")
    st.header("Turn Your Image into a Magic Story!")
    
    uploaded_file = st.file_uploader("Select an Image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)

        with st.spinner("Processing your magic story... Please wait!"):
            
            st.text('👀 Looking closely at the image...')
            scenario = process_image_to_text(image)
            
            st.text('✍️ Writing a vivid adventure...')
            story = generate_story(scenario)
            st.write(f"**📖 The Story:**\n\n{story}")

            st.text('🗣️ Recording the storyteller...')
            audio_file = convert_story_to_audio(story)

        st.success("Your story is ready! Hit play to listen.")
        st.audio(audio_file, format="audio/mp3")

if __name__ == "__main__":
    main()
