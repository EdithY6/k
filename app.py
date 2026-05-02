import streamlit as st
from transformers import pipeline
from PIL import Image

# ==========================================
# CACHED MODEL LOADERS (Protects Cloud Memory)
# ==========================================
@st.cache_resource
def load_image_model():
    return pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")

@st.cache_resource
def load_story_model():
    # Option 1 applied: Swap to an instruction-following model
    return pipeline("text2text-generation", model="google/flan-t5-base")

@st.cache_resource
def load_audio_model():
    return pipeline("text-to-audio", model="Matthijs/mms-tts-eng")


# ==========================================
# 1. IMAGE-TO-TEXT FUNCTION
# ==========================================
def process_image_to_text(image):
    """Takes a PIL image, passes it to the BLIP model, and returns a text scenario."""
    image_to_text_model = load_image_model()
    # The pipeline returns a list with a dictionary, we extract the text
    scenario_results = image_to_text_model(image)
    return scenario_results[0]["generated_text"]


# ==========================================
# 2. STORY GENERATION FUNCTION
# ==========================================
def generate_story(scenario):
    """Takes the text scenario, generates a short G-rated story, and returns the text."""
    story_pipe = load_story_model()
    
    # Give the instruction-tuned model a strict prompt
    safe_prompt = f"Write a sweet, magical, and G-rated bedtime story for a 5-year-old child about this scenario: {scenario}. Make it between 50 and 100 words."
    
    # max_new_tokens is used instead of max_length for text2text models
    story_results = story_pipe(safe_prompt, max_new_tokens=100)
    story = story_results[0]['generated_text']

    # --- The Safety Net (Option 3 applied) ---
    blocklist = ['cigarette', 'marijuana', 'smoke', 'drunk', 'blood', 'gun', 'kill', 'drugs', 'die', 'murder']
    
    # Check if any bad word is in the lowercased story
    if any(bad_word in story.lower() for bad_word in blocklist):
        # Fallback to a hardcoded safe story if the model makes a mistake
        return "The brave hero went on a magical adventure, made lots of new friends, and came home safely just in time for a yummy dinner!"
        
    return story


# ==========================================
# 3. STORY-TO-AUDIO FUNCTION
# ==========================================
def convert_story_to_audio(story_text):
    """Takes the generated story text and converts it to audio arrays."""
    audio_pipe = load_audio_model()
    audio_data = audio_pipe(story_text)
    
    # We return both the audio array and the sample rate needed to play it
    return audio_data["audio"], audio_data["sampling_rate"]


# ==========================================
# 4. MAIN APPLICATION FUNCTION
# ==========================================
def main():
    """Handles the Streamlit User Interface and executes the 3 pipeline functions."""
    
    st.set_page_config(page_title="Your Image to Audio Story", page_icon="🦜")
    st.header("Turn Your Image to Audio Story")
    
    # File Uploader
    uploaded_file = st.file_uploader("Select an Image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Open the image in memory using PIL
        image = Image.open(uploaded_file)
        
        # Display the image using the updated parameter to avoid warnings
        st.image(image, caption="Uploaded Image", use_container_width=True)

        with st.spinner("Processing your magic story... Please wait!"):
            
            # --- Stage 1: Image to Text ---
            st.text('👀 Looking at the image...')
            scenario = process_image_to_text(image)
            st.write(f"**Scenario:** {scenario}")

            # --- Stage 2: Text to Story ---
            st.text('✍️ Writing the story...')
            story = generate_story(scenario)
            st.write(f"**Story:** {story}")

            # --- Stage 3: Story to Audio ---
            st.text('🗣️ Recording the audio...')
            audio_array, sample_rate = convert_story_to_audio(story)

        # Output the Audio Player directly
        st.success("Your story is ready!")
        st.audio(audio_array, sample_rate=sample_rate)


# ==========================================
# EXECUTE THE APP
# ==========================================
# This tells Python to run the main() function when the script starts
if __name__ == "__main__":
    main()
