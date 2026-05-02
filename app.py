import streamlit as st
from transformers import pipeline
from PIL import Image

# ==========================================
# 1. IMAGE-TO-TEXT FUNCTION
# ==========================================
# We use @st.cache_resource here so the model only loads once
@st.cache_resource
def process_image_to_text(_image):
    """Takes a PIL image, passes it to the BLIP model, cleans it, and returns a scenario."""
    image_model = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")
    scenario = image_model(_image)[0]["generated_text"]
    
    # Clean the scenario of any hallucinated bad words
    blocklist = ['cigarette', 'marijuana', 'smoke', 'drunk', 'blood', 'gun', 'kill', 'drugs', 'die', 'murder', 'weed']
    for bad_word in blocklist:
        scenario = scenario.lower().replace(bad_word, "flower")
        
    return scenario

# ==========================================
# 2. STORY GENERATION FUNCTION
# ==========================================
@st.cache_resource
def generate_story(scenario):
    """Takes the clean scenario, generates a vivid G-rated story, and returns the text."""
    # Using flan-t5-large for much better, more descriptive creative writing
    story_model = pipeline("text2text-generation", model="google/flan-t5-large")
    
    # A highly specific prompt demanding vivid, descriptive language
    prompt = f"Write an exciting, vivid, and highly descriptive children's bedtime story about this scenario: {scenario}. Use lots of adjectives, colorful details, and make it magical. Keep it between 50 and 100 words."
    
    # Generate with more tokens for longer sentences
    story_results = story_model(prompt, max_new_tokens=150)
    story = story_results[0]['generated_text']

    # Final Safety Net on the output story
    blocklist = ['cigarette', 'marijuana', 'smoke', 'drunk', 'blood', 'gun', 'kill', 'drugs', 'die', 'murder', 'weed']
    if any(bad_word in story.lower() for bad_word in blocklist):
        return "The brave explorer went on a colorful, magical adventure, found a hidden treasure chest full of glowing gems, and made it home just in time for a wonderful feast!"
        
    return story

# ==========================================
# 3. STORY-TO-AUDIO FUNCTION
# ==========================================
@st.cache_resource
def convert_story_to_audio(story_text):
    """Takes the generated story text and converts it to audio arrays."""
    audio_model = pipeline("text-to-audio", model="Matthijs/mms-tts-eng")
    audio_data = audio_model(story_text)
    return audio_data["audio"], audio_data["sampling_rate"]

# ==========================================
# 4. MAIN APPLICATION FUNCTION
# ==========================================
def main():
    """Handles the Streamlit User Interface and executes the 3 pipeline functions."""
    
    st.set_page_config(page_title="Magic Story Machine", page_icon="🪄")
    st.header("Turn Your Image to a Magic Story!")
    
    uploaded_file = st.file_uploader("Select an Image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)

        with st.spinner("Processing your magic story... Please wait!"):
            
            st.text('👀 Looking closely at the image...')
            # Note: We pass the image directly; the cache decorator handles it
            scenario = process_image_to_text(image)
            st.write(f"**Scenario:** {scenario}")

            st.text('✍️ Writing a vivid adventure...')
            story = generate_story(scenario)
            st.write(f"**Story:** {story}")

            st.text('🗣️ Recording the storyteller...')
            audio_array, sample_rate = convert_story_to_audio(story)

        st.success("Your story is ready!")
        st.audio(audio_array, sample_rate=sample_rate)

# ==========================================
# EXECUTE THE APP
# ==========================================
if __name__ == "__main__":
    main()
