import streamlit as st
from transformers import pipeline
from PIL import Image

# ==========================================
# 1. CACHE THE MODELS (Loads only once!)
# ==========================================
@st.cache_resource
def load_image_model():
    return pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")

@st.cache_resource
def load_story_model():
    return pipeline("text-generation", model="pranavpsv/genre-story-generator-v2")

@st.cache_resource
def load_audio_model():
    return pipeline("text-to-audio", model="Matthijs/mms-tts-eng")

# Initialize the cached models
image_to_text_model = load_image_model()
story_pipe = load_story_model()
audio_pipe = load_audio_model()

# ==========================================
# 2. MAIN APP UI
# ==========================================
st.set_page_config(page_title="Image to Audio Story", page_icon="🦜")
st.header("Turn Your Image to Audio Story")

uploaded_file = st.file_uploader("Select an Image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open the image in memory using PIL (No need to save to disk!)
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Use a spinner so the user knows it's working
    with st.spinner("Processing your magic story... Please wait!"):
        
        # Stage 1: Image to Text
        st.text('👀 Looking at the image...')
        # Pass the PIL image directly to the pipeline
        scenario_results = image_to_text_model(image)
        scenario = scenario_results[0]["generated_text"]
        st.write(f"**Scenario:** {scenario}")

        # Stage 2: Text to Story 
        st.text('✍️ Writing the story...')
        # Note: Added max_length to prevent the model from rambling forever
        story_results = story_pipe(scenario, max_length=100)
        story = story_results[0]['generated_text']
        st.write(f"**Story:** {story}")

        # Stage 3: Story to Audio 
        st.text('🗣️ Recording the audio...')
        audio_data = audio_pipe(story)

    # The spinner disappears when done. 
    # Now we output the audio directly (No extra button needed!)
    st.success("Your story is ready!")
    
    audio_array = audio_data["audio"]
    sample_rate = audio_data["sampling_rate"]
    
    # st.audio has its own built-in play button UI
    st.audio(audio_array, sample_rate=sample_rate)    
    st.write(f"**Story:** {story}")

    # Stage 3: Story to Audio (Inline)
    st.text('Generating audio data...')
    audio_pipe = pipeline("text-to-audio", model="Matthijs/mms-tts-eng")
    audio_data = audio_pipe(story)

    # Play button
    if st.button("Play Audio"):
        audio_array = audio_data["audio"]
        sample_rate = audio_data["sampling_rate"]
        st.audio(audio_array, sample_rate=sample_rate)
