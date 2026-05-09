import streamlit as st
from transformers import pipeline
from PIL import Image
from gtts import gTTS
import io
from huggingface_hub import InferenceClient

# ==========================================
# 1. IMAGE-TO-TEXT FUNCTION (VQA INTERROGATION)
# ==========================================
# Switched from passive captioning to Active VQA!
@st.cache_resource
def load_vision_model():
    return pipeline("visual-question-answering", model="Salesforce/blip-vqa-base")

def process_image_to_text(image):
    """Extracts the scenario by actively interrogating the image."""
    
    vision_model = load_vision_model()
    
    # 1. Force the AI to focus on the character first
    q_character = "What is the person in the foreground doing?"
    character_action = vision_model(image, question=q_character)[0]["answer"]
    
    # 2. Ask what is happening around them
    q_background = "What is in the background?"
    background_setting = vision_model(image, question=q_background)[0]["answer"]
    
    # 3. Stitch the answers together into a highly focused scenario
    scenario = f"A character is {character_action}, with {background_setting} in the background".lower()
    
    # Scenario Quarantine (Safety Check)
    unsafe_words = ['smoke', 'smoking', 'cigarette', 'cigar', 'weed', 'drunk', 'blood', 'gun', 'kill', 'die']
    if any(bad_word in scenario for bad_word in unsafe_words):
        return "a brave and friendly magical puppy exploring a beautiful, colorful forest"
        
    return scenario

# ==========================================
# 2. STORY GENERATION FUNCTION (API VERSION)
# ==========================================
def generate_story(scenario):
    """Generates a vivid story and guarantees a complete, grammatically correct ending."""
    
    client = InferenceClient(token=st.secrets["HF_TOKEN"])
    model_id = "mistralai/Mistral-7B-Instruct-v0.2"
    
    messages = [
        {
            "role": "system", 
            "content": (
                "You are a highly creative but concise children's storyteller. "
                "Write a very short, exciting bedtime story. "
                "STRICT RULES: "
                "1. The story MUST be under 200 words. "
                "2. You MUST reach a satisfying conclusion. "
                "3. End the story properly with a final, complete sentence. "
                "4. Don't end with 'The End'."
            )
        },
        {
            "role": "user", 
            "content": f"Write a complete, short story based strictly on this scenario: {scenario}"
        }
    ]
    
    try:
        response = client.chat_completion(
            messages=messages,
            model=model_id,
            max_tokens=300,  
            temperature=0.7  
        )
        
        story = response.choices[0].message.content.strip()
        
        # --- THE PYTHON SAFETY NET ---
        valid_endings = ('.', '!', '?', '"', "'", '”', '’')
        if not story.endswith(valid_endings):
            last_period = story.rfind('. ')
            last_exclaim = story.rfind('! ')
            last_question = story.rfind('? ')
            
            last_valid_punctuation = max(last_period, last_exclaim, last_question)
            
            if last_valid_punctuation != -1:
                story = story[:last_valid_punctuation + 1]
                
        return story
        
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
            
            # --- Stage 1: Extract Scenario from Picture (LOCAL VQA) ---
            st.text('👀 Interrogating the image...')
            scenario = process_image_to_text(image)
            st.info(f"**What the AI sees:** {scenario.capitalize()}")
            
            # --- Stage 2: Write Story based on Scenario (API) ---
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
