import streamlit as st
from transformers import pipeline
from PIL import Image
from gtts import gTTS
import io
from huggingface_hub import InferenceClient

# ==========================================
# 1. IMAGE-TO-TEXT FUNCTION (ADVANCED VQA)
# ==========================================
@st.cache_resource
def load_vision_model():
    # We use the VQA model to actively interrogate the image
    return pipeline("visual-question-answering", model="Salesforce/blip-vqa-base")

def process_image_to_text(image):
    """Actively interrogates the image for specific storytelling elements."""
    
    vision_model = load_vision_model()
    
    # 1. THE SUBJECT (Who or what is the story about?)
    q_subject = "Who or what is the main character or subject?"
    subject = vision_model(image, question=q_subject)[0]["answer"]
    
    # 2. THE ACTION (What is the drama or movement?)
    q_action = "What is the main action taking place?"
    action = vision_model(image, question=q_action)[0]["answer"]
    
    # 3. THE SETTING (Where does the story happen?)
    q_setting = "Describe the environment, setting, or background."
    setting = vision_model(image, question=q_setting)[0]["answer"]
    
    # 4. THE FLAVOR (What are the colors, mood, or interesting objects?)
    q_flavor = "What are the most prominent colors or objects in the scene?"
    flavor = vision_model(image, question=q_flavor)[0]["answer"]
    
    # 5. THE MASTER SCENARIO
    scenario = f"The scene features {subject} actively {action}. It takes place in a setting with {setting}, highlighted by {flavor}."
    
    # Scenario Quarantine (Safety Check)
    unsafe_words = ['smoke', 'smoking', 'cigarette', 'cigar', 'weed', 'drunk', 'blood', 'gun', 'kill', 'die']
    if any(bad_word in scenario.lower() for bad_word in unsafe_words):
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
            st.text('👀 Interrogating the image like a movie director...')
            scenario = process_image_to_text(image)
            st.info(f"**What the AI sees:** {scenario}")
            
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
