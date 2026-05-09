import streamlit as st
from PIL import Image
from gtts import gTTS
import io
import traceback
import requests
from huggingface_hub import InferenceClient

# ==========================================
# 1. IMAGE-TO-TEXT FUNCTION (STRICT HTTP API)
# ==========================================
def process_image_to_text(image):
    """Extracts the scenario with strict HTTP status checking to prevent JSON crashes."""
    
    # Swapped to Hugging Face's official, permanent free-tier vision model
    API_URL = "https://api-inference.huggingface.co/models/nlpconnect/vit-gpt2-image-captioning"
    
    # Added Content-Type so the server knows exactly what kind of file it is receiving
    headers = {
        "Authorization": f"Bearer {st.secrets['HF_TOKEN']}",
        "Content-Type": "application/octet-stream"
    }
    
    try:
        # Convert the PIL image into a raw byte array
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format=image.format if image.format else 'JPEG')
        img_bytes = img_byte_arr.getvalue()

        # Send request
        response = requests.post(API_URL, headers=headers, data=img_bytes)
        
        # 1. Check if the server gave us a success code (200 OK)
        if response.status_code != 200:
            st.error(f"Hugging Face Server Error: HTTP {response.status_code}")
            with st.expander("Click to see the raw server response"):
                st.text(response.text) # Prints the HTML or raw text the server sent
            return "a quiet little garden"

        # 2. Safely try to parse the JSON
        try:
            result = response.json()
        except Exception as json_err:
            st.error("Server returned jumbled data instead of JSON.")
            with st.expander("Click to see what the server sent"):
                st.text(response.text)
            return "a quiet little garden"
        
        # 3. Process the successful result
        if isinstance(result, list) and len(result) > 0 and "generated_text" in result[0]:
            scenario = result[0]["generated_text"].lower()
            
            # Scenario Quarantine (Safety Check)
            unsafe_words = ['smoke', 'smoking', 'cigarette', 'cigar', 'weed', 'drunk', 'blood', 'gun', 'kill', 'die']
            if any(bad_word in scenario for bad_word in unsafe_words):
                return "a brave and friendly magical puppy exploring a beautiful, colorful forest"
                
            return scenario
            
        elif isinstance(result, dict) and "error" in result:
            st.warning(f"Server says: {result['error']}")
            return "a quiet little garden"
            
        else:
            return "a quiet little garden"
            
    except Exception as e:
        st.error(f"Critical API Error: {str(e)}")
        with st.expander("Click for full traceback"):
            st.code(traceback.format_exc())
        return "a quiet little garden"

# ==========================================
# 2. STORY GENERATION FUNCTION (API VERSION)
# ==========================================
def generate_story(scenario):
    """Generates a vivid story using the Hugging Face Server API (Chat Format)."""
    
    client = InferenceClient(token=st.secrets["HF_TOKEN"])
    model_id = "mistralai/Mistral-7B-Instruct-v0.2"
    
    # Strict rule in the system prompt to ensure grammatical endings
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
            max_tokens=250,  # Increased to prevent the "guillotine" cut-off
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
