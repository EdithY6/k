# 🪄 Magic Story Hub

**Magic Story Hub** is a kid-friendly Streamlit application that transforms any uploaded image into a vivid, spoken bedtime story. It utilizes a hybrid AI pipeline, combining local computer vision for scene interrogation with cloud-based Large Language Models (LLMs) for high-quality storytelling.

## ✨ Features

* **VQA Interrogation:** Instead of basic captions, the app uses **Visual Question Answering** to actively "ask" the image about characters, actions, settings, and colors to build a rich narrative foundation.
* **AI Storyteller:** Integrates the `Mistral-7B-Instruct-v0.2` model via Hugging Face Inference API for creative and context-aware storytelling.
* **Bulletproof Scrubber:** A custom Python post-processing layer that surgically removes AI meta-commentary (like sassy side-notes) and ensures every story ends on a complete, grammatical sentence.
* **Text-to-Speech (TTS):** Uses `gTTS` to narrate the stories, making it accessible for children who are still learning to read.
* **Kid-Friendly UI:** Designed with a "Magic Book" aesthetic using custom CSS, rounded containers, playful fonts, and interactive "balloon" celebrations.

## 🛠️ Technical Stack

* **Frontend:** [Streamlit](https://streamlit.io/)
* **Vision (Local):** `Salesforce/blip-vqa-base` (via `transformers`)
* **Storytelling (Cloud API):** `Mistral-7B-Instruct-v0.2` (via `huggingface_hub`)
* **Audio Engine:** `gTTS` (Google Text-to-Speech)
* **Image Handling:** `PIL` (Pillow)

## 🚀 Getting Started

### Prerequisites
* Python 3.8+
* A Hugging Face API Token (Free tier works perfectly)

### Setup & Installation

1.  **Clone the Repository:**
    ```bash
    git clone [https://github.com/yourusername/magic-story-hub.git](https://github.com/yourusername/magic-story-hub.git)
    cd magic-story-hub
    ```

2.  **Install Dependencies:**
    ```bash
    pip install streamlit transformers torch pillow gTTS huggingface_hub requests
    ```

3.  **Configure Secrets:**
    Create a `.streamlit/secrets.toml` file in your project root:
    ```toml
    HF_TOKEN = "your_huggingface_api_token_here"
    ```

4.  **Run the App:**
    ```bash
    streamlit run app.py
    ```

## 🧠 Logic Breakdown

### VQA Interrogation
The app doesn't just "see" an image; it interrogates it. It asks:
1.  *Who is the main character?*
2.  *What is the main action?*
3.  *What is the environment?*
4.  *What are the prominent colors?*

This results in a prompt like: *"The scene features a small hobbit actively wading through water. It takes place in a setting with a green river, highlighted by grey cloaks."*

### The "Scrubber" (Post-Processing)
Small LLMs occasionally fail to finish a thought or add meta-commentary like "(I know you said no 'The End')". The scrubber logic:
* Slices text before meta-commentary triggers.
* Back-steps through the string until it finds a valid punctuation mark (`.`, `!`, `?`).
* Removes dangling conjunctions like "as", "and", or "the" to ensure the voice engine doesn't sound awkward.

## 🛡️ Content Safety
Includes a **Scenario Quarantine** filter. Any description containing unsafe keywords is automatically diverted to a default "magical puppy" scenario to ensure the generated content is always safe for children.

---
*Created with magic and AI.*
