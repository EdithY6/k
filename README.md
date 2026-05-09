# 🪄 Magic Story Hub

**Magic Story Hub** is a specialized Streamlit application that transforms images into magical bedtime stories for children aged **3 to 10 years old**. By combining local computer vision with a world-class AI storyteller, it creates a safe and enchanting "living book" experience.

## ✨ Key Features

* **VQA Interrogation:** Uses **Visual Question Answering** (`blip-vqa-base`) to "interrogate" images for characters, settings, and actions, ensuring the story is grounded in reality.
* **A Tale for Every Child:** Powered by `Mistral-7B-Instruct-v0.2`, an AI specifically prompted to act as a creative storyteller for the 3-10 age group.
* **The "Bad-Word Net":** A proactive **Scenario Quarantine** filter that pre-screens image descriptions for sensitive or mature keywords, ensuring the AI never encounters "radioactive" topics.
* **Bulletproof Scrubber:** Custom Python logic that surgically removes AI meta-commentary (sassy side-notes) and guarantees every story ends on a perfect, complete sentence.
* **Audio Narration:** Integrated `gTTS` (Google Text-to-Speech) so children can listen to their adventures even if they aren't reading yet.
* **Whimsical UI:** A "Magic Book" interface featuring custom CSS, playful fonts, and celebration animations (balloons!).

## 🛠️ Technical Stack

* **Frontend:** [Streamlit](https://streamlit.io/)
* **Vision (Local):** `Salesforce/blip-vqa-base` (runs locally to save API costs)
* **Story Model (Cloud):** `Mistral-7B-Instruct-v0.2` (via Hugging Face Inference API)
* **Voice Engine:** `gTTS`
* **Logic:** Python 3.8+

## 🚀 Installation & Setup

1.  **Clone & Install:**
    ```bash
    git clone [https://github.com/yourusername/magic-story-hub.git](https://github.com/yourusername/magic-story-hub.git)
    pip install -r requirements.txt
    ```

2.  **Requirements:**
    Ensure your `requirements.txt` includes: `streamlit`, `transformers`, `torch`, `pillow`, `gTTS`, `huggingface_hub`, `requests`.

3.  **Hugging Face Secret:**
    Add your API token to `.streamlit/secrets.toml`:
    ```toml
    HF_TOKEN = "your_token_here"
    ```

4.  **Run:**
    ```bash
    streamlit run app.py
    ```

## 🛡️ Safety & Moderation

This app employs a **Dual-Layer Safety Strategy**:
1.  **The Entry Net:** Prevents unsafe scenarios from reaching the AI model based on a hard-coded list of restricted keywords.
2.  **Prompt Constraints:** The Mistral model is provided with strict "Negative Constraints" to avoid dark themes, meta-talk, and complex adult vocabulary.

## ⚖️ License
Distributed under the MIT License.

---
*Built with magic, safety, and Mistral AI.*
