"""
Text-to-Image Generator
------------------------
Streamlit front end that:
  1. Optionally enhances the user's prompt using the Groq API (LLM).
  2. Generates an image using the Hugging Face Inference API
     (no local GPU needed — works on free Streamlit Cloud).
  3. Lets the user view and download the result.

Required secrets (set in Streamlit Cloud > App settings > Secrets,
or locally in .streamlit/secrets.toml):

    HF_TOKEN = "hf_xxxxxxxxxxxxxxxxxxxx"
    GROQ_API_KEY = "gsk_xxxxxxxxxxxxxxxxxxxx"
"""

import io
import time

import requests
import streamlit as st
from PIL import Image

# -----------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------
HF_MODEL_ID = "runwayml/stable-diffusion-v1-5"
HF_API_URL = f"https://api-inference.huggingface.co/models/{HF_MODEL_ID}"
GROQ_MODEL_ID = "llama-3.3-70b-versatile"

st.set_page_config(page_title="Text-to-Image Generator", page_icon="🎨", layout="centered")

# -----------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------
def get_secret(name: str):
    """Read a secret and give a clear error if it's missing."""
    try:
        return st.secrets[name]
    except (KeyError, FileNotFoundError):
        st.error(
            f"Missing `{name}` in Streamlit secrets. "
            "Add it under App settings > Secrets (or .streamlit/secrets.toml locally)."
        )
        st.stop()


def enhance_prompt_with_groq(prompt: str, api_key: str) -> str:
    """Send the prompt to Groq's LLM to make it more descriptive."""
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": GROQ_MODEL_ID,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a prompt engineer for text-to-image models. "
                    "Rewrite the user's prompt to be more vivid and detailed "
                    "(style, lighting, composition, mood) in 1-2 sentences. "
                    "Return ONLY the rewritten prompt, no explanations or quotes."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.7,
        "max_tokens": 150,
    }
    response = requests.post(url, headers=headers, json=payload, timeout=30)
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"].strip()


def generate_image_hf(prompt: str, hf_token: str, max_retries: int = 3) -> Image.Image:
    """Call the Hugging Face Inference API and return a PIL image."""
    headers = {"Authorization": f"Bearer {hf_token}"}
    payload = {"inputs": prompt}

    for attempt in range(max_retries):
        response = requests.post(HF_API_URL, headers=headers, json=payload, timeout=60)

        if response.status_code == 200:
            return Image.open(io.BytesIO(response.content))

        # Model still loading on HF's servers — wait and retry
        if response.status_code == 503:
            wait_time = response.json().get("estimated_time", 20)
            st.info(f"Model is loading on Hugging Face's servers, retrying in {int(wait_time)}s...")
            time.sleep(min(wait_time, 30))
            continue

        # Any other error
        raise RuntimeError(f"HF API error {response.status_code}: {response.text}")

    raise RuntimeError("Model did not finish loading after several retries. Please try again shortly.")


# -----------------------------------------------------------------
# UI
# -----------------------------------------------------------------
st.title("🎨 Text-to-Image Generator")
st.caption(f"Powered by Hugging Face (`{HF_MODEL_ID}`) and Groq for prompt enhancement.")

prompt = st.text_area(
    "Enter your image prompt",
    placeholder="e.g. a cozy cabin in the mountains during a snowstorm, warm light in the windows",
    height=100,
)

enhance = st.checkbox("✨ Enhance my prompt with Groq before generating", value=False)

generate_clicked = st.button("Generate Image", type="primary", use_container_width=True)

if generate_clicked:
    if not prompt.strip():
        st.warning("Please enter a prompt first.")
        st.stop()

    hf_token = get_secret("HF_TOKEN")
    final_prompt = prompt.strip()

    # Step 1: optional prompt enhancement
    if enhance:
        groq_key = get_secret("GROQ_API_KEY")
        with st.spinner("Enhancing your prompt with Groq..."):
            try:
                final_prompt = enhance_prompt_with_groq(prompt.strip(), groq_key)
                st.success("Prompt enhanced:")
                st.info(final_prompt)
            except Exception as e:
                st.warning(f"Prompt enhancement failed, using original prompt instead. ({e})")
                final_prompt = prompt.strip()

    # Step 2: image generation
    with st.spinner("Generating image... this can take 10-30 seconds."):
        try:
            image = generate_image_hf(final_prompt, hf_token)
        except Exception as e:
            st.error(f"Image generation failed: {e}")
            st.stop()

    # Step 3: display + download
    st.image(image, caption=final_prompt, use_container_width=True)

    buf = io.BytesIO()
    image.save(buf, format="PNG")
    st.download_button(
        label="Download image",
        data=buf.getvalue(),
        file_name="generated_image.png",
        mime="image/png",
        use_container_width=True,
    )

st.divider()
st.caption(
    "Note: the first request to a Hugging Face model may take longer while it "
    "loads on their servers. Subsequent requests are faster."
)
