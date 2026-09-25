import io
import time

import requests
import streamlit as st
from PIL import Image


HF_MODEL_ID = "black-forest-labs/FLUX.1-schnell"
HF_API_URL = f"https://router.huggingface.co/hf-inference/models/{HF_MODEL_ID}"
GROQ_MODEL_ID = "llama-3.3-70b-versatile"


st.set_page_config(
    page_title="AI Image Generator",
    page_icon="🎨",
    layout="centered",
)


def get_secret(name):
    try:
        return st.secrets[name]
    except KeyError:
        st.error(f"Missing `{name}` in Streamlit Secrets.")
        st.stop()
    except FileNotFoundError:
        st.error("Streamlit Secrets file was not found.")
        st.stop()


def enhance_prompt_with_groq(prompt, api_key):
    url = "https://api.groq.com/openai/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    data = {
        "model": GROQ_MODEL_ID,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are an expert AI image prompt engineer. "
                    "Improve the user's prompt by adding useful details "
                    "about lighting, composition, atmosphere, style, "
                    "colors, and visual quality. Return only the improved prompt."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.7,
        "max_tokens": 200,
    }

    response = requests.post(
        url,
        headers=headers,
        json=data,
        timeout=60,
    )
    response.raise_for_status()

    result = response.json()
    return result["choices"][0]["message"]["content"].strip()


def generate_image(prompt, hf_token, max_retries=3):
    headers = {
        "Authorization": f"Bearer {hf_token}",
        "Content-Type": "application/json",
    }

    data = {"inputs": prompt}

    for attempt in range(max_retries):
        try:
            response = requests.post(
                HF_API_URL,
                headers=headers,
                json=data,
                timeout=180,
            )
        except requests.exceptions.RequestException as error:
            raise RuntimeError(
                f"Could not connect to Hugging Face: {error}"
            ) from error

        if response.status_code == 200:
            try:
                return Image.open(io.BytesIO(response.content))
            except Exception as error:
                raise RuntimeError(
                    "Hugging Face returned a response, but it was not a valid image."
                ) from error

        if response.status_code == 503:
            try:
                result = response.json()
                wait_time = result.get("estimated_time", 20)
            except Exception:
                wait_time = 20

            wait_time = min(int(wait_time), 30)
            st.info(
                f"FLUX is loading. Retrying in {wait_time} seconds..."
            )
            time.sleep(wait_time)
            continue

        if response.status_code == 401:
            raise RuntimeError(
                "Hugging Face rejected your token. "
                "Check that HF_TOKEN is valid and that you have access to FLUX.1-schnell."
            )

        if response.status_code == 403:
            raise RuntimeError(
                "Hugging Face denied access to the model. "
                "Check your token permissions and model access."
            )

        if response.status_code == 404:
            raise RuntimeError(
                "The FLUX model or Hugging Face endpoint could not be found. "
                f"Model: {HF_MODEL_ID}"
            )

        if response.status_code == 400:
            try:
                error_data = response.json()
                error_message = error_data.get("error", response.text)
            except Exception:
                error_message = response.text

            raise RuntimeError(
                "Hugging Face rejected the selected provider/model. "
                f"Details: {error_message}"
            )

        if response.status_code == 429:
            raise RuntimeError(
                "Hugging Face rate limit reached. Please wait and try again."
            )

        raise RuntimeError(
            f"Hugging Face API error {response.status_code}: {response.text}"
        )

    raise RuntimeError(
        "The FLUX model did not become available after several attempts."
    )


st.title("🎨 AI Image Generator")
st.write("Create an image from a text prompt using Hugging Face FLUX.")
st.divider()

prompt = st.text_area(
    "📝 Enter your image prompt",
    placeholder=(
        "Example: A beautiful castle on a mountain during sunset, "
        "cinematic lighting, highly detailed fantasy art"
    ),
    height=120,
)

enhance = st.checkbox(
    "✨ Enhance my prompt with Groq",
    value=False,
)

generate = st.button(
    "🎨 Generate Image",
    type="primary",
    use_container_width=True,
)


if generate:
    if not prompt.strip():
        st.warning("Please enter a prompt first.")
        st.stop()

    hf_token = get_secret("HF_TOKEN")
    final_prompt = prompt.strip()

    if enhance:
        groq_key = get_secret("GROQ_API_KEY")

        with st.spinner("✨ Improving your prompt with Groq..."):
            try:
                final_prompt = enhance_prompt_with_groq(
                    prompt.strip(),
                    groq_key,
                )

                st.success("Prompt enhanced successfully!")

                with st.expander("View enhanced prompt"):
                    st.write(final_prompt)

            except Exception as error:
                st.warning(
                    "Groq prompt enhancement failed. Using your original prompt."
                )
                st.caption(str(error))
                final_prompt = prompt.strip()

    with st.spinner("🎨 Generating image with FLUX..."):
        try:
            image = generate_image(
                final_prompt,
                hf_token,
            )
        except Exception as error:
            st.error("❌ Image generation failed.")
            st.code(str(error))
            st.stop()

    st.success("🎉 Image generated successfully!")

    st.image(
        image,
        caption=final_prompt,
        use_container_width=True,
    )

    image_buffer = io.BytesIO()
    image.save(image_buffer, format="PNG")

    st.download_button(
        label="⬇️ Download Image",
        data=image_buffer.getvalue(),
        file_name="generated_image.png",
        mime="image/png",
        use_container_width=True,
    )

st.divider()
st.caption("Powered by Hugging Face FLUX and Groq")
