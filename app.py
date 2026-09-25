import io
import requests
import streamlit as st
from PIL import Image
from huggingface_hub import InferenceClient

MODEL_ID = "black-forest-labs/FLUX.1-schnell"
GROQ_MODEL_ID = "llama-3.3-70b-versatile"

st.set_page_config(
    page_title="AI Image Generator",
    page_icon="🎨",
    layout="centered"
)


def get_secret(name):
    try:
        return st.secrets[name]
    except KeyError:
        st.error(f"Missing {name} in Streamlit Secrets.")
        st.stop()
    except FileNotFoundError:
        st.error("Streamlit Secrets file was not found.")
        st.stop()


def enhance_prompt(prompt, api_key):
    url = "https://api.groq.com/openai/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    data = {
        "model": GROQ_MODEL_ID,
        "messages": [
            {
                "role": "system",
                "content": (
                    "Improve the user's image prompt. Add useful "
                    "details about lighting, composition, atmosphere, "
                    "style, colors, and visual quality. Return only "
                    "the improved prompt."
                )
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        "temperature": 0.7,
        "max_tokens": 200
    }

    response = requests.post(
        url,
        headers=headers,
        json=data,
        timeout=60
    )

    response.raise_for_status()

    result = response.json()

    return result["choices"][0]["message"]["content"].strip()


def generate_image(prompt, hf_token):
    try:
        client = InferenceClient(
            provider="auto",
            api_key=hf_token
        )

        image = client.text_to_image(
            prompt=prompt,
            model=MODEL_ID
        )

        return image

    except Exception as error:
        raise RuntimeError(
            f"Hugging Face image generation failed: {error}"
        ) from error


st.title("AI Image Generator")
st.write("Create an AI image from a text prompt.")
st.divider()

prompt = st.text_area(
    "Enter your image prompt",
    placeholder=(
        "Example: A beautiful castle on a mountain at sunset, "
        "cinematic lighting, dramatic clouds, detailed fantasy art"
    ),
    height=130
)

enhance = st.checkbox(
    "Enhance my prompt with Groq",
    value=False
)

generate = st.button(
    "Generate Image",
    type="primary",
    use_container_width=True
)


if generate:

    if not prompt.strip():
        st.warning("Please enter a prompt first.")
        st.stop()

    hf_token = get_secret("HF_TOKEN")
    final_prompt = prompt.strip()

    if enhance:

        groq_key = get_secret("GROQ_API_KEY")

        with st.spinner("Improving your prompt..."):

            try:
                final_prompt = enhance_prompt(
                    prompt.strip(),
                    groq_key
                )

                st.success("Prompt enhanced!")

                with st.expander("View enhanced prompt"):
                    st.write(final_prompt)

            except Exception as error:
                st.warning(
                    "Groq enhancement failed. "
                    "Using your original prompt."
                )
                st.caption(str(error))
                final_prompt = prompt.strip()

    with st.spinner("Generating image... Please wait."):

        try:
            image = generate_image(
                final_prompt,
                hf_token
            )

        except Exception as error:
            st.error("Image generation failed.")
            st.code(str(error))
            st.stop()

    st.success("Image generated successfully!")

    st.image(
        image,
        caption=final_prompt,
        use_container_width=True
    )

    image_buffer = io.BytesIO()

    image.save(
        image_buffer,
        format="PNG"
    )

    st.download_button(
        label="Download Image",
        data=image_buffer.getvalue(),
        file_name="generated_image.png",
        mime="image/png",
        use_container_width=True
    )

st.divider()

st.caption(
    "Powered by Hugging Face FLUX and Groq"
)
