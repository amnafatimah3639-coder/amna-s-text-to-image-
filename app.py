```python
import io
import time

import requests
import streamlit as st
from PIL import Image


# ============================================================
# CONFIGURATION
# ============================================================

HF_MODEL_ID = "stable-diffusion-v1-5/stable-diffusion-v1-5"

# IMPORTANT:
# Use the current Hugging Face Router.
# Do NOT use api-inference.huggingface.co
HF_API_URL = (
    f"https://router.huggingface.co/hf-inference/models/{HF_MODEL_ID}"
)

GROQ_MODEL_ID = "llama-3.3-70b-versatile"


# ============================================================
# PAGE SETTINGS
# ============================================================

st.set_page_config(
    page_title="AI Image Generator",
    page_icon="🎨",
    layout="centered"
)


# ============================================================
# SECRET FUNCTION
# ============================================================

def get_secret(name):
    try:
        return st.secrets[name]

    except KeyError:
        st.error(
            f"Missing `{name}` in Streamlit Secrets. "
            "Please add it in your Streamlit settings."
        )
        st.stop()

    except FileNotFoundError:
        st.error(
            "Streamlit Secrets file was not found."
        )
        st.stop()


# ============================================================
# GROQ PROMPT ENHANCEMENT
# ============================================================

def enhance_prompt_with_groq(prompt, api_key):

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
                    "You are an expert AI image prompt engineer. "
                    "Improve the user's prompt by adding details "
                    "about lighting, composition, atmosphere, "
                    "style and visual quality. "
                    "Return only the improved prompt."
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


# ============================================================
# HUGGING FACE IMAGE GENERATION
# ============================================================

def generate_image(prompt, hf_token, max_retries=3):

    headers = {
        "Authorization": f"Bearer {hf_token}",
        "Content-Type": "application/json"
    }

    data = {
        "inputs": prompt
    }

    for attempt in range(max_retries):

        try:

            response = requests.post(
                HF_API_URL,
                headers=headers,
                json=data,
                timeout=120
            )

        except requests.exceptions.RequestException as error:

            raise RuntimeError(
                f"Could not connect to Hugging Face:\n{error}"
            )

        # Successful response
        if response.status_code == 200:

            try:

                return Image.open(
                    io.BytesIO(response.content)
                )

            except Exception:

                raise RuntimeError(
                    "Hugging Face returned a response, "
                    "but it was not a valid image."
                )

        # Model loading
        if response.status_code == 503:

            try:
                result = response.json()
                wait_time = result.get(
                    "estimated_time",
                    20
                )

            except Exception:
                wait_time = 20

            wait_time = min(
                int(wait_time),
                30
            )

            st.info(
                f"Model is loading. "
                f"Retrying in {wait_time} seconds..."
            )

            time.sleep(wait_time)

            continue

        # Invalid token
        if response.status_code == 401:

            raise RuntimeError(
                "Hugging Face rejected your token. "
                "Please check your HF_TOKEN."
            )

        # Permission problem
        if response.status_code == 403:

            raise RuntimeError(
                "Hugging Face denied access. "
                "Check your token permissions."
            )

        # Model not found
        if response.status_code == 404:

            raise RuntimeError(
                "Hugging Face could not find the model.\n\n"
                f"Model: {HF_MODEL_ID}\n"
                f"URL: {HF_API_URL}"
            )

        # Rate limit
        if response.status_code == 429:

            raise RuntimeError(
                "Hugging Face rate limit reached. "
                "Please wait and try again."
            )

        # Other errors
        raise RuntimeError(
            f"Hugging Face API error "
            f"{response.status_code}:\n"
            f"{response.text}"
        )

    raise RuntimeError(
        "The model did not become available "
        "after several attempts."
    )


# ============================================================
# USER INTERFACE
# ============================================================

st.title("🎨 AI Image Generator")

st.write(
    "Create an image from a text prompt using "
    "Hugging Face AI."
)

st.divider()


prompt = st.text_area(
    "📝 Enter your image prompt",
    placeholder=(
        "Example: A beautiful castle on a mountain "
        "during sunset, cinematic lighting, "
        "highly detailed fantasy art"
    ),
    height=120
)


enhance = st.checkbox(
    "✨ Enhance my prompt with Groq",
    value=False
)


generate = st.button(
    "🎨 Generate Image",
    type="primary",
    use_container_width=True
)


# ============================================================
# GENERATION
# ============================================================

if generate:

    if not prompt.strip():

        st.warning(
            "Please enter a prompt first."
        )

        st.stop()


    # Get Hugging Face token
    hf_token = get_secret("HF_TOKEN")

    final_prompt = prompt.strip()


    # --------------------------------------------------------
    # GROQ
    # --------------------------------------------------------

    if enhance:

        groq_key = get_secret("GROQ_API_KEY")

        with st.spinner(
            "✨ Improving your prompt..."
        ):

            try:

                final_prompt = enhance_prompt_with_groq(
                    prompt.strip(),
                    groq_key
                )

                st.success(
                    "Prompt enhanced successfully!"
                )

                with st.expander(
                    "View enhanced prompt"
                ):
                    st.write(final_prompt)

            except Exception as error:

                st.warning(
                    "Prompt enhancement failed. "
                    "Using your original prompt."
                )

                st.caption(
                    str(error)
                )

                final_prompt = prompt.strip()


    # --------------------------------------------------------
    # IMAGE GENERATION
    # --------------------------------------------------------

    with st.spinner(
        "🎨 Generating image... "
        "Please wait."
    ):

        try:

            image = generate_image(
                final_prompt,
                hf_token
            )

        except Exception as error:

            st.error(
                "❌ Image generation failed."
            )

            st.code(
                str(error)
            )

            st.stop()


    # --------------------------------------------------------
    # SHOW IMAGE
    # --------------------------------------------------------

    st.success(
        "🎉 Image generated successfully!"
    )

    st.image(
        image,
        caption=final_prompt,
        use_container_width=True
    )


    # --------------------------------------------------------
    # DOWNLOAD
    # --------------------------------------------------------

    image_buffer = io.BytesIO()

    image.save(
        image_buffer,
        format="PNG"
    )

    st.download_button(
        label="⬇️ Download Image",
        data=image_buffer.getvalue(),
        file_name="generated_image.png",
        mime="image/png",
        use_container_width=True
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Powered by Hugging Face and Groq"
)
```

### 2. `requirements.txt`

```text
streamlit
requests
Pillow
```

### 3. `.streamlit/secrets.toml`

```toml
HF_TOKEN = "YOUR_NEW_HUGGINGFACE_TOKEN"
GROQ_API_KEY = "YOUR_NEW_GROQ_API_KEY"
```

**Do not put your actual API keys into `app.py`.** Use the newly regenerated keys because the previous ones were exposed.

If you're using **Streamlit Cloud**, put the two secrets in **App → Settings → Secrets** instead of uploading `secrets.toml`.
