import io

import streamlit as st
from PIL import Image
from huggingface_hub import InferenceClient

# ============================================================

# CONFIGURATION

# ============================================================

MODEL_ID = "black-forest-labs/FLUX.1-schnell"
GROQ_MODEL_ID = "llama-3.3-70b-versatile"

# ============================================================

# PAGE SETTINGS

# ============================================================

st.set_page_config(
page_title="AI Image Generator",
page_icon="🎨",
layout="centered",
)

# ============================================================

# GET SECRET

# ============================================================

def get_secret(name):
try:
return st.secrets[name]

```
except KeyError:
    st.error(
        f"Missing `{name}` in Streamlit Secrets."
    )
    st.stop()

except FileNotFoundError:
    st.error(
        "Streamlit Secrets could not be found."
    )
    st.stop()
```

# ============================================================

# GROQ PROMPT ENHANCEMENT

# ============================================================

def enhance_prompt_with_groq(prompt, api_key):

```
import requests

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
                "Improve the user's prompt by adding useful "
                "details about lighting, composition, atmosphere, "
                "style, colors, camera perspective and visual quality. "
                "Return ONLY the improved image prompt."
            ),
        },
        {
            "role": "user",
            "content": prompt,
        },
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
```

# ============================================================

# HUGGING FACE IMAGE GENERATION

# ============================================================

def generate_image(prompt, hf_token):

```
try:

    client = InferenceClient(
        provider="auto",
        api_key=hf_token,
    )

    image = client.text_to_image(
        prompt=prompt,
        model=MODEL_ID,
    )

    return image

except Exception as error:

    error_text = str(error)

    raise RuntimeError(
        f"Hugging Face image generation failed:\n\n"
        f"{error_text}"
    ) from error
```

# ============================================================

# USER INTERFACE

# ============================================================

st.title("🎨 AI Image Generator")

st.write(
"Create beautiful AI-generated images from text prompts."
)

st.caption(
"Powered by Hugging Face FLUX and Groq"
)

st.divider()

# ============================================================

# PROMPT

# ============================================================

prompt = st.text_area(
"📝 Enter your image prompt",

```
placeholder=(
    "Example: A beautiful ancient castle on a mountain "
    "during sunset, cinematic golden lighting, "
    "dramatic clouds, highly detailed fantasy artwork"
),

height=130,
```

)

# ============================================================

# ENHANCE PROMPT

# ============================================================

enhance = st.checkbox(
"✨ Enhance my prompt with Groq",
value=False,
)

# ============================================================

# GENERATE BUTTON

# ============================================================

generate_clicked = st.button(
"🎨 Generate Image",
type="primary",
use_container_width=True,
)

# ============================================================

# GENERATION

# ============================================================

if generate_clicked:

```
# --------------------------------------------------------
# Check prompt
# --------------------------------------------------------

if not prompt.strip():

    st.warning(
        "Please enter an image prompt first."
    )

    st.stop()


# --------------------------------------------------------
# Get Hugging Face token
# --------------------------------------------------------

hf_token = get_secret("HF_TOKEN")

final_prompt = prompt.strip()


# --------------------------------------------------------
# Optional Groq enhancement
# --------------------------------------------------------

if enhance:

    groq_key = get_secret("GROQ_API_KEY")

    with st.spinner(
        "✨ Improving your prompt with Groq..."
    ):

        try:

            final_prompt = enhance_prompt_with_groq(
                prompt.strip(),
                groq_key,
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
                "Groq enhancement failed. "
                "Your original prompt will be used."
            )

            st.caption(
                str(error)
            )

            final_prompt = prompt.strip()


# --------------------------------------------------------
# Generate image
# --------------------------------------------------------

with st.spinner(
    "🎨 Generating your image... Please wait."
):

    try:

        image = generate_image(
            final_prompt,
            hf_token,
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
# Display image
# --------------------------------------------------------

st.success(
    "🎉 Image generated successfully!"
)

st.image(
    image,
    caption=final_prompt,
    use_container_width=True,
)


# --------------------------------------------------------
# Download image
# --------------------------------------------------------

image_buffer = io.BytesIO()

image.save(
    image_buffer,
    format="PNG",
)

st.download_button(
    label="⬇️ Download Image",
    data=image_buffer.getvalue(),
    file_name="generated_image.png",
    mime="image/png",
    use_container_width=True,
)
```

# ============================================================

# FOOTER

# ============================================================

st.divider()

st.caption(
"AI Image Generator • Hugging Face FLUX • Groq"
)
