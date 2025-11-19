# story_generator.py
import streamlit as st
from transformers import pipeline, set_seed
import textwrap
import os

# Optional TTS
try:
    from gtts import gTTS
    from pathlib import Path
    TTS_AVAILABLE = True
except Exception:
    TTS_AVAILABLE = False

@st.cache_resource(show_spinner=False)
def get_generator(model_name="distilgpt2"):
    gen = pipeline("text-generation", model=model_name, device=-1)  # CPU by default
    return gen

def build_prompt(character, setting, mood, theme):
    # Clean inputs and build a clear prompt for the model
    prompt = (
        f"Write a short, imaginative story for kids.\n"
        f"Character: {character}\n"
        f"Setting: {setting}\n"
        f"Mood: {mood}\n"
        f"Theme: {theme}\n\n"
        f"Story:\n"
    )
    return prompt

def generate_story(generator, prompt, max_length=200, num_return_sequences=1, seed=42, temperature=0.9, top_k=50, top_p=0.95):
    set_seed(seed)
    out = generator(prompt,
                    max_length=max_length,
                    num_return_sequences=num_return_sequences,
                    do_sample=True,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p)
    # The pipeline returns a list of dicts with 'generated_text'
    story = out[0]['generated_text']
    # Strip the prompt from the result if model echoed it
    if story.startswith(prompt):
        story = story[len(prompt):].strip()
    # Try to trim to a sensible ending (stop at last full sentence)
    if '.' in story:
        idx = story.rfind('.')
        story = story[:idx+1]
    return textwrap.fill(story.strip(), width=90)

def save_tts(text, filename="story.mp3"):
    if not TTS_AVAILABLE:
        return None
    tts = gTTS(text=text, lang='en')
    outpath = Path(filename)
    tts.save(str(outpath))
    return str(outpath)

# ----- Streamlit UI -----
st.set_page_config(page_title="Mini Story Generator", page_icon="📖")
st.title("📖 Mini Story Generator — Generative AI Project")
st.markdown("Enter a few details and the model will create a short, creative story.")

with st.sidebar:
    st.header("Model settings")
    model_name = st.selectbox("Model", ["distilgpt2", "gpt2"], index=0,
                              help="distilgpt2 is smaller and faster.")
    max_len = st.slider("Max tokens / length", 80, 400, value=200)
    temp = st.slider("Temperature (creativity)", 0.1, 1.2, value=0.9)
    top_p = st.slider("Top-p (nucleus sampling)", 0.1, 1.0, value=0.95)
    top_k = st.slider("Top-k", 10, 200, value=50)
    seed = st.number_input("Random seed", 0, 9999, value=42)
    st.write("Tip: increase temperature (e.g. 1.0) for more surprising stories.")

st.header("Story inputs")
col1, col2 = st.columns(2)
with col1:
    character = st.text_input("Main character name", "Maya the Cat")
    mood = st.selectbox("Mood", ["happy", "mysterious", "adventurous", "calm", "silly"])
with col2:
    setting = st.text_input("Setting", "a floating island in the clouds")
    theme = st.text_input("Theme (one-line)", "friendship and courage")

if st.button("Generate story"):
    with st.spinner("Loading model and generating..."):
        generator = get_generator(model_name)
        prompt = build_prompt(character, setting, mood, theme)
        story = generate_story(generator, prompt,
                               max_length=max_len,
                               seed=seed,
                               temperature=temp,
                               top_k=top_k,
                               top_p=top_p)
    st.subheader("Generated story")
    st.write(story)

    # Offer TTS if available
    if TTS_AVAILABLE:
        tts_file = save_tts(story, filename=f"story_{seed}.mp3")
        if tts_file:
            st.audio(tts_file)
            st.success("Audio created (play above).")
    else:
        st.info("TTS not installed. To enable, pip install gTTS and restart.")

    # Download button for story text
    st.download_button("Download story (txt)", data=story, file_name="story.txt", mime="text/plain")

st.markdown("---")
st.markdown("**Notes / safety:** This demo uses a small pretrained model and doesn't filter for all unsafe outputs. For a public demo, add a simple profanity filter and disallowed-content checks.")
st.markdown("**Extending this project:** Add images (Stable Diffusion / DALL·E) to create an illustrated story, or fine-tune the model on a small dataset of children's stories for improved output.")
