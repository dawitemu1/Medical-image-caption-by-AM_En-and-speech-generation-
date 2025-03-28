import streamlit as st
from keras.models import load_model
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import VGG16, preprocess_input
from PIL import Image
import numpy as np
import pyttsx3
import pickle
from keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt
import librosa
import librosa.display
import requests
import json

# Load your custom-trained VGG16 + LSTM model and tokenizer
@st.cache_resource
def load_custom_captioning_model():
    try:
        model = load_model("my_model.keras")
        with open("caption.pkl", "rb") as f:
            tokenizer = pickle.load(f)
        vgg16_model = VGG16(weights="imagenet", include_top=True)
        return model, tokenizer, vgg16_model
    except Exception as e:
        st.error(f"Error loading models: {e}")
        st.stop()

model, tokenizer, vgg16_model = load_custom_captioning_model()

def preprocess_image(image):
    try:
        image = image.resize((224, 224))
        image_array = img_to_array(image)
        image_array = np.expand_dims(image_array, axis=0)
        image_array = preprocess_input(image_array)
        return image_array
    except Exception as e:
        raise RuntimeError(f"Image preprocessing error: {e}")

def extract_features(image):
    try:
        image_array = preprocess_image(image)
        features = vgg16_model.predict(image_array, verbose=0)
        return features
    except Exception as e:
        raise RuntimeError(f"Feature extraction error: {e}")

def generate_caption(image):
    try:
        features = extract_features(image)
        max_sequence_length = model.input[1].shape[1]

        caption_sequence = [tokenizer.word_index.get("<start>", 0)]

        for _ in range(max_sequence_length):
            sequence = pad_sequences([caption_sequence], maxlen=max_sequence_length, padding='post')
            predicted_word_probs = model.predict([features, sequence], verbose=0)
            predicted_word_index = np.argmax(predicted_word_probs)
            predicted_word = tokenizer.index_word.get(predicted_word_index, '')

            if not predicted_word or predicted_word == "<end>":
                break

            caption_sequence.append(predicted_word_index)

        caption = ' '.join(
            [tokenizer.index_word[word] for word in caption_sequence 
             if word != tokenizer.word_index.get("<start>", 0)]
        )
        return caption
    except Exception as e:
        raise RuntimeError(f"Caption generation error: {e}")

def text_to_audio_wav(text, output_audio_path):
    try:
        engine = pyttsx3.init()
        engine.save_to_file(text, output_audio_path)
        engine.runAndWait()
    except Exception as e:
        raise RuntimeError(f"Audio generation error: {e}")

def translate_to_amharic(text, max_length=500):
    """Translate text to Amharic with chunking for long texts"""
    try:
        # Split long text into chunks
        if len(text) > max_length:
            chunks = [text[i:i+max_length] for i in range(0, len(text), max_length)]
            translated_chunks = []
            
            for chunk in chunks:
                params = {'q': chunk, 'langpair': 'en|am'}
                response = requests.get('https://api.mymemory.translated.net/get', params=params)
                response.raise_for_status()
                translation = response.json().get('responseData', {}).get('translatedText', '')
                if translation:
                    translated_chunks.append(translation)
            
            return " ".join(translated_chunks) if translated_chunks else "Translation failed for long text"
        else:
            # Normal translation for short texts
            params = {'q': text, 'langpair': 'en|am'}
            response = requests.get('https://api.mymemory.translated.net/get', params=params)
            response.raise_for_status()
            translation = response.json().get('responseData', {}).get('translatedText', '')
            return translation if translation else "Translation service unavailable"
    except Exception as e:
        raise RuntimeError(f"Translation error: {e}")

# Streamlit App UI
st.image("cancer.jpg", width=700)
st.markdown('<h3 style="text-align: center;">Medical Image Caption Generation in Text & Audio</h3>', unsafe_allow_html=True)

# Image Upload
uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

# Initialize session state
if 'caption' not in st.session_state:
    st.session_state.caption = None
if 'amharic_caption' not in st.session_state:
    st.session_state.amharic_caption = None
if 'audio_generated' not in st.session_state:
    st.session_state.audio_generated = False

audio_path = "generated_audio.wav"

if uploaded_image:
    image = Image.open(uploaded_image).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Generate Caption"):
        with st.spinner("Generating caption..."):
            try:
                st.session_state.caption = generate_caption(image)
                st.session_state.amharic_caption = None
                st.session_state.audio_generated = False
                st.success("Caption generated successfully!")
            except Exception as e:
                st.error(f"Error: {e}")

    if st.session_state.caption:
        st.markdown(f" **Generated Caption:** {st.session_state.caption}")

        # Translation section
        st.markdown("---")
        # st.subheader("Translation")
        
        if st.button("Translate to Amharic"):
            with st.spinner("Translating to Amharic..."):
                try:
                    st.session_state.amharic_caption = translate_to_amharic(st.session_state.caption)
                    if "LIMIT EXCEEDED" not in st.session_state.amharic_caption:
                        st.success("Translation completed!")
                except Exception as e:
                    st.error(f"Translation error: {e}")

        if st.session_state.amharic_caption:
            st.markdown(f" **Amharic Translation:** {st.session_state.amharic_caption}")

        # Audio section
        st.markdown("---")
        # st.subheader("Audio Generation")
        
        if st.button("Generate Audio"):
            with st.spinner("Creating audio..."):
                try:
                    text_to_audio_wav(st.session_state.caption, audio_path)
                    st.session_state.audio_generated = True
                    st.success("Audio generated successfully!")
                except Exception as e:
                    st.error(f"Audio generation error: {e}")

        if st.session_state.audio_generated:
            try:
                st.markdown("### Sound Wave Visualization:")
                audio_data, sr = librosa.load(audio_path)
                fig, ax = plt.subplots(figsize=(10, 4))
                librosa.display.waveshow(audio_data, sr=sr, ax=ax)
                ax.set_title("Waveform of the Generated Audio")
                ax.set_xlabel("Time (s)")
                ax.set_ylabel("Amplitude")
                st.pyplot(fig)

                st.audio(audio_path, format="audio/wav")

                with open(audio_path, "rb") as f:
                    st.download_button(
                        label="Download Audio",
                        data=f,
                        file_name="medical_caption.wav",
                        mime="audio/wav"
                    )
            except Exception as e:
                st.error(f"Audio display error: {e}")