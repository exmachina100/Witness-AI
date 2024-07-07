import streamlit as st
import torch
import os
from PIL import Image
from model import load_models, get_model_list
from utils import get_canny_image, generate_image

# Set page configuration
st.set_page_config(layout="wide", page_title="WITNESS V 1.0", page_icon="🎨")

# Custom CSS for light theme
st.markdown("""
    <style>
        .stApp {
            max-width: 100%;
            padding-top: 2rem;
        }
        .stButton>button {
            width: 100%;
            background-color: #4CAF50;
            color: #ffffff;
            border: none;
            border-radius: 5px;
            padding: 10px;
            cursor: pointer;
        }
        .stButton>button:hover {
            background-color: #45a049;
        }
        .stSelectbox, .stSlider, .stTextArea {
            background-color: #f9f9f9;
            color: #333333;
            border: 1px solid #ddd;
            border-radius: 5px;
            padding: 8px;
            margin-bottom: 0.5rem;
        }
        .title {
            font-size: 2rem;
            font-weight: bold;
            text-align: left;
            color: #333333;
            margin-bottom: 1rem;
        }
        .header-container {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 1rem;
        }
        .image-container {
            text-align: center;
            margin-bottom: 1rem;
        }
        .image-container img {
            max-width: 100%;
            border: 1px solid #ddd;
            border-radius: 5px;
        }
        .small-text {
            font-size: 0.8rem;
            color: #888;
        }
        .image-row {
            display: flex;
            justify-content: space-between;
            gap: 1rem;
        }
        .image-column {
            flex: 1;
        }
        .slider-container {
            margin-bottom: 0.5rem;
        }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_cached_models(sd_model_path, lora_model_path):
    return load_models(sd_model_path, lora_model_path)

def main():
    # Header with title and model selection
    st.markdown("<h1 class='title'>WITNESS V 1.0</h1>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        sd_models_dir = "models/stable_diffusion"
        sd_models = get_model_list(sd_models_dir)
        selected_sd_model = st.selectbox("Select Stable Diffusion model", sd_models)
    
    with col2:
        lora_models_dir = "models/lora"
        lora_models = get_model_list(lora_models_dir)
        selected_lora_model = st.selectbox("Select LoRA model (optional)", ["None"] + lora_models)
    
    sd_model_path = os.path.join(sd_models_dir, selected_sd_model)
    lora_model_path = os.path.join(lora_models_dir, selected_lora_model) if selected_lora_model != "None" else None

    pipe = load_cached_models(sd_model_path, lora_model_path)

    # Prompt and Generation Controls
    prompt = st.text_area("Prompt", "A detailed rendering of the sketch", height=50)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        num_inference_steps = st.slider("Inference Steps", 20, 100, 30, 1)
    with col2:
        guidance_scale = st.slider("Guidance Scale", 1.0, 20.0, 7.5, 0.5)
    with col3:
        clip_skip = st.slider("Clip Skip", 1, 12, 1, 1)
    with col4:
        generate_button = st.empty()

    # Main content layout in a row
    image_col1, image_col2, image_col3 = st.columns(3)

    with image_col1:
        st.subheader("Input")
        uploaded_file = st.file_uploader("Choose a sketch image...", type=["png", "jpg", "jpeg"])
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Uploaded Sketch", use_column_width=True)

    with image_col2:
        st.subheader("Canny Edge Detection")
        if uploaded_file is not None:
            low_threshold = st.slider("Low Threshold", 50, 300, 100, 10)
            high_threshold = st.slider("High Threshold", 100, 500, 200, 10)
            canny_image = get_canny_image(image, low_threshold, high_threshold)
            st.image(canny_image, caption="Canny Edge Detection", use_column_width=True)

    with image_col3:
        st.subheader("Generated Image")
        if uploaded_file is not None:
            if generate_button.button("Generate Render", key="generate"):
                with st.spinner("Generating render..."):
                    rendered_image = generate_image(pipe, prompt, canny_image, num_inference_steps, guidance_scale, clip_skip)
                    st.image(rendered_image, caption="Generated Render", use_column_width=True)
                    st.download_button("Download Rendered Image", rendered_image.tobytes(), "rendered_image.png", "image/png")
                    st.success("Render generated successfully!")

    # Instructions
    with st.expander("How it works"):
        st.markdown("""
        1. **Select models**: Choose a Stable Diffusion model and an optional LoRA model at the top.
        2. **Upload a sketch**: Use the file uploader to input your sketch.
        3. **Adjust edge detection**: Fine-tune the Canny edge detection thresholds for optimal results.
        4. **Set render options**: Enter a prompt describing the desired output and adjust inference steps, guidance scale, and clip skip.
        5. **Generate render**: Click the "Generate Render" button to create the rendered image.
        """)
        
        st.markdown("<p class='small-text'>Note: This demo uses the Canny edge detection ControlNet model with Stable Diffusion. The quality of the output depends on the input sketch, prompt, and parameter settings.</p>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()