import streamlit as st
from PIL import Image
import io
import torch
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from diffusers.utils import load_image
import cv2
import numpy as np

@st.cache_resource
def load_models():
    controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny")
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5", controlnet=controlnet, torch_dtype=torch.float16
    )
    pipe = pipe.to("cuda")
    return pipe

def get_canny_image(image, low_threshold, high_threshold):
    image = np.array(image)
    image = cv2.Canny(image, low_threshold, high_threshold)
    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    canny_image = Image.fromarray(image)
    return canny_image

def generate_image(pipe, prompt, canny_image, num_inference_steps, guidance_scale):
    output = pipe(
        prompt,
        canny_image,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
    )
    return output.images[0]

st.set_page_config(layout="wide")
st.title("Sketch to Render Dashboard with ControlNet")

pipe = load_models()

col1, col2 = st.columns(2)

with col1:
    st.header("Input Sketch")
    uploaded_file = st.file_uploader("Choose a sketch image...", type=["png", "jpg", "jpeg"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Sketch", use_column_width=True)
        
        st.subheader("Canny Edge Detection")
        low_threshold = st.slider("Low Threshold", 50, 300, 100)
        high_threshold = st.slider("High Threshold", 100, 500, 200)
        canny_image = get_canny_image(image, low_threshold, high_threshold)
        st.image(canny_image, caption="Canny Edge Detection", use_column_width=True)
        
        st.subheader("Render Options")
        prompt = st.text_input("Prompt", "A detailed rendering of the sketch")
        num_inference_steps = st.slider("Number of Inference Steps", 20, 100, 30)
        guidance_scale = st.slider("Guidance Scale", 1.0, 20.0, 7.5)

        if st.button("Generate Render"):
            with st.spinner("Generating render..."):
                rendered_image = generate_image(pipe, prompt, canny_image, num_inference_steps, guidance_scale)
                
                with col2:
                    st.header("Generated Render")
                    st.image(rendered_image, caption="Generated Render", use_column_width=True)
                    st.success("Render generated successfully!")

st.markdown("""
## How it works

1. **Upload a sketch**: Use the file uploader to input your sketch.
2. **Canny Edge Detection**: Adjust the thresholds to get the desired edge detection.
3. **Set render options**: Enter a prompt describing the desired output and adjust inference steps and guidance scale.
4. **Generate render**: Click the button to process your sketch and create a rendered image using Stable Diffusion with ControlNet.

Note: This demo uses the Canny edge detection ControlNet model with Stable Diffusion. The quality of the output depends on the input sketch, prompt, and parameter settings.
""")