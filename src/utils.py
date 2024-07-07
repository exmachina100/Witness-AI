import cv2
import numpy as np
from PIL import Image
from controlnet_aux import CannyDetector

canny_detector = CannyDetector()

def get_canny_image(image, low_threshold, high_threshold):
    image_np = np.array(image)
    canny_image = canny_detector(image_np, low_threshold, high_threshold)
    return Image.fromarray(canny_image)

def generate_image(pipe, prompt, canny_image, num_inference_steps, guidance_scale, clip_skip):
    # Assuming the `pipe` object has a method to set the clip skip value
    if hasattr(pipe, 'set_skip_clip_layers'):
        pipe.set_skip_clip_layers(clip_skip)
    else:
        # If pipe does not have this method, log a warning or handle as needed
        print("Warning: Pipe object does not support clip skip. Continuing without setting clip skip.")

    output = pipe(
        prompt,
        canny_image,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
    )
    return output.images[0]
