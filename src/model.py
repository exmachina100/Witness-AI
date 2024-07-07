import os
import torch
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from safetensors.torch import load_file

def load_models(sd_model_path, lora_model_path=None):
    # Load ControlNet
    controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny", torch_dtype=torch.float16)
    
    # Load Stable Diffusion model
    if sd_model_path.endswith('.safetensors'):
        state_dict = load_file(sd_model_path, device="cpu")
    else:
        state_dict = torch.load(sd_model_path, map_location="cpu")

    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5", 
        controlnet=controlnet, 
        torch_dtype=torch.float16,
        state_dict=state_dict
    )
    
    if lora_model_path:
        pipe.unet.load_attn_procs(lora_model_path)
    
    pipe = pipe.to("cuda")
    return pipe

def get_model_list(directory):
    return [f for f in os.listdir(directory) if f.endswith(('.safetensors', '.ckpt'))]