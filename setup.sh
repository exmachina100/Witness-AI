#!/bin/bash
python3 -m venv venv
source venv/bin/activate

echo "Installing PyTorch with CUDA 11.8 support..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

echo "Installing other requirements..."
pip install -r requirements.txt

echo "Downloading ControlNet model..."
python -c "from diffusers import ControlNetModel; ControlNetModel.from_pretrained('lllyasviel/sd-controlnet-canny', use_auth_token=True)"

echo "Setup complete. Run 'sh run.sh' to start the application."