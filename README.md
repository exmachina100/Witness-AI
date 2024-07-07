# Sketch to Render Application

This application uses ControlNet and Stable Diffusion to transform sketches into rendered images.

## Requirements

- Python 3.10 or later
- NVIDIA GPU with CUDA 11.8 support (for optimal performance)

## Setup

1. Clone this repository
2. Make sure you have CUDA 11.8 installed on your system if you have an NVIDIA GPU
3. Run the setup script:
   - On Windows: Double-click `setup.bat` or run it from the command prompt
   - On macOS/Linux: Open a terminal and run `sh setup.sh`
4. Place your Stable Diffusion models (safetensors or ckpt) in the `models/stable_diffusion/` directory
5. Place your LoRA models (safetensors) in the `models/lora/` directory

The setup script will install all necessary dependencies, including PyTorch with CUDA support and the ControlNet model.

## Running the Application

- On Windows: Double-click `run.bat` or run it from the command prompt
- On macOS/Linux: Open a terminal and run `sh run.sh`

The script will verify CUDA availability before starting the application.

## Usage

1. Select a Stable Diffusion model and an optional LoRA model
2. Upload a sketch image
3. Adjust the Canny edge detection thresholds
4. Set the render options (prompt, inference steps, guidance scale)
5. Click "Generate Render" to create the rendered image

Note: While the application can run on CPU, a CUDA-capable GPU is strongly recommended for reasonable performance.