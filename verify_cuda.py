import torch

def verify_cuda():
    if torch.cuda.is_available():
        print(f"CUDA is available. PyTorch version: {torch.__version__}")
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU device name: {torch.cuda.get_device_name(0)}")
    else:
        print("CUDA is not available. The application will run on CPU, which may be very slow.")

if __name__ == "__main__":
    verify_cuda()