"""Simple GPU test script to verify CUDA availability."""
import torch

print("=" * 50)
print("GPU/CUDA Test")
print("=" * 50)

# Check CUDA availability
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU count: {torch.cuda.device_count()}")
    print(f"Current GPU: {torch.cuda.current_device()}")
    print(f"GPU name: {torch.cuda.get_device_name(0)}")
    
    # Test tensor on GPU
    x = torch.tensor([1.0, 2.0, 3.0]).cuda()
    print(f"\nTest tensor on GPU: {x}")
    print(f"Tensor device: {x.device}")
    print("\nGPU is working correctly!")
else:
    print("\nNo CUDA GPU detected.")
    print("Possible reasons:")
    print("  - No NVIDIA GPU installed")
    print("  - CUDA drivers not installed")
    print("  - PyTorch installed without CUDA support")
    print("\nTo install PyTorch with CUDA, visit: https://pytorch.org/get-started/locally/")

print("=" * 50)
