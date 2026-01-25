"""Comprehensive GPU and environment test script.

Tests CUDA availability, Triton installation, and ModernBERT model loading
in both GPU and CPU modes.

Usage:
    python test_gpu.py              # Test with auto-detected device
    CUDA_VISIBLE_DEVICES="" python test_gpu.py  # Force CPU mode
"""
import sys


def print_section(title: str) -> None:
    """Print a section header."""
    print("\n" + "=" * 60)
    print(f" {title}")
    print("=" * 60)


def test_python_version() -> bool:
    """Test Python version (ModernBERT requires >= 3.10)."""
    print(f"Python version: {sys.version}")
    major, minor = sys.version_info[:2]
    if (major, minor) >= (3, 10):
        print("Python version: OK (>= 3.10 required)")
        return True
    else:
        print("WARNING: Python >= 3.10 required for ModernBERT")
        return False


def test_cuda() -> bool:
    """Test CUDA availability and properties."""
    import torch
    
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"cuDNN enabled: {torch.backends.cudnn.enabled}")
        print(f"cuDNN version: {torch.backends.cudnn.version()}")
        print(f"GPU count: {torch.cuda.device_count()}")
        
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"\nGPU {i}: {props.name}")
            print(f"  - Compute capability: {props.major}.{props.minor}")
            print(f"  - Total memory: {props.total_memory / 1e9:.2f} GB")
            print(f"  - Multi-processor count: {props.multi_processor_count}")
        
        # Test tensor allocation
        try:
            x = torch.randn(1000, 1000, device="cuda")
            y = x @ x.T  # Matrix multiply to test computation
            print(f"\nGPU tensor allocation: OK")
            print(f"GPU computation test: OK")
            del x, y
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"\nGPU test failed: {e}")
            return False
        
        return True
    else:
        print("\nNo CUDA GPU detected.")
        print("Possible reasons:")
        print("  - No NVIDIA GPU installed")
        print("  - CUDA drivers not installed")
        print("  - PyTorch installed without CUDA support")
        print("  - CUDA_VISIBLE_DEVICES set to empty")
        return False


def test_triton() -> bool:
    """Test Triton installation (required for torch.compile on GPU)."""
    try:
        import triton
        print(f"Triton version: {triton.__version__}")
        print("Triton: OK")
        return True
    except ImportError:
        print("Triton: NOT INSTALLED")
        print("  - Required for GPU acceleration with ModernBERT")
        print("  - Install with: pip install triton")
        print("  - Note: Triton only works on Linux with NVIDIA GPUs")
        return False


def test_transformers() -> bool:
    """Test transformers library version."""
    import transformers
    print(f"Transformers version: {transformers.__version__}")
    
    # Check if version is sufficient for ModernBERT
    version_parts = transformers.__version__.split(".")
    major, minor = int(version_parts[0]), int(version_parts[1])
    if (major, minor) >= (4, 48):
        print("Transformers version: OK (>= 4.48.0 required for ModernBERT)")
        return True
    else:
        print("WARNING: Transformers >= 4.48.0 required for ModernBERT")
        return False


def test_setfit() -> bool:
    """Test SetFit installation."""
    try:
        import setfit
        print(f"SetFit version: {setfit.__version__}")
        return True
    except ImportError:
        print("SetFit: NOT INSTALLED")
        return False


def test_model_loading() -> bool:
    """Test ModernBERT model loading with GPU/CPU compatibility."""
    try:
        from utils.gpu_utils import get_device_config, load_model
        
        print("\nTesting model loading with device configuration...")
        config = get_device_config()
        print(f"Device: {config.device}")
        print(f"Use AMP: {config.use_amp}")
        print(f"Batch size: {config.batch_size}")
        
        print("\nLoading model...")
        model = load_model(config)
        
        # Quick inference test
        test_sentence = "This is a test sentence for embedding generation."
        result = model.encode([test_sentence])
        
        print(f"Model inference: OK")
        print(f"Embedding dimension: {result.shape[-1]}")
        print(f"Embedding dtype: {result.dtype}")
        
        return True
        
    except Exception as e:
        print(f"\nModel loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print_section("Environment Test")
    
    results = {}
    
    # Basic tests
    results["python"] = test_python_version()
    
    print_section("PyTorch & CUDA")
    results["cuda"] = test_cuda()
    
    print_section("Triton")
    results["triton"] = test_triton()
    
    print_section("ML Libraries")
    results["transformers"] = test_transformers()
    results["setfit"] = test_setfit()
    
    print_section("Model Loading Test")
    results["model"] = test_model_loading()
    
    # Summary
    print_section("Summary")
    
    all_passed = all(results.values())
    critical_passed = results.get("python", False) and results.get("model", False)
    
    for test_name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {test_name}: {status}")
    
    print()
    if results.get("cuda", False):
        print("Mode: GPU (with CUDA acceleration)")
        if not results.get("triton", False):
            print("  WARNING: Triton not installed - some optimizations disabled")
    else:
        print("Mode: CPU (no GPU acceleration)")
        print("  Note: Training will be slower on CPU")
    
    print()
    if critical_passed:
        print("Status: READY - You can run the training scripts")
    else:
        print("Status: NOT READY - Please fix the failed tests above")
    
    return 0 if critical_passed else 1


if __name__ == "__main__":
    sys.exit(main())
