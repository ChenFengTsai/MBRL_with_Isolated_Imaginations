import torch

# Test basic tensor operations on GPU
try:
    # Create tensor on GPU
    x = torch.randn(1000, 1000).cuda()
    y = torch.randn(1000, 1000).cuda()
    
    # Test computation
    z = torch.mm(x, y)
    print("✅ Basic GPU operations work!")
    print(f"GPU device: {torch.cuda.get_device_name(0)}")
    
except Exception as e:
    print(f"❌ GPU operations failed: {e}")