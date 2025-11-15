# Test RTX 5090 Compatibility Layer
import sys
sys.path.insert(0, 'backend')

from app.services.rtx5090_compat import patch_rtx5090_compatibility
import torch

print("🧪 Testing RTX 5090 Compatibility Layer\n")

# Apply patches
print("🔧 Applying compatibility patches...")
orig_linear, orig_conv = patch_rtx5090_compatibility()

print("\n🧪 Test 1: Linear layer with compat patches...")
try:
    layer = torch.nn.Linear(100, 50).cuda()
    input_data = torch.randn(10, 100).cuda()
    output = layer(input_data)
    print(f"✅ SUCCESS! Output shape: {output.shape}, on device: {output.device}")
except Exception as e:
    print(f"❌ Failed: {e}")

print("\n🧪 Test 2: Conv1d with compat patches...")
try:
    conv = torch.nn.Conv1d(1, 10, 3).cuda()
    input_audio = torch.randn(1, 1, 1000).cuda()
    output = conv(input_audio)
    print(f"✅ SUCCESS! Output shape: {output.shape}, on device: {output.device}")
except Exception as e:
    print(f"❌ Failed: {e}")

print("\n🧪 Test 3: Multiple operations...")
try:
    x = torch.randn(32, 128).cuda()
    linear1 = torch.nn.Linear(128, 64).cuda()
    linear2 = torch.nn.Linear(64, 32).cuda()
    
    out1 = linear1(x)
    out2 = linear2(out1)
    
    print(f"✅ SUCCESS! Chain of operations works!")
    print(f"   Final shape: {out2.shape}, device: {out2.device}")
except Exception as e:
    print(f"❌ Failed: {e}")

print("\n🎯 If all tests passed, voice synthesis should work on RTX 5090!")
