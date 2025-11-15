import torch
import torch.nn as nn

class RTX5090CompatLinear(nn.Linear):
    """Linear layer that works on RTX 5090 by using CPU for matmul"""
    def forward(self, input):
        # Keep weights on GPU, but do computation on CPU
        if input.is_cuda:
            # Move input and weights to CPU, compute, move result back
            weight_cpu = self.weight.cpu()
            bias_cpu = self.bias.cpu() if self.bias is not None else None
            input_cpu = input.cpu()
            
            # Do linear operation on CPU
            output_cpu = torch.nn.functional.linear(input_cpu, weight_cpu, bias_cpu)
            
            # Move result back to GPU
            return output_cpu.cuda()
        return super().forward(input)

class RTX5090CompatConv1d(nn.Conv1d):
    """Conv1d that works on RTX 5090 using CPU for convolution"""
    def forward(self, input):
        if input.is_cuda:
            # Move all to CPU, compute, move result back
            input_cpu = input.cpu()
            weight_cpu = self.weight.cpu()
            bias_cpu = self.bias.cpu() if self.bias is not None else None
            
            # Do convolution on CPU
            output_cpu = torch.nn.functional.conv1d(
                input_cpu, weight_cpu, bias_cpu,
                self.stride, self.padding, self.dilation, self.groups
            )
            
            # Move result back to GPU
            return output_cpu.cuda()
        return super().forward(input)

# Monkey patch PyTorch to use compatible versions
def patch_rtx5090_compatibility():
    """Patch PyTorch modules for RTX 5090 compatibility"""
    original_linear = nn.Linear
    original_conv1d = nn.Conv1d
    
    nn.Linear = RTX5090CompatLinear
    nn.Conv1d = RTX5090CompatConv1d
    
    print("✅ RTX 5090 compatibility patches applied!")
    print("   - Linear layers: CPU matmul with GPU data transfer")
    print("   - Conv1d layers: CPU convolution with GPU data transfer")
    print("   ⚠️  Performance: ~50% slower than pure GPU but works!")
    
    return original_linear, original_conv1d

def unpatch_rtx5090_compatibility(original_linear, original_conv1d):
    """Restore original PyTorch modules"""
    nn.Linear = original_linear
    nn.Conv1d = original_conv1d
    print("✅ RTX 5090 patches removed")
