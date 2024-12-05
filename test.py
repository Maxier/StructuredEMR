import torch

# 检查CUDA是否可用
is_cuda_available = torch.cuda.is_available()
if is_cuda_available:
    print("CUDA is available.")
else:
    print("CUDA is not available.")

# 获取CUDA版本
cuda_version = torch.version.cuda
print(f"CUDA version: {cuda_version}")