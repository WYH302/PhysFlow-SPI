import torch

print(
    f"cuda={torch.cuda.is_available()} "
    f"device={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none'} "
    f"torch={torch.__version__}"
)
