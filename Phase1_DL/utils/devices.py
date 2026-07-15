"""Device management utilities."""

import torch


def get_device() -> torch.device:
    """Get available device (CUDA if available, else CPU).

    Returns:
        torch.device instance.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        return device
    return torch.device("cpu")


def get_device_name() -> str:
    """Get device name as string.

    Returns:
        Device name (e.g., 'cuda', 'cpu').
    """
    if torch.cuda.is_available():
        return f"cuda ({torch.cuda.get_device_name(0)})"
    return "cpu"
