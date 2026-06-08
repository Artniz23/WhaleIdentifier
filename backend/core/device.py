import torch


def get_device() -> torch.device:
    """
    Возвращает доступное устройство для инференса.
    """
    return torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
