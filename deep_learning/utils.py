import torch


def device() -> torch.device:
    """
    Return the appropriate device to run the model on (GPU if available, else CPU).
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_model(model: torch.nn.Module, path: str) -> None:
    """
    Save a PyTorch model to disk.
    """
    torch.save(model.state_dict(), path)


def load_model(model: torch.nn.Module, path: str) -> None:
    """
    Load a PyTorch model from disk.
    """
    model.load_state_dict(torch.load(path, map_location=device()))
    model.to(device())
