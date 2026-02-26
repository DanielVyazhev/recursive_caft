import torch


def compute_entropy_from_logits(logits: torch.Tensor) -> torch.Tensor:
    """
    Compute entropy from logits.

    Parameters:
    ----------
    logits : torch.Tensor
        Logits from the model.

    Returns:
    -------
    torch.Tensor
        Entropy values.
    """
    probabilities = torch.softmax(logits, dim=-1)
    log_probabilities = torch.log(probabilities + 1e-12)
    entropy = -torch.sum(probabilities * log_probabilities, dim=-1)
    return entropy
