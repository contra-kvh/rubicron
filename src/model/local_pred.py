import torch
import numpy as np

from utils import config
from model.builder import RLModel

def predict_local(model: RLModel, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    model.to(config.DEVICE)
    inputs = torch.from_numpy(inputs.astype(np.float32)).to(device=config.DEVICE)

    with torch.no_grad():
        policy_logits, value_pred = model(inputs)
    
    return policy_logits, value_pred
