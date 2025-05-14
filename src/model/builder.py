import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import config


def conv_block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )

def residual_block(channels):
    return nn.Sequential(
        conv_block(channels, channels),
        nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(channels),
    )

class PolicyHead(nn.Module):
    def __init__(self, input_channels, output_size):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(input_channels, 73, kernel_size=1),  # Matches saved shape (73, 256, 1, 1)
            nn.BatchNorm2d(73),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(73 * config.INPUT_SHAPE[1] * config.INPUT_SHAPE[2], output_size),
        )

    def forward(self, x):
        return self.head(x)

class ValueHead(nn.Module):
    def __init__(self, input_channels, output_size):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(input_channels, 1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(config.INPUT_SHAPE[1] * config.INPUT_SHAPE[2], 512),  # Matches saved shape (512, 64)
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, output_size),  # Extra layer (head.7 in state_dict)
            nn.Tanh()
        )

    def forward(self, x):
        return self.head(x)

class RLModel(nn.Module):
    def __init__(self, input_shape, output_shape):
        super().__init__()
        self.input_shape = input_shape
        self.output_shape = output_shape

        c, _, _ = input_shape
        self.conv1 = conv_block(c, config.CONV_FILTERS)
        self.residuals = nn.Sequential(*[
            residual_block(config.CONV_FILTERS) for _ in range(config.N_RESIDUAL_BLOCKS)
        ])
        self.policy_head = PolicyHead(config.CONV_FILTERS, output_shape[0])
        self.value_head = ValueHead(config.CONV_FILTERS, output_shape[1])
        self.load_state_dict(torch.load(config.SL_MODEL_PATH, map_location=config.DEVICE))

    def forward(self, x):
        x = self.conv1(x)
        for res in self.residuals:
            x = F.relu(res(x) + x)
        return self.policy_head(x), self.value_head(x)
