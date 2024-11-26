import torch
import torch.nn as nn

"""
MLP Mixer PyTorch implementation, adapted from the code provided in the
paper:
https://arxiv.org/pdf/2105.01601
"""

class MlpBlock(nn.Module):
    def __init__(self, mlp_dim: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.linear1 = nn.Linear(mlp_dim, mlp_dim)
        self.gelu = nn.GELU()
        self.linear2 = nn.Linear(mlp_dim, mlp_dim)

    def forward (self, x):
        y = self.linear1(x)
        y = self.gelu(y)
        return self.linear2(y)


class MixerBlock(nn.Module):
    def __init__(self, tokens_mlp_dim: int, channels_mlp_dim: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tokens_mlp_dim: int = tokens_mlp_dim
        self.channels_mlp_dim: int = channels_mlp_dim
        self.norm1 = nn.LayerNorm(channels_mlp_dim)
        self.mlp_block1 = MlpBlock(tokens_mlp_dim)
        self.norm2 = nn.LayerNorm(channels_mlp_dim)
        self.mlp_block2 = MlpBlock(channels_mlp_dim)

    def forward(self, x):
        y = self.norm1(x)
        y = torch.swapaxes (y, 1, 2)
        y = self.mlp_block1(y)
        y = torch.swapaxes(y, 1, 2)
        x = x + y
        y = self.norm2(x)
        return x + self.mlp_block2(y)


class MlpMixer(nn.Module):
    def __init__(self, num_classes: int, num_blocks: int, patch_size: int,
                 hidden_dim: int, tokens_mlp_dim: int, channels_mlp_dim: int,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_classes : int = num_classes
        self.num_blocks : int = num_blocks
        self.patch_size : int = patch_size
        self.hidden_dim : int = hidden_dim
        self.tokens_mlp_dim : int = tokens_mlp_dim
        self.channels_mlp_dim : int = channels_mlp_dim

        s = self.patch_size
        self.conv = nn.Conv2d(
            in_channels=3,
            out_channels=hidden_dim,
            kernel_size=(s, s),
            stride=(s, s)
        )
        mixer_blocks = []
        for _ in range(self.num_blocks):
            mixer_blocks.append(
                MixerBlock(self.tokens_mlp_dim, self.channels_mlp_dim)
            )
        self.mixer_blocks = nn.Sequential(*mixer_blocks)
        self.head_norm = nn.LayerNorm(channels_mlp_dim)
        self.head = nn.Linear(channels_mlp_dim, num_classes)

        self._init_weights_zero(self.head)

    def forward(self, x):
        x = self.conv( x )
        x = torch.einops.rearrange(x,"nhwc->n(hw)c")
        x = self.mixer_blocks(x)
        x = self.head_norm(x)
        x = torch.mean(x, dim=1)
        return self.head(x)

    @staticmethod
    def _init_weights_zero(m):
        """Initialize new layers with zeros.
        """
        if isinstance(m, (nn.Linear, nn.Conv3d)):
            nn.init.zeros_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
