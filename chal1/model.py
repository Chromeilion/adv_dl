import torch.nn as nn


class MLPMixer(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim):

