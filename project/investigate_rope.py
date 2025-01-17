import torch
import seaborn as sns
import math
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F


def compute_freqs_cis(dim: int, t: torch.tensor, theta: float = 10000.0):
    """
    Compute the frequency tensor for complex exponentials (cis) with given dimensions.

    This function calculates a frequency tensor with complex exponentials using the given dimension 'dim'
    and the time 't'. The 'theta' parameter scales the frequencies.
    The returned tensor contains complex values in complex64 data type.

    Args:
        dim (int): Dimension of the frequency tensor.
        t (tensor): End index for precomputing frequencies.
        theta (float, optional): Scaling factor for frequency computation. Defaults to 10000.0.

    Returns:
        torch.Tensor: Precomputed frequency tensor with complex exponentials.
    """
    freqs = 1.0 / (
                theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    freqs = torch.outer(t, freqs).float()  # type: ignore
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    """
    Reshape frequency tensor for broadcasting it with another tensor.

    This function reshapes the frequency tensor to have the same shape as the target tensor 'x'
    for the purpose of broadcasting the frequency tensor during element-wise operations.

    Args:
        freqs_cis (torch.Tensor): Frequency tensor to be reshaped.
        x (torch.Tensor): Target tensor for broadcasting compatibility.

    Returns:
        torch.Tensor: Reshaped frequency tensor.

    Raises:
        AssertionError: If the frequency tensor doesn't match the expected shape.
        AssertionError: If the target tensor 'x' doesn't have the expected number of dimensions.
    """
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in
             enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(
        xq: torch.Tensor,
        xk: torch.Tensor,
        freqs_cis: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary embeddings to input tensors using the given frequency tensor.

    This function applies rotary embeddings to the given query 'xq' and key 'xk' tensors using the provided
    frequency tensor 'freqs_cis'. The input tensors are reshaped as complex numbers, and the frequency tensor
    is reshaped for broadcasting compatibility. The resulting tensors contain rotary embeddings and are
    returned as real tensors.

    Args:
        xq (torch.Tensor): Query tensor to apply rotary embeddings.
        xk (torch.Tensor): Key tensor to apply rotary embeddings.
        freqs_cis (torch.Tensor): Precomputed frequency tensor for complex exponentials.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Tuple of modified query tensor and key tensor with rotary embeddings.
    """
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)

def main():
    emb_dim = 768
    times = torch.arange(0, 6, 0.33)
    embs = torch.ones((1, times.shape[0], emb_dim))
    embs = embs.view(embs.shape[0], embs.shape[1], 1, embs.shape[2])
    freq_emb = compute_freqs_cis(emb_dim, times)
    k, v = apply_rotary_emb(embs, embs, freq_emb)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)
    scores = torch.matmul(k, v.transpose(2, 3)) / math.sqrt(emb_dim)
    scores = scores.squeeze()
    weight = torch.tensor([1., -1.])[None, None, ...]
    neighbouring_scores_1 = torch.diagonal(scores, offset=1)
    neighbouring_scores_2 = torch.diagonal(scores, offset=2)
    neighbouring_scores_3 = torch.diagonal(scores, offset=3)

    scores = scores.cpu().numpy()
    mask = np.zeros_like(scores)
    mask[np.triu_indices_from(mask, k=1)] = True
    with sns.axes_style("white"):
        ax = sns.heatmap(scores, mask=mask, square=True,  cmap="YlGnBu")
        ax.set_xlabel("Position in sequence")
        ax.set_title("Attention scores between query and key")
        plt.show()

    fig, ax = plt.subplots()
    for idx, r_score in enumerate((neighbouring_scores_1, neighbouring_scores_2, neighbouring_scores_3)):
        relative_distances = F.conv1d(r_score[None, :], weight).cpu().numpy().squeeze()
        ax.plot(list(range(relative_distances.shape[0])), relative_distances, label=f"{idx}")
    ax.set_xlabel("Position in sequence")
    ax.set_ylabel("Dot product")
    fig.suptitle("Relative dot product between query and key")
    fig.legend()
    fig.show()


if __name__ == '__main__':
    main()
