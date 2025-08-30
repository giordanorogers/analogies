import torch
from nnsight import LanguageModel
import circuitsvis as cv
from circuitsvis.tokens import colored_tokens
from IPython.display import display, HTML
from typing import List, Union

def visualize_token_attention(
    attn_output: torch.Tensor,
    tokens: list[str],
    layer_indices: Union[int, List[int]],
    head_indices: Union[int, List[int]],
    query_idx: int = -1,
    start_from: int = 1,
):
    """
    Display colorized token-by-token attention for a specific head.
    """
    # Standardize index lists
    if isinstance(layer_indices, int):
        layer_indices = [layer_indices]
    if isinstance(head_indices, int):
        head_indices = [head_indices]

    assert len(tokens) == attn_output[0].shape[-1], (
        f"Number of tokens in the prompt ({len(tokens)}) does not match the number of tokens in the attention output ({attn_output[0].shape[-1]})"
    )

    for layer_idx in layer_indices:
        for head_idx in head_indices:
            # Set the query token
            query_token = attn_output[layer_idx][:, head_idx, query_idx, start_from:][0]

            display(
                f"Layer {layer_idx}. Head {head_idx}",
                colored_tokens(
                    tokens=tokens[start_from:],
                    values=query_token
                )
            )