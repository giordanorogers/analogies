import matplotlib.pyplot as plt
import numpy as np
import torch
from nnsight import LanguageModel
from typing import Literal

def patching_heatmap(
    model: LanguageModel,
    patched_activations: list,
    source_input_ids: list,
    base_input_ids: list,
    kind: Literal["residual", "mlp", "attention"] = "residual",
    window_size: int = 1,
) -> None:
    """
    Visualize the patching heatmap.
    """
    # Create a 2D array: layers as rows, tokens as columns, then transpose
    scores = np.array(patched_activations).T  # Transpose so tokens are rows, layers are columns

    # Decode tokens to actual text
    source_tokens = [model.tokenizer.decode([token_id]) for token_id in source_input_ids]
    base_tokens = [model.tokenizer.decode([token_id]) for token_id in base_input_ids]

    # Create token labels
    tokens = []
    for idx, (source_tok, base_tok) in enumerate(zip(source_tokens, base_tokens)):
        if source_tok == base_tok:
            tokens.append(f'"{source_tok}"')
        else:
            tokens.append(f'"{source_tok}"/"{base_tok}"')

    # We will truncate to the 0 to 1 range to prioritize positive values
    # but this is the full actual range.
    print(f"Data range: min={scores.min():.4f}, max={scores.max():.4f}")

    plt.rcdefaults()
    with plt.rc_context(
        rc={
            "font.family": "Times New Roman",
            "font.size": 6,
        }
    ):
        # Set figure size
        fig, ax = plt.subplots(
            figsize=(
                6,
                len(tokens) * 0.08 + 1.8
            ),
            dpi=200
        )
        
        # Scale range
        scale_kwargs = {
            "vmin": 0,
            "vmax": 1
        }

        if kind == "residual":
            title = f"Indirect Effects of Residual Layers"
            color_map = "Purples"
        elif kind == "mlp":
            if window_size > 1:
                title = f"Indirect Effects of MLP Layers (Window Size {window_size})"
            else:
                title = f"Indirect Effects of MLP Layers"
            color_map = "Greens"
        elif kind == "attention":
            title = f"Indirect Effects of Attention Layers"
            color_map = "Reds"

        heatmap = ax.pcolor(
            scores,
            cmap=color_map,
            **scale_kwargs,
        )

        ax.invert_yaxis()
        
        # Y-axis: token labels (rows)
        ax.set_yticks([0.5 + i for i in range(scores.shape[0])])  # Number of tokens
        ax.set_yticklabels(tokens)
        
        # X-axis: layer labels (columns)
        num_layers = scores.shape[1]
        tick_indices = np.arange(0, num_layers, 5)
        ax.set_xticks(tick_indices + 0.5)  # Number of layers
        ax.set_xticklabels(tick_indices)

        title = title
        ax.set_title(title)

        ax.set_xlabel("Layer")
        #ax.set_ylabel("Tokens")

        color_scale = plt.colorbar(heatmap)
        color_scale.ax.set_title(
            f"Normalized Score",
            y=-0.12,
            fontsize=8
        )

        plt.tight_layout()
        plt.show()
