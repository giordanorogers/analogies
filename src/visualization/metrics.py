import matplotlib.pyplot as plt
import numpy as np
import torch
from typing import Dict, List, Union, Any

def plot_metric_by_layer(
    data: Dict[str, Union[List[float], torch.Tensor]],
    title: str = "",
    xlabel: str = "Layer",
    ylabel: str = "",
    styles: Dict[str, Dict[str, Any]] = None,
    figsize: tuple = (10, 6),
    dpi=300
):
    """
    Creates and displays a line plot of one or more metrics against model layers.
    Designed for plotting logit difference or accuracy across layers for different conditions.
    """
    if styles is None:
        styles = {}

    plt.style.use('seaborn-v0_8-bright')
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    for i, (label, values) in enumerate(data.items()):
        # Convert torch tensors to numpy arrays for plotting
        if isinstance(values, torch.Tensor):
            values = values.cpu().numpy()

        x_values = range(len(values))

        # Style Logic
        plot_kwargs = {
            'label': label
        }
        # Override with any provided styles
        if label in styles:
            plot_kwargs.update(styles[label])

        ax.plot(x_values, values, **plot_kwargs)

    # Final Touches
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.legend(title="")

    plt.tight_layout()
    plt.show()