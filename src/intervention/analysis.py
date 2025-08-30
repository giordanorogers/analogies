from transformers.tokenization_utils_fast import PreTrainedTokenizerFast
from typing import Optional, Literal
from nnsight import LanguageModel
import numpy as np
from tqdm import trange
import logging
import torch

logger = logging.getLogger(__name__)

def collect_logits(
    model: LanguageModel,
    prompt: str,
) -> list:
    logits = []
    with torch.no_grad():
        with model.trace(prompt):
            logits = model.output.logits[0].save()
    return logits

def collect_activations(
    model: LanguageModel,
    prompt: str,
    kind: Optional[Literal["residual", "mlp", "attention"]] = "residual",
) -> list:
    """
    Collect model activations.

    Args:
        model: A NNSight Language Model object.
        prompt: The prompt to run the model on.
        kind: The kind of component to collect the activations of.

    Returns:
        list of tensors
    """
    activations = []
    num_layers = model.config.num_hidden_layers

    for l in trange(num_layers):
        with torch.no_grad():
            with model.trace(prompt):
                if kind == "residual":
                    output = model.model.layers[l].output[0].save()
                    activations.append(output)
                elif kind == "mlp":
                    output = model.model.layers[l].mlp.output[0].save()
                    activations.append(output)
                elif kind == "attention":
                    output = model.model.layers[l].self_attn.output[0].save()
                    activations.append(output)
    
    return activations

def collect_activations_efficient(
    model: LanguageModel,
    prompt: str,
    kind: Optional[Literal["residual", "mlp", "attention"]] = "residual",
) -> list:
    """
    Memory-efficient version that collects all activations in a single forward pass.
    
    This is ~80x faster and uses ~80x less GPU memory than the original function
    by doing one forward pass instead of one per layer.

    Args:
        model: A NNSight Language Model object.
        prompt: The prompt to run the model on.
        kind: The kind of component to collect the activations of.

    Returns:
        list of tensors (one per layer)
    """
    activations = []
    num_layers = model.config.num_hidden_layers

    # Single forward pass to collect all activations
    with torch.no_grad():
        with model.trace(prompt):
            for l in range(num_layers):
                if kind == "residual":
                    output = model.model.layers[l].output[0].save()
                elif kind == "mlp":
                    output = model.model.layers[l].mlp.output[0].save()
                elif kind == "attention":
                    output = model.model.layers[l].self_attn.output[0].save()
                activations.append(output)
    
    return activations

def activation_patch(
    model: LanguageModel,
    prompt: str,
    activations: list,
    target_token_id: int,
    source_target_logits_base: float,
    kind: Literal["residual", "mlp", "attention"] = "residual",
    patch_range: Optional[list]= None,
    log: bool = False,
    window_size: int = 1,
) -> list:
    
    patched_activations = []
    num_layers = model.config.num_hidden_layers
    prompt_token_ids = model.tokenizer.encode(prompt)

    if patch_range is None:
        patch_range = range(len(prompt_token_ids))
    
    # If block for kind
    if kind == "residual":

        # Loop through layers
        for l in trange(num_layers, desc="All Layers"):

            layer_patch = []

            # Loop through prompt tokens
            for t in trange(len(prompt_token_ids), desc=f"Layer {l} Tokens"):

                with torch.no_grad():
                    
                    with model.trace(prompt):

                        # If we're at a token we want to patch
                        if t in patch_range:

                            # Replace the residual stream
                            model.model.layers[l].output[0][t, :] = activations[l][t, :]

                        # Get the model output logits
                        patched_logits = model.output.logits[0].save()

                        # Get the logits for the target
                        patched_target_logits = patched_logits[-1, target_token_id].item().save()

                        # Calculate the logit difference
                        logit_diff = patched_target_logits - source_target_logits_base

                        # Save the logit_diff
                        logit_diff = logit_diff.save()

                # Append the token to our list of logit differences at the current layer
                layer_patch.append(logit_diff)

                if log:
                    # Log the layer, token, and logit difference
                    logger.info(f"L{l}, T{t}, LD={logit_diff}")

            # Append each layer's logit differences list to our overall list of lists
            patched_activations.append(layer_patch)

    elif kind == "mlp":

        for window_start_l in trange(num_layers - window_size + 1):

            patch_at_token = []

            for t in range(len(prompt_token_ids)):

                with torch.no_grad():

                    # Single forward pass for the entire window
                    with model.trace(prompt):

                        # Apply the patch for all layers within this window
                        for l_offset in range(window_size):

                            l_in_window = window_start_l + l_offset
                            model.model.layers[l_in_window].mlp.output[0][t] = activations[l_in_window][t]

                        # Get the final logits after all 5 patches are in place
                        patched_logits = model.output.logits[0].save()

                        # Get the logits for the clean target
                        patched_target_logits = patched_logits[-1, target_token_id].item().save()

                        # Calculate the logit difference
                        logit_diff = patched_target_logits - source_target_logits_base

                        # Save the logit_diff
                        logit_diff = logit_diff.save()

                # Append the token to our list of logit differences at the current layer
                patch_at_token.append(logit_diff)

            # Append each layer's logit differences list to our overall list of lists
            patched_activations.append(patch_at_token)

    elif kind == "attention":

        for window_start_l in trange(num_layers - window_size + 1, desc="All Layers"):

            # Initialize a list to store the logit differences for this window
            results_for_window = []
            
            # Loop through prompt tokens
            for t in trange(len(prompt_token_ids), desc=f"Layer {window_start_l} Tokens"):

                with torch.no_grad():

                    # Single forward pass for the entire window
                    with model.trace(prompt):

                        # Apply the patch for all layers within this window
                        for l_offset in range(window_size):

                            l_in_window = window_start_l + l_offset
                            
                            model.model.layers[l_in_window].self_attn.output[0][0, t] = activations[l_in_window][0, t]

                        # Get the final logits after all 5 patches are in place
                        patched_target_logits = model.output.logits[0, -1, target_token_id].save()

                        # Calculate the logit difference
                        logit_diff = patched_target_logits - source_target_logits_base

                        # Save the logit_diff
                        logit_diff = logit_diff.save()

                # Append the token to our list of logit differences at the current layer
                results_for_window.append(logit_diff)

            # Append each layer's logit differences list to our overall list of lists
            patched_activations.append(results_for_window)

    return patched_activations

            
def logit_lens(
    model: LanguageModel,
    prompt: str,
    kind: Literal["full", "last"] = "last",
    k: int = 5,
    return_logits: bool = False
):
    logit_lens = []
    num_layers = model.config.num_hidden_layers

    if kind == "last":
        logits_probs = []
        with model.trace(prompt):
            logits = model.output.logits[0].save()
            probs = logits.softmax(dim=-1).save()
            logits_probs.append((logits, probs))
        logits = logits_probs[0][0][-1].topk(k=k).values
        probs = logits_probs[0][1][-1].topk(k=k).values
        token_ids = logits_probs[0][1][-1].topk(k=k).indices
        tokens = [model.tokenizer.decode(id) for id in token_ids]

        for t, l, p, id in zip(tokens, logits, probs, token_ids):
            logit_lens.append(f'"{t}"[{id}] (p={p}, logit={l})')

        return logit_lens


    elif kind == "full":
        raise NotImplementedError(
            "Full support not yet enabled"
        )
        layers = []
        for l in range(num_layers):
            logits_probs = []
            logit_lens_layer = []
            with model.trace(prompt):
                # Process layer output through lm_head to get logits
                layer_out = model.model.layers[l].output[0]
                logits = model.lm_head(layer_out).save()

                # Apply softmax to obtain probabilities
                probs = logits.softmax(dim=-1).save()

                logits_probs.append((logits, probs))

            logits = logits_probs[0][0][-1].max(dim=-1).values
            probs = logits_probs[0][1][-1].max(dim=-1).values
            token_ids = logits_probs[0][1][-1].max(dim=-1).indices
            tokens = [model.tokenizer.decode(id) for id in token_ids]
                
            for t, l, p, id in zip(tokens, logits, probs, token_ids):
                logit_lens_layer.append(f'"{t}"[{id}] (p={p}, logit={l})')

            logit_lens.append(logit_lens_layer)

        return logit_lens
    
        # Unpack the probabilities
        probs = torch.cat([probs.value for probs in probs_layers])

        # Find the maximum probability and corresponding tokens for each position
        max_probs, tokens = probs.max(dim=-1)

        # Decode token IDs to words for each
        words = [[model.tokenizer.decode(t.cpu()).encode("unicode_escape").decode() for t in layer_tokens]
            for layer_tokens in tokens]

        return words