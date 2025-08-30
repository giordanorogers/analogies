import torch
import numpy as np
from nnsight import LanguageModel
from dataclasses import dataclass
from dataclasses_json import DataClassJsonMixin


@dataclass(frozen=False)
class AttentionInformation(DataClassJsonMixin):
    """
    Class to store data for analyzing attention patterns?
    """
    tokenized_prompt: list[str] # a list of token strings
    attention_matrices: np.ndarray # a numpy matrix of the attention pattern
    logits: torch.Tensor | None = None # the logits corresponding to the next token prediction for the tokenized prompt

    def __init__(
        self,
        prompt: str,
        tokenized_prompt: list[str],
        attention_matrices: torch.tensor,
        logits: torch.Tensor,
    ):
        assert (
            len(tokenized_prompt) == attention_matrices.shape[-1]
        ), "Tokenized prompt and attention matrices must have the same length."
        # For each layer and each head there is a (token x token) attention matrix,
        # representing how much each token attends to every other token (including itself)
        assert (
            len(attention_matrices.shape) == 4
        ), "Attention matrices must be of shape (layers, heads, tokens, tokens)"
        assert (
            attention_matrices.shape[-1] == attention_matrices.shape[-2]
        ), "Attention matrices must be square"

        self.prompt = prompt
        self.logits = logits
        self.tokenized_prompt = tokenized_prompt
        self.attention_matrices = attention_matrices

    def get_attn_matrix(
        self,
        layer: int,
        head: int
    ) -> torch.tensor:
        return self.attention_matrices[layer, head]

@torch.inference_mode()
def get_attention_matrices(
    input: str,
    model: LanguageModel
) -> torch.tensor:
    """
    Returns:
        attention_matrices: (layers, heads, tokens, tokens)
    """
    if isinstance(input, str):
        input = model.tokenizer(
            input,
            return_tensors="pt"
        )
    
    with torch.no_grad():
        with model.trace(input, output_attentions=True):
            # Save the final output (including the attentions)
            output = model.model.output.save()
            # Save the last token logits
            logits = model.output.logits[0][-1].save()

    # Move all the attention tensors from the model output to GPU memory
    # Allows for subsequent computation.
    output.attentions = [
        attn.cuda() for attn in output.attentions
    ]

    # (layers, heads, tokens, tokens)
    attentions = torch.vstack(output.attentions)
    
    return AttentionInformation(
        prompt=input,
        tokenized_prompt=[
            model.tokenizer.decode(tok) for tok in input.input_ids[0]
        ],
        attention_matrices=attentions.detach().cpu().to(torch.float32).numpy(),
        logits=logits.detach().cpu()
    )
