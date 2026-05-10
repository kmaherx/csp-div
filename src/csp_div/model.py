"""Model loading — the single place AutoModelForCausalLM is instantiated.

Every pipeline script calls `load_llama()` once at startup; downstream
library functions consume the returned model + tokenizer. This is what
collapses ~150 redundant model loads in the old pipeline down to one
load per pod.
"""
from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from .config import MODEL_NAME


def load_llama(
    device: str | torch.device | None = None,
    dtype: torch.dtype = torch.bfloat16,
) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """Load the headline Llama-3.1-8B-Instruct model in eval mode.

    Parameters are frozen; the caller is expected to wrap a trainable
    SoftPrompt parameter alongside.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=dtype, device_map="auto",
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    hidden = model.get_input_embeddings().weight.shape[1]
    print(f"  hidden_size={hidden}, device={model.device}, dtype={dtype}")
    return model, tokenizer


def get_transformer_layers(model: PreTrainedModel) -> torch.nn.ModuleList:
    """Return the model's transformer-block ModuleList.

    Different HF architectures expose this at different paths:
      - Llama/Qwen/most: model.model.layers
      - some wrapped models: model.model.language_model.layers
    """
    if hasattr(model.model, "language_model"):
        return model.model.language_model.layers
    return model.model.layers
