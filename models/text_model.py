from typing import Dict, List, Optional
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

from .base_model import BaseModel


class TextModel(BaseModel):
    """
    DistilBERT wrapper for sequence classification.
    Pretrained on SST-2 (positive/negative sentiment, 2 classes) by default.
    Supports gradient extraction and activation hooks for clustering.
    """

    DEFAULT_CHECKPOINT = "distilbert-base-uncased-finetuned-sst-2-english"

    def __init__(self, checkpoint: str = DEFAULT_CHECKPOINT):
        self._tokenizer = DistilBertTokenizer.from_pretrained(checkpoint)
        self._model = DistilBertForSequenceClassification.from_pretrained(checkpoint)
        # safetensors memory-maps weights from disk; on Apple Silicon the mmap'd
        # addresses can be misaligned for Apple Accelerate BLAS (EXC_ARM_DA_ALIGN).
        # Cloning every parameter forces a copy into properly-aligned heap memory.
        for param in self._model.parameters():
            param.data = param.data.clone()
        self._model.eval()
        self._activation_cache: Dict[str, torch.Tensor] = {}
        self._hooks: List = []

    @property
    def model_name(self) -> str:
        return "distilbert-sst2"

    @property
    def input_shape(self) -> tuple:
        return (512,)  # max token sequence length

    def tokenize(self, texts: List[str], max_length: int = 128) -> Dict[str, torch.Tensor]:
        return self._tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )

    def predict(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        inputs: token id tensor of shape (batch, seq_len).
        Returns logits of shape (batch, num_classes).
        """
        self._model.eval()
        with torch.no_grad():
            out = self._model(input_ids=inputs)
        return out.logits

    def predict_from_text(self, texts: List[str]) -> torch.Tensor:
        """Convenience method — tokenizes then predicts."""
        encoded = self.tokenize(texts)
        self._model.eval()
        with torch.no_grad():
            out = self._model(**encoded)
        return out.logits

    def get_gradients(
        self, inputs: torch.Tensor, target_class: int
    ) -> torch.Tensor:
        """
        Gradient w.r.t. input token embeddings (not discrete token ids).
        This is the standard approach for text adversarial attacks.
        Returns gradient tensor of shape (batch, seq_len, hidden_size).
        """
        self._model.eval()
        embeddings = self._model.distilbert.embeddings(inputs)
        embeddings.retain_grad()

        out = self._model(inputs_embeds=embeddings)
        self._model.zero_grad()
        out.logits[0, target_class].backward()
        return embeddings.grad.detach()

    def get_activations(
        self, inputs: torch.Tensor, layer_name: str
    ) -> torch.Tensor:
        self._clear_hooks()
        self._activation_cache = {}

        target_layer = dict(self._model.named_modules()).get(layer_name)
        if target_layer is None:
            raise ValueError(
                f"Layer '{layer_name}' not found. "
                f"Available: {self.get_available_layers()}"
            )

        def hook_fn(module, input, output):
            act = output[0] if isinstance(output, tuple) else output
            self._activation_cache[layer_name] = act.detach()

        handle = target_layer.register_forward_hook(hook_fn)
        self._hooks.append(handle)

        self._model.eval()
        with torch.no_grad():
            self._model(input_ids=inputs)

        self._clear_hooks()
        return self._activation_cache[layer_name]

    def get_available_layers(self) -> List[str]:
        return [name for name, _ in self._model.named_modules() if name]

    def _clear_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks = []
