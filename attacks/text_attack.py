import random
from typing import List
import torch

from models.base_model import BaseModel
from attacks.base_attack import BaseAttack, AdversarialExample

# Small synonym map — sufficient for unit tests and demo runs.
# A production version would use WordNet or a masked-LM for context-aware swaps.
_SYNONYMS = {
    "good": ["great", "fine", "decent"],
    "bad": ["poor", "terrible", "awful"],
    "happy": ["glad", "pleased", "content"],
    "sad": ["unhappy", "miserable", "down"],
    "big": ["large", "huge", "massive"],
    "small": ["tiny", "little", "minor"],
    "fast": ["quick", "rapid", "swift"],
    "slow": ["sluggish", "gradual", "leisurely"],
}


def _synonym_swap(tokens: List[str], rng: random.Random) -> List[str]:
    """Replace one random swappable token with a synonym."""
    swappable = [i for i, t in enumerate(tokens) if t.lower() in _SYNONYMS]
    if not swappable:
        return tokens
    idx = rng.choice(swappable)
    token = tokens[idx].lower()
    tokens[idx] = rng.choice(_SYNONYMS[token])
    return tokens


def _char_flip(tokens: List[str], rng: random.Random) -> List[str]:
    """Duplicate a character in a random token (typo simulation)."""
    if not tokens:
        return tokens
    idx = rng.randrange(len(tokens))
    word = tokens[idx]
    if len(word) > 1:
        pos = rng.randrange(len(word))
        tokens[idx] = word[:pos] + word[pos] + word[pos:]
    return tokens


def _word_insert(tokens: List[str], rng: random.Random) -> List[str]:
    """Insert a neutral filler word at a random position."""
    fillers = ["really", "very", "quite", "just", "actually"]
    idx = rng.randrange(len(tokens) + 1)
    tokens.insert(idx, rng.choice(fillers))
    return tokens


_PERTURBATIONS = [_synonym_swap, _char_flip, _word_insert]


class TextAttack(BaseAttack):
    """
    Token-level perturbations for text models.
    Operates on raw text strings decoded from token IDs, then re-tokenises.
    Three strategies: synonym swap, character flip, filler word insertion.
    """

    def __init__(self, model: BaseModel, seed: int = 42):
        super().__init__(model)
        self._rng = random.Random(seed)
        # Tokeniser lives on the model wrapper
        self._tokeniser = getattr(model, "_tokenizer", None)

    def _perturb(self, text: str) -> str:
        tokens = text.split()
        fn = self._rng.choice(_PERTURBATIONS)
        tokens = fn(tokens, self._rng)
        return " ".join(tokens)

    def attack(
        self,
        inputs: torch.Tensor,
        labels: List[int],
        profile: dict,
    ) -> List[AdversarialExample]:
        if self._tokeniser is None:
            raise ValueError("TextAttack requires model._tokeniser to be set.")

        examples = []

        for i, label in enumerate(labels):
            input_ids = inputs[i].unsqueeze(0)

            with torch.no_grad():
                orig_logits = self.model._model(input_ids)
                if hasattr(orig_logits, "logits"):
                    orig_logits = orig_logits.logits
                orig_pred = int(orig_logits.argmax(dim=1).item())

            # Decode → perturb → re-tokenise
            text = self._tokeniser.decode(input_ids[0], skip_special_tokens=True)
            perturbed_text = self._perturb(text)

            enc = self._tokeniser(
                perturbed_text,
                return_tensors="pt",
                truncation=True,
                max_length=input_ids.shape[1],
                padding="max_length",
            )
            adv_ids = enc["input_ids"]

            with torch.no_grad():
                adv_logits = self.model._model(adv_ids)
                if hasattr(adv_logits, "logits"):
                    adv_logits = adv_logits.logits
                adv_pred = int(adv_logits.argmax(dim=1).item())

            examples.append(
                AdversarialExample(
                    original_input=input_ids.squeeze(0).float(),
                    perturbed_input=adv_ids.squeeze(0).float(),
                    true_label=label,
                    original_pred=orig_pred,
                    adversarial_pred=adv_pred,
                    attack_name="TextAttack",
                    attack_params={"original_text": text, "perturbed_text": perturbed_text},
                )
            )

        return examples
