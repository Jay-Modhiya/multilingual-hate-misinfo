"""
src/models/muril_classifier.py
───────────────────────────────
MuRIL-based sequence classifier for hate speech and misinformation detection.

Architecture:
  MuRIL (google/muril-base-cased) → [CLS] pooled → Dropout → Linear head

Mirrors the DistilBERT approach from CA-3 but:
  - Uses MuRIL instead (multilingual, Hindi-optimised)
  - Supports configurable num_labels for multi-task setup
  - Adds label smoothing for class imbalance (Davidson: 5.8% hate)
"""

import logging
import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
#  Model Definition
# ─────────────────────────────────────────────

class MuRILClassifier(nn.Module):
    """
    Fine-tuned MuRIL for binary / multi-class text classification.

    Args:
        cfg         : project config dict
        num_labels  : number of output classes
        task        : "hate" | "misinfo"  (for logging clarity)
    """

    def __init__(self, cfg: dict, num_labels: int = 2, task: str = "hate"):
        super().__init__()
        self.task       = task
        self.num_labels = num_labels
        checkpoint      = cfg["model"]["hf_checkpoint"]
        dropout_p       = cfg["model"]["dropout"]

        logger.info(f"Initialising MuRILClassifier | task={task} | labels={num_labels}")

        # Load pre-trained MuRIL encoder
        self.muril = AutoModel.from_pretrained(checkpoint)
        hidden     = self.muril.config.hidden_size     # 768 for muril-base

        # Classification head (same design as CA-3 DistilBERT head)
        self.pre_classifier = nn.Linear(hidden, hidden)
        self.dropout        = nn.Dropout(dropout_p)
        self.classifier     = nn.Linear(hidden, num_labels)

        # Weight init for the new head
        nn.init.xavier_uniform_(self.pre_classifier.weight)
        nn.init.xavier_uniform_(self.classifier.weight)

        logger.info(
            f"MuRIL backbone: {checkpoint} | hidden={hidden} | "
            f"params: {self._count_params():,}"
        )

    def forward(
        self,
        input_ids:      torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        labels:         Optional[torch.Tensor] = None,
    ) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
        """
        Forward pass.

        Returns:
            (loss, logits)  — loss is None when labels is not provided
        """
        outputs = self.muril(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        # [CLS] token representation  →  shape (batch, hidden)
        cls_output = outputs.last_hidden_state[:, 0, :]

        # Head
        pooled  = torch.relu(self.pre_classifier(cls_output))
        pooled  = self.dropout(pooled)
        logits  = self.classifier(pooled)              # (batch, num_labels)

        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss(label_smoothing=0.05)
            loss = loss_fn(logits, labels)

        return loss, logits

    def _count_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def freeze_backbone(self):
        """Freeze MuRIL encoder layers (useful for quick head-only training)."""
        for param in self.muril.parameters():
            param.requires_grad = False
        logger.info("MuRIL backbone frozen — training head only.")

    def unfreeze_backbone(self):
        """Unfreeze all layers for full fine-tuning."""
        for param in self.muril.parameters():
            param.requires_grad = True
        logger.info("MuRIL backbone unfrozen — full fine-tuning enabled.")


# ─────────────────────────────────────────────
#  Factory helper
# ─────────────────────────────────────────────

def build_model(cfg: dict, task: str = "hate") -> MuRILClassifier:
    """
    Factory: creates the right MuRILClassifier based on task.

    task="hate"    → num_labels = cfg.model.num_labels_hate
    task="misinfo" → num_labels = cfg.model.num_labels_fake
    """
    label_key  = "num_labels_hate" if task == "hate" else "num_labels_fake"
    num_labels = cfg["model"][label_key]
    model      = MuRILClassifier(cfg, num_labels=num_labels, task=task)
    return model


# ─────────────────────────────────────────────
#  Save / Load helpers
# ─────────────────────────────────────────────

def save_model(model: MuRILClassifier, path: str):
    import os
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)
    logger.info(f"Model saved → {path}")


def load_model(cfg: dict, path: str, task: str = "hate") -> MuRILClassifier:
    model = build_model(cfg, task=task)
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    logger.info(f"Model loaded ← {path}")
    return model


# ─────────────────────────────────────────────
#  Smoke-test (no GPU needed — just checks shapes)
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import yaml
    with open("configs/config.yaml") as f:
        cfg = yaml.safe_load(f)

    # Override checkpoint to a tiny BERT to avoid downloading MuRIL
    cfg["model"]["hf_checkpoint"] = "google/muril-base-cased"

    print("Building hate model …")
    model = build_model(cfg, task="hate")
    model.eval()

    # Dummy batch
    B, L = 4, 128
    ids   = torch.zeros(B, L, dtype=torch.long)
    mask  = torch.ones(B, L, dtype=torch.long)
    lbls  = torch.tensor([0, 1, 0, 1])

    with torch.no_grad():
        loss, logits = model(ids, mask, labels=lbls)

    print(f"Loss   : {loss.item():.4f}")
    print(f"Logits : {logits.shape}")   # (4, 2)
    print("Smoke test passed ✓")
