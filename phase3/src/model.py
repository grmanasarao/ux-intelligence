from __future__ import annotations

from typing import Optional

import torch
from torch import nn
from transformers import AutoModel, PreTrainedModel, PretrainedConfig


class MultiTaskConfig(PretrainedConfig):
    model_type = "ux_multi_task_deberta"

    def __init__(
        self,
        encoder_name: str = "microsoft/deberta-v3-base",
        num_topics: int = 8,
        num_severities: int = 5,
        num_sentiments: int = 3,
        topic_labels: Optional[list[str]] = None,
        sentiment_labels: Optional[list[str]] = None,
        dropout: float = 0.1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.encoder_name = encoder_name
        self.num_topics = num_topics
        self.num_severities = num_severities
        self.num_sentiments = num_sentiments
        self.topic_labels = topic_labels or []
        self.sentiment_labels = sentiment_labels or []
        self.dropout = dropout


class UXMultiTaskModel(PreTrainedModel):
    config_class = MultiTaskConfig

    def __init__(self, config: MultiTaskConfig):
        super().__init__(config)
        # IMPORTANT: load the pretrained encoder and do not call post_init()
        # after it, otherwise the pretrained encoder weights could be reinitialized.
        self.encoder = AutoModel.from_pretrained(config.encoder_name)
        hidden = self.encoder.config.hidden_size

        self.dropout = nn.Dropout(config.dropout)
        self.topic_head = nn.Linear(hidden, config.num_topics)
        self.severity_head = nn.Linear(hidden, config.num_severities)
        self.sentiment_head = nn.Linear(hidden, config.num_sentiments)

        nn.init.normal_(self.topic_head.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.topic_head.bias)
        nn.init.normal_(self.severity_head.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.severity_head.bias)
        nn.init.normal_(self.sentiment_head.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.sentiment_head.bias)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        topic_targets: Optional[torch.Tensor] = None,
        severity_targets: Optional[torch.Tensor] = None,
        sentiment_targets: Optional[torch.Tensor] = None,
    ):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0]
        pooled = self.dropout(pooled)

        topic_logits = self.topic_head(pooled)
        severity_logits = self.severity_head(pooled)
        sentiment_logits = self.sentiment_head(pooled)

        losses = {}

        if topic_targets is not None:
            example_mask = (topic_targets >= 0).any(dim=1)
            if example_mask.any():
                target = topic_targets[example_mask].float()
                logits = topic_logits[example_mask]
                losses["topic"] = nn.functional.binary_cross_entropy_with_logits(
                    logits, target
                )

        if severity_targets is not None:
            valid = severity_targets >= 0
            if valid.any():
                losses["severity"] = nn.functional.cross_entropy(
                    severity_logits[valid], severity_targets[valid]
                )

        if sentiment_targets is not None:
            valid = sentiment_targets >= 0
            if valid.any():
                losses["sentiment"] = nn.functional.cross_entropy(
                    sentiment_logits[valid], sentiment_targets[valid]
                )

        loss = None
        if losses:
            weights = {"topic": 1.0, "severity": 1.0, "sentiment": 0.7}
            loss = sum(weights[name] * value for name, value in losses.items())

        return {
            "loss": loss,
            "topic_logits": topic_logits,
            "severity_logits": severity_logits,
            "sentiment_logits": sentiment_logits,
            "losses": losses,
        }
