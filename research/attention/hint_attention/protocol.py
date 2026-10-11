"""Replace HINT's eligibility mean pooling with two linear attention scorers.

Integration reference: futianfan/clinical-trial-outcome-prediction,
commit 8dc0497f23fdb84e2905da7655924a91e6e79798, HINT/protocol_encode.py.
See THIRD_PARTY.md for upstream attribution and usage restrictions.
"""

from copy import deepcopy

import torch
from torch import nn


class AttentionProtocolEncoder(nn.Module):
    """Pool each unpadded inclusion/exclusion group, then reuse HINT's projection.

    Input: iterable of (inclusion_tensor, exclusion_tensor) pairs; each tensor
    has shape [number_of_sentences, input_dim]. Output: [batch, output_dim].
    Sentence embeddings are detached deliberately: the text encoder is frozen.
    No sentence segmentation, cache lookup, or inclusion/exclusion parsing is
    changed here. Zero-vector fallback rows from upstream are retained.
    """

    def __init__(self, original_encoder):
        super().__init__()
        if isinstance(original_encoder, AttentionProtocolEncoder):
            raise ValueError("Attention is already installed on this encoder.")
        if not isinstance(getattr(original_encoder, "fc", None), nn.Linear):
            raise TypeError("Expected HINT's Protocol_Embedding with a linear fc.")

        self.input_dim = original_encoder.input_dim
        self.output_dim = original_encoder.output_dim
        self.highway_num = original_encoder.highway_num
        if original_encoder.fc.in_features != 2 * self.input_dim:
            raise ValueError("Protocol projection must accept two sentence summaries.")
        if original_encoder.fc.out_features != self.output_dim:
            raise ValueError("Protocol projection output dimension is inconsistent.")

        # Copy learned parameters; do not share projection weights with baseline.
        self.fc = deepcopy(original_encoder.fc)
        self.f = deepcopy(original_encoder.f)
        self.inclusion_score = nn.Parameter(self.fc.weight.new_zeros(self.input_dim))
        self.exclusion_score = nn.Parameter(self.fc.weight.new_zeros(self.input_dim))
        self.train(original_encoder.training)

    @property
    def embedding_size(self):
        return self.output_dim

    @property
    def device(self):
        # Follow actual parameters after .to(), avoiding a stale device field.
        return self.fc.weight.device

    @device.setter
    def device(self, value):
        self.to(value)

    def set_device(self, device):
        self.to(device)
        return self

    def _pool(self, features, scorer):
        if not isinstance(features, torch.Tensor):
            raise TypeError("Sentence embeddings must be a torch.Tensor.")
        if features.ndim != 2 or features.shape[1] != self.input_dim:
            raise ValueError(
                "Expected an unpadded sentence matrix [n, {}], got {}.".format(
                    self.input_dim, tuple(features.shape)
                )
            )
        if features.shape[0] == 0:
            raise ValueError(
                "Empty sentence group. Use the same documented missing-data "
                "policy as the baseline; upstream supplies a single zero row."
            )
        features = features.detach().to(device=self.device, dtype=self.fc.weight.dtype)
        # With zero scorers, softmax gives 1/n and pooling equals the mean,
        # up to floating-point roundoff. No scorer bias: softmax cancels it.
        scores = torch.mv(features, scorer)
        weights = torch.softmax(scores, dim=0)
        summary = torch.sum(features * weights.unsqueeze(1), dim=0, keepdim=True)
        return summary, weights

    def forward_single(self, inclusion_feature, exclusion_feature):
        inclusion, _ = self._pool(inclusion_feature, self.inclusion_score)
        exclusion, _ = self._pool(exclusion_feature, self.exclusion_score)
        return inclusion, exclusion

    def _encode(self, in_ex_feature, include_attention):
        summaries = []
        attention = []
        for inclusion_feature, exclusion_feature in in_ex_feature:
            inclusion, inclusion_weights = self._pool(inclusion_feature, self.inclusion_score)
            exclusion, exclusion_weights = self._pool(exclusion_feature, self.exclusion_score)
            summaries.append(torch.cat((inclusion, exclusion), dim=1))
            if include_attention:
                # Avoid retaining training graphs or GPU memory for UI metadata.
                attention.append({
                    "inclusion": inclusion_weights.detach().cpu(),
                    "exclusion": exclusion_weights.detach().cpu(),
                })
        if not summaries:
            raise ValueError("Protocol batch must contain at least one trial.")
        encoded = self.f(self.fc(torch.cat(summaries, dim=0)))
        return encoded, attention

    def forward(self, in_ex_feature):
        encoded, _ = self._encode(in_ex_feature, include_attention=False)
        return encoded

    def forward_with_attention(self, in_ex_feature):
        """Return representations and per-group weights in input row order.

        Upstream drops uncached sentences, so weights refer to retained rows,
        not necessarily every sentence in the original trial text. The caller
        must retain a matching sentence map. A synthetic fallback row is not
        a real sentence. Attention weights are not causal explanations.
        """
        return self._encode(in_ex_feature, include_attention=True)


def install_attention(model):
    """Replace model.protocol_encoder IN PLACE, before creating an optimizer.

    All other modules remain untouched. Deep-copy the baseline model first if
    both versions must coexist. Return the same model for convenient chaining.
    """
    if not hasattr(model, "protocol_encoder"):
        raise TypeError("Expected a HINT model with a protocol_encoder attribute.")
    model.protocol_encoder = AttentionProtocolEncoder(model.protocol_encoder)
    return model


def set_training_scope(model, scope):
    """Select 'attention_only' for diagnostics, or 'full' for matched training.

    'full' enables every model parameter; cached sentence inputs remain detached.
    Apply 'full' to BOTH baseline and modified HINT for the main experiment.
    Recreate the optimizer after changing scope. Do not call full-model no_grad
    during attention-only training: gradients must pass through frozen layers.
    """
    if scope not in ("attention_only", "full"):
        raise ValueError("scope must be 'attention_only' or 'full'.")
    if scope == "attention_only" and not isinstance(
        getattr(model, "protocol_encoder", None), AttentionProtocolEncoder
    ):
        raise TypeError("Install attention before selecting attention_only scope.")

    for parameter in model.parameters():
        parameter.requires_grad_(scope == "full")
        parameter.grad = None
    if scope == "attention_only":
        model.protocol_encoder.inclusion_score.requires_grad_(True)
        model.protocol_encoder.exclusion_score.requires_grad_(True)
    return model
