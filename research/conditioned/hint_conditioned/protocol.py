"""Drug/disease-conditioned sentence pooling; downstream HINT is inherited.

Upstream: futianfan/clinical-trial-outcome-prediction at
8dc0497f23fdb84e2905da7655924a91e6e79798. Its license still applies.
This module changes only the three-encoder junction and eligibility pooling.
"""

import math
from copy import deepcopy

import torch
from torch import nn
from HINT.model import HINTModel


class ConditionedProtocolEncoder(nn.Module):
    """Separate inclusion/exclusion queries, shared sentence keys, raw values.

q_g = W_g LayerNorm([drug; disease]) + b_g
a_g = softmax((sentences W_key^T) q_g / sqrt(32))
pooled_g = sum(a_g * sentences)

The original projection and activation consume the two pooled vectors.
Zero query weights/biases preserve mean pooling at initialization. Cached
sentence vectors remain frozen; gradients flow through query/context and
through the otherwise unchanged drug/disease encoders.
"""

    def __init__(self, original_encoder, context_dim, attention_dim=32):
        super().__init__()
        if not isinstance(original_encoder.fc, nn.Linear):
            raise TypeError("Expected the upstream linear protocol projection.")
        self.input_dim = original_encoder.input_dim
        self.output_dim = original_encoder.output_dim
        self.highway_num = original_encoder.highway_num
        self.context_dim = context_dim
        self.attention_dim = attention_dim
        self.fc = deepcopy(original_encoder.fc)
        self.f = deepcopy(original_encoder.f)
        self.context_norm = nn.LayerNorm(context_dim, elementwise_affine=False)
        self.sentence_key = nn.Linear(self.input_dim, attention_dim, bias=False)
        self.inclusion_query = nn.Linear(context_dim, attention_dim)
        self.exclusion_query = nn.Linear(context_dim, attention_dim)
        for query in (self.inclusion_query, self.exclusion_query):
            nn.init.zeros_(query.weight)
            nn.init.zeros_(query.bias)
        self.to(device=self.fc.weight.device, dtype=self.fc.weight.dtype)
        self.train(original_encoder.training)

    @property
    def embedding_size(self):
        return self.output_dim

    @property
    def device(self):
        return self.fc.weight.device

    @device.setter
    def device(self, device):
        self.to(device)

    def set_device(self, device):
        self.to(device)
        return self

    def _pool(self, features, query):
        if features.ndim != 2 or features.shape[0] == 0 or features.shape[1] != self.input_dim:
            raise ValueError("Expected nonempty, unpadded sentence features [N, 768].")
        values = features.detach().to(device=self.device, dtype=self.fc.weight.dtype)
        scores = torch.mv(self.sentence_key(values), query) / math.sqrt(self.attention_dim)
        weights = torch.softmax(scores, dim=0)
        return torch.sum(values * weights[:, None], dim=0, keepdim=True), weights

    def _encode(self, groups, context, include_attention):
        if context.ndim != 2 or context.shape != (len(groups), self.context_dim):
            raise ValueError("Context rows must align exactly with eligibility groups.")
        context = context.to(device=self.device, dtype=self.fc.weight.dtype)
        normalized = self.context_norm(context)
        inclusion_queries = self.inclusion_query(normalized)
        exclusion_queries = self.exclusion_query(normalized)
        summaries, explanations = [], []
        for row, (inclusion, exclusion) in enumerate(groups):
            inc, inc_weights = self._pool(inclusion, inclusion_queries[row])
            exc, exc_weights = self._pool(exclusion, exclusion_queries[row])
            summaries.append(torch.cat((inc, exc), dim=1))
            if include_attention:
                explanations.append({
                    "inclusion": inc_weights.detach().cpu(),
                    "exclusion": exc_weights.detach().cpu(),
                })
        if not summaries:
            raise ValueError("Empty trial batch.")
        return self.f(self.fc(torch.cat(summaries, dim=0))), explanations

    def forward(self, groups, context):
        encoded, _ = self._encode(groups, context, False)
        return encoded

    def forward_with_attention(self, groups, context):
        """Weights index retained cache rows; they are not causal attributions."""
        return self._encode(groups, context, True)


class ConditionedHINTModel(HINTModel):
    """Reuse upstream graph forward; override only its encoder junction."""

    def forward_get_three_encoders(self, smiles, codes, criteria):
        drug = self.molecule_encoder.forward_smiles_lst_lst(smiles)
        disease = self.disease_encoder.forward_code_lst3(codes)
        context = torch.cat((drug, disease), dim=1)
        protocol = self.protocol_encoder(criteria, context)
        return drug, disease, protocol


def install_conditioned(model, seed=0):
    """Retain the trusted model's weights and exact GRAM lookup metadata.

Change its Python subclass without reconstructing encoders or changing their
index mapping. Save tensor state_dicts only. To reload, load the same trusted
base checkpoint, call this installer, then load the saved state_dict strictly.
"""
    if type(model) is not HINTModel:
        raise TypeError("Install only onto the original, unmodified HINTModel.")
    context_dim = model.molecule_encoder.embedding_size + model.disease_encoder.embedding_size
    # Isolate new-layer initialization from the matched training RNG stream.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        encoder = ConditionedProtocolEncoder(model.protocol_encoder, context_dim, 32)
    model.protocol_encoder = encoder
    model.__class__ = ConditionedHINTModel
    return model
