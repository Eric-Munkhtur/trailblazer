"""Eligibility attention extension for HINT; requires PyTorch."""

from .protocol import AttentionProtocolEncoder, install_attention, set_training_scope

__all__ = ["AttentionProtocolEncoder", "install_attention", "set_training_scope"]
