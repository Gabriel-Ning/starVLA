# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
"""
PI05Beast Framework — PI0.5 (PaliGemma) + continuous BEAST action space.

Keeps the OpenPI flow-matching action expert, but treats each FM step as a
B-spline **weight row** rather than a raw control tick:

  - ``action_horizon``  ≡ ``num_basis``  (FM sequence length)
  - ``action_dim``      ≡ robot DoF
  - ``beast_seq_len``   ≡ decoded control chunk length (env steps)

Train: encode GT action chunk → weights ``[B, num_basis, dof]`` → PI05 FM loss.
Infer: sample weights → BEAST decode → ``[B, beast_seq_len, dof]`` actions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import torch

from starVLA.model.framework.VLM4A.PI05 import PI05, PI05DefaultConfig, IMAGE_KEYS
from starVLA.model.modules.action_model.OpenPI_ActionHead import OpenPI05ActionHead
from starVLA.model.modules.action_model.beast_ActionHeader import BeastTokenizer
from starVLA.model.tools import FRAMEWORK_REGISTRY


@dataclass
class PI05BeastDefaultConfig(PI05DefaultConfig):
    name: str = "PI05Beast"
    precision: Literal["bfloat16", "float32"] = "bfloat16"

    # FM predicts (num_basis, dof); defaults match continuous BEAST settings.
    action_dim: int = 7
    action_horizon: int = 8  # == num_basis
    max_state_dim: int = 32
    discrete_state_input: bool = True
    max_token_len: int = 200
    num_inference_steps: int = 10

    # Decoded control horizon (env / dataset action chunk length).
    beast_seq_len: int = 10
    num_basis: int = 8
    degree_p: int = 4
    vocab_size: int = 256
    gripper_zero_order: bool = True
    gripper_dof: int = 1

    image_resolution: list[int] = field(default_factory=lambda: [224, 224])
    image_keys: list[str] = field(default_factory=lambda: list(IMAGE_KEYS))


@FRAMEWORK_REGISTRY.register("PI05Beast")
@FRAMEWORK_REGISTRY.register("Pi05Beast")
class PI05Beast(PI05):
    """PI0.5 backbone + continuous BEAST weights as flow-matching targets."""

    default_config_cls = PI05BeastDefaultConfig
    action_head_cls = OpenPI05ActionHead
    model_name = "PI05Beast"
    discrete_state_input = True

    def __init__(self, config=None, **kwargs):
        # Ensure FM dims match BEAST weight layout before parent builds the head.
        cfg = config
        if cfg is not None and hasattr(cfg, "framework"):
            fw = cfg.framework
            num_basis = int(getattr(fw, "num_basis", getattr(getattr(fw, "action_model", None), "num_basis", 8) or 8))
            # Prefer explicit action_model.num_basis when nested.
            am = getattr(fw, "action_model", None)
            if am is not None and hasattr(am, "num_basis"):
                num_basis = int(am.num_basis)
            fw.action_horizon = num_basis
            if am is not None:
                am.action_horizon = num_basis
                if hasattr(am, "future_action_window_size"):
                    am.future_action_window_size = num_basis

        super().__init__(config=cfg, **kwargs)

        fw = self.config.framework
        self.num_basis = int(getattr(fw, "num_basis", self.action_horizon))
        self.beast_seq_len = int(getattr(fw, "beast_seq_len", self.action_horizon))
        degree_p = int(getattr(fw, "degree_p", 4))
        vocab_size = int(getattr(fw, "vocab_size", 256))
        gripper_zero_order = bool(getattr(fw, "gripper_zero_order", True))
        gripper_dof = int(getattr(fw, "gripper_dof", 1))

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.beast_tokenizer = BeastTokenizer(
            num_dof=self.action_dim,
            num_basis=self.num_basis,
            seq_len=self.beast_seq_len,
            vocab_size=vocab_size,
            degree_p=degree_p,
            gripper_zero_order=gripper_zero_order,
            gripper_dof=gripper_dof,
            device=device,
        )

    def _actions_to_beast_weights(self, actions: torch.Tensor) -> torch.Tensor:
        """``[B, T, D]`` raw actions → ``[B, num_basis, D]`` continuous BEAST weights (denormalized scale for FM).

        Uses normalized encode then denormalize so FM targets stay in a stable range
        close to the underlying weight magnitudes after bound updates.
        """
        # Crop / pad time to beast_seq_len for encoding.
        bsz, t, dof = actions.shape
        if t >= self.beast_seq_len:
            chunk = actions[:, -self.beast_seq_len :, :]
        else:
            pad = actions[:, -1:, :].expand(bsz, self.beast_seq_len - t, dof)
            chunk = torch.cat([actions, pad], dim=1)

        # Move tokenizer to action device if needed.
        if next(self.beast_tokenizer.buffers()).device != chunk.device:
            self.beast_tokenizer.to(chunk.device)

        with torch.no_grad():
            # Normalized params in [-1, 1], shape [B, num_basis * dof]
            norm_w = self.beast_tokenizer.encode_continuous(chunk, update_bounds=True)
        # Flow-match in normalized weight space; reshape to (horizon=num_basis, dim=dof)
        return norm_w.view(bsz, self.num_basis, self.action_dim)

    def _weights_to_actions(self, weights: torch.Tensor) -> torch.Tensor:
        """``[B, num_basis, D]`` → decoded actions ``[B, beast_seq_len, D]``."""
        bsz = weights.shape[0]
        flat = weights.reshape(bsz, self.num_basis * self.action_dim)
        if next(self.beast_tokenizer.buffers()).device != flat.device:
            self.beast_tokenizer.to(flat.device)
        return self.beast_tokenizer.decode_continuous(flat)

    def forward(self, examples: list[dict] = None, **kwargs):
        batch = self._prepare_examples(examples, include_actions=True)
        observation = self._build_observation_from_batch(batch)
        actions = batch["action"].to(device=self.action_head.action_in_proj.weight.device, dtype=torch.float32)
        # `_prepare_examples` pads to (action_horizon, action_dim) == (num_basis, dof).
        # Re-encode from the raw example actions for correct BEAST fitting length.
        raw = []
        for example in examples:
            a = np.asarray(example["action"], dtype=np.float32)
            if a.ndim == 1:
                a = a[None, :]
            raw.append(a)
        # Pad to common T then stack
        max_t = max(a.shape[0] for a in raw)
        stacked = np.stack(
            [np.pad(a, ((0, max_t - a.shape[0]), (0, 0)), mode="edge") for a in raw],
            axis=0,
        )
        action_t = torch.as_tensor(stacked, device=actions.device, dtype=torch.float32)
        weight_targets = self._actions_to_beast_weights(action_t)
        return self.forward_from_observation(
            observation,
            weight_targets,
            noise=kwargs.get("noise"),
            time=kwargs.get("time"),
            return_debug=bool(kwargs.get("return_debug", False)),
        )

    @torch.inference_mode()
    def predict_action(self, examples: list[dict] | dict = None, **kwargs):
        if not isinstance(examples, list):
            examples = [examples]
        batch = self._prepare_examples(examples, include_actions=False)
        observation = self._build_observation_from_batch(batch)
        weights = self.sample_actions(
            observation,
            noise=kwargs.get("noise"),
            num_steps=int(kwargs.get("num_steps", kwargs.get("num_inference_steps", self.num_inference_steps))),
        )
        actions = self._weights_to_actions(weights)
        return {"normalized_actions": actions.cpu().numpy()}
