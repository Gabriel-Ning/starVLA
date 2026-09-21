# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
"""
QwenBeastFast Framework — Qwen-VL + discrete BEAST action tokens.

Fork of QwenFast: same autoregressive next-token training path, but action
tokenization uses BEAST discrete B-spline bins instead of physical-intelligence/fast.
Requires a Qwen checkpoint whose vocabulary includes ``<robot_action_*>`` specials
(same Action-token range as QwenFast).
"""

from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image

from deployment.model_server.tools.image_tools import to_pil_preserve
from starVLA.model.framework.base_framework import baseframework
from starVLA.model.framework.share_tools import merge_framework_config
from starVLA.model.modules.action_model.beast_ActionHeader import get_action_model
from starVLA.model.modules.vlm import get_vlm_model
from starVLA.model.tools import FRAMEWORK_REGISTRY
from starVLA.training.trainer_utils import initialize_overwatch

logger = initialize_overwatch(__name__)


@dataclass
class QwenBeastFastDefaultConfig:
    """Qwen + discrete BEAST tokens (FAST-style AR over BEAST bins)."""

    name: str = "QwenBeastFast"

    qwenvl: dict = field(
        default_factory=lambda: {
            "base_vlm": "./playground/Pretrained_models/Qwen3-VL-4B-Instruct-Action",
            "attn_implementation": "flash_attention_2",
        }
    )

    action_model: dict = field(
        default_factory=lambda: {
            "action_model_type": "BEAST",
            "mode": "discrete",
            "with_regressor": False,
            "action_dim": 7,
            "num_basis": 8,
            "vocab_size": 256,
            "degree_p": 4,
            "gripper_zero_order": True,
            "gripper_dof": 1,
            "future_action_window_size": 10,
            "past_action_window_size": 0,
            "action_horizon": 10,
        }
    )


@FRAMEWORK_REGISTRY.register("QwenBeastFast")
class Qwen_BeastFast(baseframework):
    """Qwen-VL backbone + discrete BEAST tokenizer + AR action tokens."""

    def __init__(self, config: Optional[dict] = None, **kwargs) -> None:
        super().__init__()
        self.config = merge_framework_config(QwenBeastFastDefaultConfig, config)
        self.qwen_vl_interface = get_vlm_model(config=self.config)

        self.config.framework.action_model.seq_len = int(self.config.framework.action_model.action_horizon)
        self.config.framework.action_model.mode = "discrete"
        self.config.framework.action_model.with_regressor = False
        self.action_model = get_action_model(config=self.config)

        self.action_horizon = int(self.config.framework.action_model.action_horizon)
        self.action_dim = int(self.config.framework.action_model.action_dim)
        self.num_action_tokens = int(self.action_model.num_params)

    def _actions_to_tensor(self, actions: List) -> torch.Tensor:
        arr = np.asarray(actions, dtype=np.float32)
        return torch.as_tensor(arr, dtype=torch.float32)

    def _encode_to_vlm_strings(self, actions: List) -> List[str]:
        """Encode action chunks → BEAST discrete tokens → ``<robot_action_k>`` strings."""
        device = next(self.action_model.parameters(), torch.tensor(0.0)).device
        # Prefer tokenizer buffers' device.
        device = self.action_model.tokenizer.w_min.device
        action_t = self._actions_to_tensor(actions).to(device)
        # [B, T, D] — take last action_horizon steps (matches continuous BEAST chunk).
        if action_t.shape[1] > self.action_horizon:
            action_t = action_t[:, -self.action_horizon :, :]
        tokens = self.action_model.encode(action_t, update_bounds=True)  # [B, num_params] long
        return [self.map_beast_tokens_to_vlm_action(seq.tolist()) for seq in tokens]

    def map_beast_tokens_to_vlm_action(self, tokens: List[int]) -> str:
        return "".join(f"<robot_action_{int(token)}>" for token in tokens)

    def forward(self, examples: List[dict] = None, **kwargs) -> Tuple:
        batch_images = [example["image"] for example in examples]
        instructions = [example["lang"] for example in examples]
        actions = [example["action"] for example in examples]

        vlm_action_tokens = self._encode_to_vlm_strings(actions)
        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(
            images=batch_images, instructions=instructions, solutions=vlm_action_tokens
        )

        with torch.autocast("cuda", dtype=torch.bfloat16):
            qwenvl_outputs = self.qwen_vl_interface(
                **qwen_inputs,
                output_attentions=False,
                output_hidden_states=False,
                return_dict=True,
            )

        vlm_action_loss = qwenvl_outputs.loss
        if vlm_action_loss is None or torch.isnan(vlm_action_loss):
            vlm_action_loss = torch.tensor(0.0, device=self.qwen_vl_interface.model.device)

        return {"action_loss": vlm_action_loss}

    @torch.inference_mode()
    def predict_action(self, examples: List[dict] = None, **kwargs) -> np.ndarray:
        if type(examples) is not list:
            examples = [examples]
        batch_images = [to_pil_preserve(example["image"]) for example in examples]
        instructions = [example["lang"] for example in examples]

        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(images=batch_images, instructions=instructions)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            generated_ids = self.qwen_vl_interface.model.generate(**qwen_inputs, max_length=2048)

        batch_vlm_action_token_ids = self._extract_action_token_ids(generated_ids)
        batch_beast_ids = self._decode_action_tokens(batch_vlm_action_token_ids)

        device = self.action_model.tokenizer.w_min.device
        decoded = []
        for ids in batch_beast_ids:
            if ids is None or len(ids) == 0:
                decoded.append(np.zeros((self.action_horizon, self.action_dim), dtype=np.float32))
                continue
            # Pad / crop to expected num_params
            if len(ids) < self.num_action_tokens:
                ids = ids + [0] * (self.num_action_tokens - len(ids))
            ids = ids[: self.num_action_tokens]
            tok = torch.as_tensor(ids, device=device, dtype=torch.long).view(1, -1)
            recon = self.action_model.decode(tok)  # [1, seq_len, dof]
            decoded.append(recon[0].detach().cpu().numpy())

        return {"normalized_actions": np.stack(decoded, axis=0)}

    def _extract_action_token_ids(self, generated_ids: torch.LongTensor) -> List[List[int]]:
        act_min = self.qwen_vl_interface._ACTION_TOKEN_MIN
        act_max = self.qwen_vl_interface._ACTION_TOKEN_MAX
        mask = (generated_ids >= act_min) & (generated_ids <= act_max)
        results = []
        for b in range(generated_ids.size(0)):
            idx = mask[b].nonzero(as_tuple=False).flatten()
            if idx.numel() == 0:
                results.append([])
                continue
            results.append(generated_ids[b, idx].tolist())
        return results

    def _decode_action_tokens(self, batch_vlm_tokens: List[List[int]]) -> List[Any]:
        act_min = self.qwen_vl_interface._ACTION_TOKEN_MIN
        batch_ids = []
        for seq in batch_vlm_tokens:
            if not seq:
                batch_ids.append(None)
                continue
            batch_ids.append([t - act_min for t in seq])
        return batch_ids


if __name__ == "__main__":
    print("QwenBeastFast registered defaults OK:", QwenBeastFastDefaultConfig.name)
