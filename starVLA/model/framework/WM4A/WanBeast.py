# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
"""
WanBeast Framework — Wan2.2-TI2V World Model + continuous BEAST action head.

Fork of WanOFT: same WM pooling path, replace MLP action head with BEAST
continuous encode/decode + regressor.
"""

import sys
from pathlib import Path

_workspace_root = Path(__file__).parent.parent.parent.parent.parent
if str(_workspace_root) not in sys.path:
    sys.path.insert(0, str(_workspace_root))

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from deployment.model_server.tools.image_tools import to_pil_preserve
from starVLA.model.framework.base_framework import baseframework
from starVLA.model.framework.share_tools import add_discretized_state_to_instruction, merge_framework_config
from starVLA.model.modules.action_model.beast_ActionHeader import get_action_model
from starVLA.model.modules.world_model import get_world_model
from starVLA.model.tools import FRAMEWORK_REGISTRY
from starVLA.training.trainer_utils import initialize_overwatch
from starVLA.training.trainer_utils.trainer_tools import resize_images

logger = initialize_overwatch(__name__)


@dataclass
class WanBeastDefaultConfig:
    """Wan WM backbone + continuous BEAST action representation."""

    name: str = "WanBeast"

    world_model: dict = field(
        default_factory=lambda: {
            "base_wm": "./playground/Pretrained_models/Wan-AI/Wan2.2-TI2V-5B-Diffusers",
            "extract_layers": [-1],
        }
    )

    qwenvl: dict = field(
        default_factory=lambda: {
            "base_vlm": "./playground/Pretrained_models/Wan-AI/Wan2.2-TI2V-5B-Diffusers",
        }
    )

    action_model: dict = field(
        default_factory=lambda: {
            "action_model_type": "BEAST",
            "mode": "continuous",
            "with_regressor": True,
            "action_dim": 7,
            "action_hidden_dim": 3072,
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


@FRAMEWORK_REGISTRY.register("WanBeast")
class Wan_Beast(baseframework):
    """Wan2.2-TI2V backbone + continuous BEAST action header."""

    def __init__(self, config: Optional[dict] = None, **kwargs) -> None:
        super().__init__()
        self.config = merge_framework_config(WanBeastDefaultConfig, config)

        self.backbone = get_world_model(config=self.config)
        wm_hidden = self.backbone.model.config.hidden_size
        self.config.framework.action_model.action_hidden_dim = wm_hidden
        self.config.framework.action_model.seq_len = int(self.config.framework.action_model.action_horizon)
        self.config.framework.action_model.mode = "continuous"
        self.config.framework.action_model.with_regressor = True
        self.action_model = get_action_model(config=self.config)

        self.action_horizon = int(self.config.framework.action_model.action_horizon)
        self.chunk_len = self.action_horizon
        self.action_query_proj = nn.Linear(wm_hidden, self.chunk_len * wm_hidden)
        self.l1_loss = nn.L1Loss()

    def _pool_to_action_queries(self, hidden_states: torch.Tensor) -> torch.Tensor:
        B, N, H = hidden_states.shape
        pooled = hidden_states.mean(dim=1)
        queries = self.action_query_proj(pooled)
        return queries.view(B, self.chunk_len, H)

    def forward(self, examples: List[dict] = None, **kwargs) -> Tuple:
        batch_images = [example["image"] for example in examples]
        instructions = [example["lang"] for example in examples]
        actions = [example["action"] for example in examples]
        state = [example["state"] for example in examples] if "state" in examples[0] else None
        instructions = (
            add_discretized_state_to_instruction(instructions, state) if state is not None else instructions
        )

        wm_inputs = self.backbone.build_inputs(images=batch_images, instructions=instructions)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            wm_outputs = self.backbone(**wm_inputs, output_hidden_states=True, return_dict=True)
            last_hidden = wm_outputs.hidden_states[-1]

        with torch.autocast("cuda", dtype=torch.float32):
            action_queries = self._pool_to_action_queries(last_hidden)
            pred_actions = self.action_model.predict_action(action_queries)

            actions = torch.tensor(np.array(actions), device=pred_actions.device, dtype=pred_actions.dtype)
            actions_target = actions[:, -self.action_horizon :, :]
            with torch.no_grad():
                self.action_model.encode(actions_target, update_bounds=True)
            action_loss = self.l1_loss(pred_actions, actions_target)

        return {"action_loss": action_loss}

    @torch.inference_mode()
    def predict_action(self, examples: List[dict], **kwargs) -> np.ndarray:
        if type(examples) is not list:
            examples = [examples]
        batch_images = [to_pil_preserve(example["image"]) for example in examples]
        instructions = [example["lang"] for example in examples]
        state = [example["state"] for example in examples] if "state" in examples[0] else None
        instructions = (
            add_discretized_state_to_instruction(instructions, state) if state is not None else instructions
        )

        train_obs_image_size = getattr(self.config.datasets.vla_data, "obs_image_size", None)
        if train_obs_image_size:
            batch_images = resize_images(batch_images, target_size=train_obs_image_size)

        wm_inputs = self.backbone.build_inputs(images=batch_images, instructions=instructions)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            wm_outputs = self.backbone(**wm_inputs, output_hidden_states=True, return_dict=True)
            last_hidden = wm_outputs.hidden_states[-1]

        with torch.autocast("cuda", dtype=torch.float32):
            action_queries = self._pool_to_action_queries(last_hidden)
            pred_actions = self.action_model.predict_action(action_queries)

        return {"normalized_actions": pred_actions.detach().cpu().numpy()}


if __name__ == "__main__":
    print("WanBeast registered defaults OK:", WanBeastDefaultConfig.name)
