# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
"""
QwenBeast Framework — Qwen-VL + continuous BEAST action representation.

Fork of QwenOFT: same special-token gathering + L1 regression path, but the
action head encodes/decodes B-spline weights (BEAST-F style continuous space).
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import torch
from PIL import Image

from deployment.model_server.tools.image_tools import to_pil_preserve
from starVLA.model.framework.base_framework import baseframework
from starVLA.model.framework.share_tools import add_discretized_state_to_instruction, merge_framework_config
from starVLA.model.modules.action_model.beast_ActionHeader import get_action_model
from starVLA.model.modules.vlm import get_vlm_model
from starVLA.model.tools import FRAMEWORK_REGISTRY
from starVLA.training.trainer_utils import initialize_overwatch
from starVLA.training.trainer_utils.trainer_tools import resize_images

logger = initialize_overwatch(__name__)


def masked_l1_loss(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    error = torch.abs(prediction - target)
    if mask is None:
        return error.mean()
    valid = mask.to(device=error.device, dtype=error.dtype)
    if valid.shape != error.shape:
        raise ValueError(f"action_mask shape {tuple(valid.shape)} != action shape {tuple(error.shape)}")
    denominator = valid.sum()
    if not torch.any(mask):
        raise ValueError("action_mask contains no valid targets")
    return (error * valid).sum() / denominator


@dataclass
class QwenBeastDefaultConfig:
    """Qwen + continuous BEAST (OFT-style regression on B-spline reconstruction)."""

    name: str = "QwenBeast"

    qwenvl: dict = field(
        default_factory=lambda: {
            "base_vlm": "./playground/Pretrained_models/Qwen3-VL-4B-Instruct-Action",
            "attn_implementation": "flash_attention_2",
        }
    )

    action_model: dict = field(
        default_factory=lambda: {
            "action_model_type": "BEAST",
            "mode": "continuous",
            "with_regressor": True,
            "action_dim": 7,
            "action_hidden_dim": 2560,
            "num_basis": 8,
            "vocab_size": 256,
            "degree_p": 4,
            "gripper_zero_order": True,
            "gripper_dof": 1,
            # chunk length = BEAST seq_len (also used as # of action query tokens)
            "future_action_window_size": 10,
            "past_action_window_size": 0,
            "action_horizon": 10,
        }
    )


@FRAMEWORK_REGISTRY.register("QwenBeast")
class Qwen_Beast(baseframework):
    """Qwen-VL backbone + continuous BEAST action header."""

    def __init__(self, config: Optional[dict] = None, **kwargs) -> None:
        super().__init__()
        self.config = merge_framework_config(QwenBeastDefaultConfig, config)
        self.qwen_vl_interface = get_vlm_model(config=self.config)
        self.config.framework.action_model.action_hidden_dim = self.qwen_vl_interface.model.config.hidden_size
        # Keep BEAST seq_len aligned with action_horizon.
        self.config.framework.action_model.seq_len = int(self.config.framework.action_model.action_horizon)
        self.config.framework.action_model.mode = "continuous"
        self.config.framework.action_model.with_regressor = True
        self.action_model = get_action_model(config=self.config)

        self.action_horizon = int(self.config.framework.action_model.action_horizon)
        self.chunk_len = self.action_horizon
        self.action_token = "🔍"
        self.action_token_id = self.qwen_vl_interface.processor.tokenizer("🔍", add_special_tokens=False)["input_ids"][0]

    def forward(self, examples: List[dict] = None, **kwargs) -> Tuple:
        batch_images = [example["image"] for example in examples]
        instructions = [example["lang"] for example in examples]
        actions = [example["action"] for example in examples]
        action_masks = [example.get("action_mask") for example in examples]
        if any(mask is not None for mask in action_masks) and not all(mask is not None for mask in action_masks):
            raise ValueError("action_mask must be present for every example in a batch or for none")
        state = [example["state"] for example in examples] if "state" in examples[0] else None
        instructions = (
            self.add_discretized_state_to_instruction(instructions, state) if state is not None else instructions
        )

        action_tokens = self.action_token * self.chunk_len
        prompt_suffix = f" Please predict the next {self.chunk_len} robot actions: <action>{action_tokens}<action>."
        instructions = [instruction + prompt_suffix for instruction in instructions]

        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(images=batch_images, instructions=instructions)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            qwenvl_outputs = self.qwen_vl_interface(
                **qwen_inputs,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
            )
            last_hidden = qwenvl_outputs.hidden_states[-1]

        with torch.autocast("cuda", dtype=torch.float32):
            input_ids = qwen_inputs.get("input_ids", None)
            action_queries = self._gather_action_token_embeddings(
                last_hidden, input_ids, action_token_id=self.action_token_id
            )
            pred_actions = self.action_model.predict_action(action_queries)

            actions = torch.tensor(np.array(actions), device=pred_actions.device, dtype=pred_actions.dtype)
            actions_target = actions[:, -self.action_horizon :, :]

            # Refresh BEAST normalization bounds from GT chunks (no grad).
            with torch.no_grad():
                self.action_model.encode(actions_target, update_bounds=True)

            action_mask = None
            if action_masks[0] is not None:
                action_mask = torch.as_tensor(
                    np.asarray(action_masks), device=pred_actions.device, dtype=torch.bool
                )[:, -self.action_horizon :, :]
            action_loss = masked_l1_loss(pred_actions, actions_target, action_mask)

        return {"action_loss": action_loss}

    @torch.inference_mode()
    def predict_action(self, examples: List[dict] = None, **kwargs) -> np.ndarray:
        if type(examples) is not list:
            examples = [examples]
        batch_images = [to_pil_preserve(example["image"]) for example in examples]
        instructions = [example["lang"] for example in examples]
        state = [example["state"] for example in examples] if "state" in examples[0] else None
        instructions = (
            self.add_discretized_state_to_instruction(instructions, state) if state is not None else instructions
        )

        train_obs_image_size = getattr(self.config.datasets.vla_data, "obs_image_size", None)
        if train_obs_image_size:
            batch_images = resize_images(batch_images, target_size=train_obs_image_size)

        action_tokens = self.action_token * self.chunk_len
        prompt_suffix = f" Please predict the next {self.chunk_len} robot actions: <action>{action_tokens}<action>."
        instructions = [instruction + prompt_suffix for instruction in instructions]

        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(images=batch_images, instructions=instructions)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            qwenvl_outputs = self.qwen_vl_interface(
                **qwen_inputs,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
            )
            last_hidden = qwenvl_outputs.hidden_states[-1]

        with torch.autocast("cuda", dtype=torch.float32):
            input_ids = qwen_inputs.get("input_ids", None)
            action_queries = self._gather_action_token_embeddings(
                last_hidden, input_ids, action_token_id=self.action_token_id
            )
            pred_actions = self.action_model.predict_action(action_queries)

        return {"normalized_actions": pred_actions.detach().cpu().numpy()}

    def _gather_action_token_embeddings(
        self,
        last_hidden: torch.Tensor,
        input_ids: torch.Tensor,
        action_token_id=None,
    ) -> torch.Tensor:
        if action_token_id is None:
            raise ValueError("action_token_id must not be None")

        device = input_ids.device
        B, L, H = last_hidden.shape

        if isinstance(action_token_id, (list, tuple, set)):
            id_list = torch.tensor(list(action_token_id), device=device, dtype=input_ids.dtype)
            mask = torch.isin(input_ids, id_list)
        else:
            mask = input_ids == action_token_id

        counts = mask.sum(dim=1)
        if (counts < self.chunk_len).any():
            insufficient = (counts < self.chunk_len).nonzero(as_tuple=False).flatten().tolist()
            raise RuntimeError(
                f"The following samples have insufficient action tokens (< {self.chunk_len}): {insufficient} |"
                f" counts={counts.tolist()}"
            )

        idx = torch.arange(L, device=device).unsqueeze(0).expand(B, L)
        masked_pos = torch.where(mask, idx, torch.full_like(idx, -1))
        topk_pos = masked_pos.topk(k=self.chunk_len, dim=-1).values
        selected_pos = topk_pos.sort(dim=-1).values
        expanded_index = selected_pos.unsqueeze(-1).expand(-1, -1, H)
        return last_hidden.gather(dim=1, index=expanded_index)

    add_discretized_state_to_instruction = staticmethod(add_discretized_state_to_instruction)


if __name__ == "__main__":
    from omegaconf import OmegaConf

    cfg = OmegaConf.create({"framework": {"name": "QwenBeast"}})
    print("QwenBeast registered defaults OK:", QwenBeastDefaultConfig.name)
