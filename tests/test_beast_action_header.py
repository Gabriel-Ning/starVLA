"""Unit tests for BEAST action representation (beast_ActionHeader)."""

import unittest
from types import SimpleNamespace

import torch

from starVLA.model.modules.action_model.beast_ActionHeader import (
    BeastActionHeader,
    get_action_model,
)


def _smooth_traj(batch: int = 2, seq_len: int = 10, dof: int = 7) -> torch.Tensor:
    t = torch.linspace(0.0, 1.0, seq_len).view(1, seq_len, 1).expand(batch, seq_len, dof)
    return (t + 0.1 * torch.sin(2.0 * torch.pi * t)).contiguous()


class BeastActionHeaderTest(unittest.TestCase):
    def test_continuous_roundtrip_on_smooth_traj(self):
        header = BeastActionHeader(
            action_dim=7,
            num_basis=8,
            seq_len=10,
            mode="continuous",
            with_regressor=False,
            device="cpu",
        )
        traj = _smooth_traj()
        weights = header.encode(traj, update_bounds=True)
        recon = header.decode(weights)

        self.assertEqual(tuple(weights.shape), (2, 8 * 7))
        self.assertEqual(tuple(recon.shape), (2, 10, 7))
        self.assertLess(float((traj - recon).pow(2).mean()), 1e-3)

    def test_discrete_roundtrip_on_smooth_traj(self):
        header = BeastActionHeader(
            action_dim=7,
            num_basis=8,
            seq_len=10,
            mode="discrete",
            with_regressor=False,
            device="cpu",
        )
        traj = _smooth_traj()
        tokens = header.encode(traj, update_bounds=True)
        recon = header.decode(tokens)

        self.assertEqual(tuple(tokens.shape), (2, 8 * 7))
        self.assertEqual(tokens.dtype, torch.int64)
        self.assertTrue(torch.all(tokens >= 0))
        self.assertTrue(torch.all(tokens < header.vocab_size))
        self.assertEqual(tuple(recon.shape), (2, 10, 7))
        self.assertLess(float((traj - recon).pow(2).mean()), 5e-3)

    def test_predict_action_backprop(self):
        header = BeastActionHeader(
            action_dim=7,
            num_basis=8,
            seq_len=10,
            mode="continuous",
            with_regressor=True,
            action_hidden_dim=64,
            device="cpu",
        )
        # Seed bounds so differentiable decode has valid w_min/w_max.
        header.encode(_smooth_traj(), update_bounds=True)

        hidden = torch.randn(2, 10, 64, requires_grad=True)
        pred = header.predict_action(hidden)
        loss = pred.pow(2).mean()
        loss.backward()

        self.assertEqual(tuple(pred.shape), (2, 10, 7))
        self.assertIsNotNone(hidden.grad)

    def test_get_action_model_factory(self):
        cfg = SimpleNamespace(
            framework=SimpleNamespace(
                action_model=SimpleNamespace(
                    action_dim=7,
                    action_horizon=10,
                    num_basis=8,
                    mode="continuous",
                    with_regressor=True,
                    action_hidden_dim=64,
                    vocab_size=256,
                    degree_p=3,
                    gripper_zero_order=True,
                    gripper_dof=1,
                )
            )
        )
        model = get_action_model(cfg)
        self.assertIsInstance(model, BeastActionHeader)
        self.assertEqual(model.mode, "continuous")
        self.assertEqual(model.num_params, 56)


if __name__ == "__main__":
    unittest.main()
