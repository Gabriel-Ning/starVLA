# Copyright 2025 starVLA community / BEAST authors. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
"""BEAST Action Header for StarVLA.

B-Spline Encoded Action Sequences (BEAST) as a pluggable action representation.

This module provides:
  - ``BSpline`` / ``BeastTokenizer``: encode action chunks to continuous B-spline
    weights or discrete tokens, and decode back to trajectories.
  - ``BeastActionHeader``: StarVLA action-model wrapper with ``continuous`` /
    ``discrete`` modes and an optional MLP regressor for OFT-style training.
  - ``get_action_model(config)``: factory matching other ``*_ActionHeader.py`` files.

Core tokenizer logic adapted from ``beast_calvin/beast_tokenizer.py``
(Zhou et al., NeurIPS 2025).
"""

from __future__ import annotations

from functools import wraps
from typing import ClassVar, Optional

import numpy as np
import torch
import torch.nn as nn


def autocast_float32(fn):
    """Decorator to ensure computation runs in float32 precision."""
    @wraps(fn)
    def wrapped(*args, **kwargs):
        if hasattr(torch.amp, 'autocast'):
            with torch.amp.autocast('cuda', dtype=torch.float32):
                return fn(*args, **kwargs)
        else:
            with torch.cuda.amp.autocast(dtype=torch.float32):
                return fn(*args, **kwargs)
    return wrapped


# =============================================================================
# Utility Functions
# =============================================================================

def continuous_to_discrete(tensor: torch.Tensor, min_val: torch.Tensor = None,
                           max_val: torch.Tensor = None, num_bins: int = 256) -> torch.Tensor:
    """
    Convert continuous tensor values to discrete tokens.

    Args:
        tensor: Input tensor with continuous values
        min_val: Minimum value for normalization (uses tensor.min() if None)
        max_val: Maximum value for normalization (uses tensor.max() if None)
        num_bins: Number of discrete bins (default 256 for 0-255 range)

    Returns:
        Discretized tensor with integer values in [0, num_bins-1]
    """
    if min_val is None:
        min_val = tensor.min()
    if max_val is None:
        max_val = tensor.max()

    assert torch.all(tensor >= min_val - 1e-3), "Input tensor has values below min_val"
    assert torch.all(tensor <= max_val + 1e-3), "Input tensor has values above max_val"

    normalized = (tensor - min_val) / (max_val - min_val)
    normalized = torch.clamp(normalized, 0, 1)
    discrete = torch.round(normalized * (num_bins - 1)).to(torch.long)
    return discrete


def discrete_to_continuous(discrete_tensor: torch.Tensor, min_val: torch.Tensor = 0,
                           max_val: torch.Tensor = 1, num_bins: int = 256) -> torch.Tensor:
    """
    Convert discrete tokens back to continuous values.

    Args:
        discrete_tensor: Input tensor with discrete values in [0, num_bins-1]
        min_val: Minimum value of target continuous range
        max_val: Maximum value of target continuous range
        num_bins: Number of discrete bins

    Returns:
        Continuous tensor with values in [min_val, max_val]
    """
    normalized = discrete_tensor.float() / (num_bins - 1)
    continuous = normalized * (max_val - min_val) + min_val
    return torch.clamp(continuous, min_val, max_val)


def normalize_tensor(tensor: torch.Tensor, w_min: torch.Tensor, w_max: torch.Tensor,
                     norm_min: float = -1.0, norm_max: float = 1.0) -> torch.Tensor:
    """
    Normalize tensor from [w_min, w_max] to [norm_min, norm_max].

    Args:
        tensor: Input tensor to normalize
        w_min: Minimum bound of original range
        w_max: Maximum bound of original range
        norm_min: Target minimum (default -1.0)
        norm_max: Target maximum (default 1.0)

    Returns:
        Normalized tensor in [norm_min, norm_max]
    """
    clipped = torch.clamp(tensor, w_min, w_max)
    normalized = (clipped - w_min) / (w_max - w_min)
    return normalized * (norm_max - norm_min) + norm_min


def denormalize_tensor(normalized_tensor: torch.Tensor, w_min: torch.Tensor, w_max: torch.Tensor,
                       norm_min: float = -1.0, norm_max: float = 1.0) -> torch.Tensor:
    """
    Denormalize tensor from [norm_min, norm_max] back to [w_min, w_max].

    Args:
        normalized_tensor: Normalized input tensor
        w_min: Target minimum bound
        w_max: Target maximum bound
        norm_min: Source minimum (default -1.0)
        norm_max: Source maximum (default 1.0)

    Returns:
        Denormalized tensor in [w_min, w_max]
    """
    clipped = torch.clamp(normalized_tensor, norm_min, norm_max)
    denormalized = (clipped - norm_min) / (norm_max - norm_min)
    return denormalized * (w_max - w_min) + w_min


# =============================================================================
# BSpline Class (Merged from UniBSplineBasis + UniformBSpline)
# =============================================================================

class BSpline(torch.nn.Module):
    """
    Uniform B-Spline for trajectory representation and fitting.

    Combines B-spline basis function computation with trajectory fitting and
    reconstruction. Supports position, velocity, and acceleration computation.

    Args:
        num_basis: Number of B-spline basis functions (control points for free params)
        degree: B-spline degree (3=cubic, 4=quartic, 0=piecewise constant)
        num_dof: Degrees of freedom (e.g., 7 for robot arm)
        tau: Time duration of trajectory (default 1.0, normalized time)
        init_cond_order: Order of initial conditions (0=none, 1=pos, 2=pos+vel)
        end_cond_order: Order of end conditions (0=none, 1=pos, 2=pos+vel)
        dtype: Torch data type (default float32)
        device: Torch device ('cuda' or 'cpu')

    Example:
        >>> bspline = BSpline(num_basis=10, degree=4, num_dof=7, device='cuda')
        >>> result = bspline.learn_mp_params_from_trajs(times, trajectories)
        >>> reconstructed = bspline.get_traj_pos(times, result['params'])
    """

    def __init__(self, num_basis: int = 10, degree: int = 3, num_dof: int = 1,
                 tau: float = 1.0, init_cond_order: int = 0, end_cond_order: int = 0,
                 dtype: torch.dtype = torch.float32, device: str = 'cpu'):
        super().__init__()

        self.num_basis = num_basis
        self.degree = degree
        self.num_dof = num_dof
        self.init_cond_order = init_cond_order
        self.end_cond_order = end_cond_order
        self._dtype = dtype
        self._device = device

        # Number of control points = basis + boundary conditions
        self.num_ctrlp = num_basis + init_cond_order + end_cond_order

        # Create uniform knot vector
        num_knots = self.degree + 1 + self.num_ctrlp
        num_internal = num_knots - 2 * self.degree
        knots = torch.linspace(0, 1, num_internal, dtype=dtype, device=device)
        knots = torch.cat([
            torch.zeros(self.degree, dtype=dtype, device=device),
            knots,
            torch.ones(self.degree, dtype=dtype, device=device)
        ])
        self.register_buffer("knots", knots, persistent=False)
        self.register_buffer("tau", torch.tensor(tau, dtype=dtype, device=device), persistent=False)

        # Runtime state
        self.times = None
        self.params = None
        self.init_pos = None
        self.init_vel = None
        self.end_pos = None
        self.end_vel = None
        self.params_init = None
        self.params_end = None
        self._pos_cache = None
        self._vel_cache = None
        self.add_dim = []

    @property
    def device(self):
        return self.knots.device

    @property
    def dtype(self):
        return self.knots.dtype

    @property
    def num_params(self) -> int:
        """Total number of learnable parameters."""
        return self.num_basis * self.num_dof

    def _clear_cache(self):
        """Clear cached computation results."""
        self._pos_cache = None
        self._vel_cache = None

    def _time_to_phase(self, times: torch.Tensor) -> torch.Tensor:
        """Convert times to normalized phase [0, 1]."""
        tau = times.reshape(-1)[-1]
        self.tau.copy_(tau)
        return torch.clip(times / self.tau[..., None], 0, 1)

    def _basis_function(self, i: int, k: int, knots: torch.Tensor,
                        u: torch.Tensor, num_ctrlp: int = None) -> torch.Tensor:
        """
        Compute B-spline basis using de Boor's recursive algorithm.

        Args:
            i: Basis function index
            k: Current degree level
            knots: Knot vector
            u: Evaluation points (phase values)
            num_ctrlp: Number of control points (for boundary handling)

        Returns:
            Basis function values at evaluation points
        """
        if num_ctrlp is None:
            num_ctrlp = self.num_ctrlp

        if k == 0:
            # Base case: piecewise constant
            if i == num_ctrlp - 1:
                # Handle right endpoint (closed interval)
                return torch.where((u >= knots[i]) & (u <= knots[i + 1]),
                                   1.0, 0.0).to(dtype=self.dtype, device=self.device)
            else:
                return torch.where((u >= knots[i]) & (u < knots[i + 1]),
                                   1.0, 0.0).to(dtype=self.dtype, device=self.device)
        else:
            # Recursive case
            denom1 = knots[i + k] - knots[i]
            term1 = 0.0 if denom1 == 0 else (u - knots[i]) / denom1 * \
                    self._basis_function(i, k - 1, knots, u, num_ctrlp)

            denom2 = knots[i + k + 1] - knots[i + 1]
            term2 = 0.0 if denom2 == 0 else (knots[i + k + 1] - u) / denom2 * \
                    self._basis_function(i + 1, k - 1, knots, u, num_ctrlp)

            return term1 + term2

    def basis(self, times: torch.Tensor) -> torch.Tensor:
        """
        Compute B-spline basis values at given time points.

        Args:
            times: Time points tensor of shape [*batch, num_times]

        Returns:
            Basis values of shape [*batch, num_times, num_ctrlp]
        """
        phase = self._time_to_phase(times)
        basis = [self._basis_function(i, self.degree, self.knots, phase)
                 for i in range(self.num_ctrlp)]
        return torch.stack(basis, dim=-1)

    def vel_basis(self, times: torch.Tensor) -> torch.Tensor:
        """
        Compute velocity B-spline basis (derivative of position basis).

        Args:
            times: Time points tensor

        Returns:
            Velocity basis values of shape [*batch, num_times, num_ctrlp-1]
        """
        phase = self._time_to_phase(times)
        vel_knots = self.knots[1:-1]
        basis = [self._basis_function(i, self.degree - 1, vel_knots, phase,
                                       num_ctrlp=self.num_ctrlp - 1)
                 for i in range(self.num_ctrlp - 1)]
        return torch.stack(basis, dim=-1)

    def acc_basis(self, times: torch.Tensor) -> torch.Tensor:
        """
        Compute acceleration B-spline basis (second derivative).

        Args:
            times: Time points tensor

        Returns:
            Acceleration basis values of shape [*batch, num_times, num_ctrlp-2]
        """
        phase = self._time_to_phase(times)
        acc_knots = self.knots[2:-2]
        basis = [self._basis_function(i, self.degree - 2, acc_knots, phase,
                                       num_ctrlp=self.num_ctrlp - 2)
                 for i in range(self.num_ctrlp - 2)]
        return torch.stack(basis, dim=-1)

    def velocity_control_points(self, ctrl_pts: torch.Tensor) -> torch.Tensor:
        """
        Compute velocity control points from position control points.

        Args:
            ctrl_pts: Position control points [*batch, num_dof, num_ctrlp]

        Returns:
            Velocity control points [*batch, num_dof, num_ctrlp-1]
        """
        diff = ctrl_pts[..., 1:] - ctrl_pts[..., :-1]
        delta = self.knots[1 + self.degree:self.num_ctrlp + self.degree] - \
                self.knots[1:self.num_ctrlp]
        return diff * (self.degree / delta)

    def _compute_init_params(self, init_pos: torch.Tensor,
                             init_vel: torch.Tensor = None) -> Optional[torch.Tensor]:
        """Compute initial boundary condition control points."""
        if self.init_cond_order == 0:
            return None

        params = init_pos[..., None]
        if self.init_cond_order == 2 and init_vel is not None:
            p1 = init_vel * self.tau * (self.knots[1 + self.degree] - self.knots[1]) / self.degree + init_pos
            params = torch.cat([params, p1[..., None]], dim=-1)
        return params

    def _compute_end_params(self, end_pos: torch.Tensor,
                            end_vel: torch.Tensor = None) -> Optional[torch.Tensor]:
        """Compute end boundary condition control points."""
        if self.end_cond_order == 0:
            return None

        params = end_pos[..., None]
        if self.end_cond_order == 2 and end_vel is not None:
            pn = end_pos - end_vel * self.tau * \
                 (self.knots[self.num_ctrlp - 1 + self.degree] - self.knots[self.num_ctrlp - 1]) * self.degree
            params = torch.cat([pn[..., None], params], dim=-1)
        return params

    def set_times(self, times: torch.Tensor):
        """
        Set evaluation time points.

        Args:
            times: Time points [*batch, num_times]
        """
        self.times = torch.as_tensor(times, dtype=self.dtype, device=self.device)
        tau = times.reshape(-1)[-1]
        self.tau.copy_(tau)
        self._clear_cache()

    def set_params(self, params: torch.Tensor) -> torch.Tensor:
        """
        Set B-spline parameters (control point weights).

        Args:
            params: Parameters [*batch, num_params]

        Returns:
            Any unused parameters (for chaining)
        """
        params = torch.as_tensor(params, dtype=self.dtype, device=self.device)
        assert params.shape[-1] == self.num_params
        self.add_dim = list(params.shape[:-1])
        self.params = params[..., :self.num_params]
        self._clear_cache()
        return params[..., self.num_params:]

    def set_duration(self, duration: float, dt: float):
        """
        Set trajectory duration and generate time grid.

        Args:
            duration: Total trajectory duration
            dt: Time step (control frequency)
        """
        times = torch.linspace(0, duration, int(round(duration / dt)) + 1,
                               dtype=self.dtype, device=self.device)
        # Expand for batch dimensions
        for _ in self.add_dim:
            times = times.unsqueeze(0)
        times = times.expand(*self.add_dim, -1)
        self.set_times(times)

    def set_initial_conditions(self, init_pos: torch.Tensor, init_vel: torch.Tensor = None):
        """
        Set initial position and velocity conditions.

        Args:
            init_pos: Initial position [*batch, num_dof]
            init_vel: Initial velocity [*batch, num_dof] (optional)
        """
        self.init_pos = torch.as_tensor(init_pos, dtype=self.dtype, device=self.device)
        self.init_vel = torch.as_tensor(init_vel, dtype=self.dtype, device=self.device) if init_vel is not None else None
        self.params_init = self._compute_init_params(self.init_pos, self.init_vel)
        self._clear_cache()

    def set_end_conditions(self, end_pos: torch.Tensor, end_vel: torch.Tensor = None):
        """
        Set end position and velocity conditions.

        Args:
            end_pos: End position [*batch, num_dof]
            end_vel: End velocity [*batch, num_dof] (optional)
        """
        self.end_pos = torch.as_tensor(end_pos, dtype=self.dtype, device=self.device) if end_pos is not None else None
        self.end_vel = torch.as_tensor(end_vel, dtype=self.dtype, device=self.device) if end_vel is not None else None
        self.params_end = self._compute_end_params(self.end_pos, self.end_vel)
        self._clear_cache()

    def update_inputs(self, times: torch.Tensor = None, params: torch.Tensor = None,
                      init_pos: torch.Tensor = None, init_vel: torch.Tensor = None, **kwargs):
        """
        Update multiple inputs at once.

        Args:
            times: Time points
            params: B-spline parameters
            init_pos: Initial position
            init_vel: Initial velocity
            **kwargs: Additional args (end_pos, end_vel)
        """
        if params is not None:
            self.set_params(params)
        if times is not None:
            self.set_times(times)
        if init_pos is not None:
            self.set_initial_conditions(init_pos, init_vel)
        if kwargs.get('end_pos') is not None or kwargs.get('end_vel') is not None:
            self.set_end_conditions(kwargs.get('end_pos'), kwargs.get('end_vel'))

    def _get_full_params(self) -> torch.Tensor:
        """Get full control points including boundary conditions."""
        params = self.params.reshape(*self.add_dim, self.num_dof, -1)
        if self.params_init is not None:
            params = torch.cat([self.params_init, params], dim=-1)
        if self.params_end is not None:
            params = torch.cat([params, self.params_end], dim=-1)
        return params

    def get_traj_pos(self, times: torch.Tensor = None, params: torch.Tensor = None,
                     init_pos: torch.Tensor = None, init_vel: torch.Tensor = None,
                     flat_shape: bool = False, **kwargs) -> torch.Tensor:
        """
        Compute trajectory positions from B-spline parameters.

        Args:
            times: Time points [*batch, num_times]
            params: B-spline parameters [*batch, num_params]
            init_pos: Initial position (optional)
            init_vel: Initial velocity (optional)
            flat_shape: If True, return flattened [*batch, num_dof*num_times]

        Returns:
            Position trajectory [*batch, num_times, num_dof] or flattened
        """
        self.update_inputs(times, params, init_pos, init_vel, **kwargs)

        if self._pos_cache is not None:
            pos = self._pos_cache
        else:
            assert self.params is not None
            full_params = self._get_full_params()
            basis = self.basis(self.times)
            # Einsum: [*batch, num_times, num_ctrlp] @ [*batch, num_dof, num_ctrlp]
            pos = torch.einsum('...ik,...jk->...ij', basis, full_params)
            self._pos_cache = pos

        if flat_shape:
            pos = torch.einsum('...ji->...ij', pos).reshape(*self.add_dim, -1)
        return pos

    def get_traj_vel(self, times: torch.Tensor = None, params: torch.Tensor = None,
                     init_pos: torch.Tensor = None, init_vel: torch.Tensor = None,
                     flat_shape: bool = False, **kwargs) -> torch.Tensor:
        """
        Compute trajectory velocities from B-spline parameters.

        Args:
            times: Time points [*batch, num_times]
            params: B-spline parameters [*batch, num_params]
            init_pos: Initial position (optional)
            init_vel: Initial velocity (optional)
            flat_shape: If True, return flattened [*batch, num_dof*num_times]

        Returns:
            Velocity trajectory [*batch, num_times, num_dof] or flattened
        """
        self.update_inputs(times, params, init_pos, init_vel, **kwargs)

        if self._vel_cache is not None:
            vel = self._vel_cache
        else:
            assert self.params is not None
            full_params = self._get_full_params()
            vel_ctrlp = self.velocity_control_points(full_params) / self.tau
            vel_basis = self.vel_basis(self.times)
            vel = torch.einsum('...ik,...jk->...ij', vel_basis, vel_ctrlp)
            self._vel_cache = vel

        if flat_shape:
            vel = torch.einsum('...ji->...ij', vel).reshape(*self.add_dim, -1)
        return vel

    def _basis_multi_dofs(self, times: torch.Tensor) -> torch.Tensor:
        """
        Compute multi-DOF basis matrix for least squares fitting.

        Args:
            times: Time points [*batch, num_times]

        Returns:
            Block-diagonal basis [*batch, num_dof*num_times, num_dof*num_basis]
        """
        add_dim = list(times.shape[:-1])
        num_times = times.shape[-1]
        basis_single = self.basis(times)[..., self.init_cond_order:self.num_ctrlp - self.end_cond_order]

        basis_multi = torch.zeros(*add_dim, self.num_dof * num_times, self.num_dof * self.num_basis,
                                  dtype=self.dtype, device=self.device)
        for i in range(self.num_dof):
            row_slice = slice(i * num_times, (i + 1) * num_times)
            col_slice = slice(i * self.num_basis, (i + 1) * self.num_basis)
            basis_multi[..., row_slice, col_slice] = basis_single

        return basis_multi

    def learn_mp_params_from_trajs(self, times: torch.Tensor, trajs: torch.Tensor,
                                   reg: float = 1e-4, **kwargs) -> dict:
        """
        Learn B-spline parameters from trajectory data via least squares.

        Args:
            times: Time points [*batch, num_times]
            trajs: Trajectory data [*batch, num_times, num_dof]
            reg: Regularization coefficient (default 1e-4)
            **kwargs: Optional init_pos, init_vel, end_pos, end_vel

        Returns:
            Dict with 'params' and boundary conditions
        """
        assert trajs.shape[:-1] == times.shape
        assert trajs.shape[-1] == self.num_dof

        times = torch.as_tensor(times, dtype=self.dtype, device=self.device)
        trajs = torch.as_tensor(trajs, dtype=self.dtype, device=self.device)

        self.add_dim = list(trajs.shape[:-2])
        self.set_times(times)

        # Initialize dummy params for boundary condition contribution
        dummy_params = torch.zeros(*self.add_dim, self.num_dof, self.num_basis,
                                   device=self.device, dtype=self.dtype)

        # Handle initial conditions
        if self.init_cond_order != 0:
            init_pos = kwargs.get("init_pos", trajs[..., 0, :])
            dt = times[..., 1] - times[..., 0]
            init_vel = kwargs.get("init_vel", torch.diff(trajs, dim=-2)[..., 0, :] / dt[..., None])
            self.set_initial_conditions(init_pos, init_vel)
            if self.params_init is not None:
                dummy_params = torch.cat([self.params_init, dummy_params], dim=-1)

        # Handle end conditions
        if self.end_cond_order != 0:
            end_pos = kwargs.get("end_pos", trajs[..., -1, :])
            dt = times[..., 1] - times[..., 0]
            end_vel = kwargs.get("end_vel", torch.diff(trajs, dim=-2)[..., -1, :] / dt[..., None])
            self.set_end_conditions(end_pos, end_vel)
            if self.params_end is not None:
                dummy_params = torch.cat([dummy_params, self.params_end], dim=-1)

        # Compute position from boundary conditions only
        basis_single = self.basis(times)
        pos_boundary = torch.einsum('...ik,...jk->...ij', basis_single, dummy_params)
        pos_boundary = torch.einsum('...ij->...ji', pos_boundary).reshape(*self.add_dim, -1)

        # Build least squares system: A @ w = B
        basis_multi = self._basis_multi_dofs(self.times)
        A = torch.einsum('...ki,...kj->...ij', basis_multi, basis_multi)
        A += torch.eye(self.num_params, dtype=self.dtype, device=self.device) * reg

        # Flatten trajectories and subtract boundary contribution
        trajs_flat = torch.einsum("...ij->...ji", trajs).reshape(*self.add_dim, -1)
        pos_residual = trajs_flat - pos_boundary
        B = torch.einsum('...ki,...k->...i', basis_multi, pos_residual)

        # Solve for parameters
        params = torch.linalg.solve(A, B)
        self.set_params(params)

        return {
            "params": params,
            "init_pos": self.init_pos,
            "init_vel": self.init_vel,
            "end_pos": self.end_pos,
            "end_vel": self.end_vel,
        }


# =============================================================================
# BeastTokenizer Class
# =============================================================================

class BeastTokenizer(torch.nn.Module):
    """
    B-spline based tokenizer for trajectory encoding/decoding.

    Converts continuous robot trajectories to discrete tokens and vice versa
    using B-splines. Supports separate handling for continuous actions (joints)
    and discrete states (e.g., binary gripper).

    Args:
        num_dof: Total degrees of freedom (joints + gripper)
        num_basis: Number of B-spline basis functions
        seq_len: Trajectory sequence length
        vocab_size: Discrete token vocabulary size (default 256)
        degree_p: B-spline degree (default 4 = quartic)
        gripper_zero_order: Use zero-order splines for gripper (piecewise constant)
        gripper_dof: Number of gripper DOFs (only used if gripper_zero_order=True)
        init_cond_order: Initial condition order (0=none, 1=pos, 2=pos+vel)
        end_cond_order: End condition order
        enforce_init_pos: Enforce initial position constraint in decoding
        device: Torch device ('cuda' or 'cpu')

    Example:
        >>> tokenizer = BeastTokenizer(num_dof=7, num_basis=10, seq_len=50)
        >>> tokens = tokenizer.encode_discrete(trajectories)
        >>> reconstructed = tokenizer.decode_discrete(tokens)
    """

    DEFAULT_DT = 0.01  # 100 Hz sampling rate
    attributes: ClassVar[list[str]] = []

    def __init__(self, num_dof: int = 1, num_basis: int = 10, seq_len: int = 50,
                 vocab_size: int = 256, degree_p: int = 4, gripper_zero_order: bool = False,
                 gripper_dof: int = 1, init_cond_order: int = 0, end_cond_order: int = 0,
                 enforce_init_pos: bool = False, device: str = "cuda"):
        super().__init__()

        self.device = device
        self.seq_length = seq_len
        self.vocab_size = vocab_size
        self.num_basis = num_basis
        self.enforce_init_pos = enforce_init_pos
        self.init_cond_order = init_cond_order
        self.end_cond_order = end_cond_order
        self.dt = self.DEFAULT_DT
        self.init_pos = None

        # DOF distribution
        self.gripper_dof = gripper_dof if gripper_zero_order else 0
        self.joint_dof = num_dof - self.gripper_dof
        self.num_dof = self.joint_dof + self.gripper_dof

        # Create B-spline components
        self.bsp = BSpline(
            num_basis=num_basis, degree=degree_p, num_dof=self.joint_dof,
            init_cond_order=init_cond_order, end_cond_order=end_cond_order,
            device=device
        )
        self.gripper_bsp = BSpline(
            num_basis=num_basis, degree=0, num_dof=self.gripper_dof, device=device
        ) if gripper_zero_order else None

        # Time grid (normalized [0, 1])
        self.times = torch.linspace(0, 1.0, seq_len, device=device)
        self._initialize_weight_bounds()
        self.to(self.device)

    def _initialize_weight_bounds(self):
        """Initialize weight bounds for normalization."""
        total_params = self.num_dof * self.num_basis
        self.register_buffer("w_min", -0.02 * torch.ones(total_params))
        self.register_buffer("w_max", 0.02 * torch.ones(total_params))

    def _get_repeated_times(self, batch_size: int) -> torch.Tensor:
        """Repeat time grid for batch processing."""
        return self.times.unsqueeze(0).expand(batch_size, -1).contiguous()

    @autocast_float32
    def _learn_trajectory_params(self, times: torch.Tensor, trajs: torch.Tensor) -> dict:
        """Learn B-spline parameters from trajectories."""
        joint_params = self.bsp.learn_mp_params_from_trajs(times, trajs[..., :self.joint_dof])

        if self.gripper_bsp is not None:
            gripper_params = self.gripper_bsp.learn_mp_params_from_trajs(
                times, trajs[..., -self.gripper_dof:]
            )
            joint_params['params'] = torch.cat(
                [joint_params['params'], gripper_params['params']], dim=-1
            )

        return joint_params

    @autocast_float32
    def _reconstruct_trajectory(self, params: torch.Tensor, times: torch.Tensor) -> torch.Tensor:
        """Reconstruct trajectory from B-spline parameters."""
        joint_params = params[..., :self.joint_dof * self.num_basis]
        self.bsp.update_inputs(times=times, params=joint_params)
        position = self.bsp.get_traj_pos()

        if self.gripper_bsp is not None:
            gripper_params = params[..., -self.gripper_dof * self.num_basis:]
            self.gripper_bsp.update_inputs(times=times, params=gripper_params)
            position = torch.cat([position, self.gripper_bsp.get_traj_pos()], dim=-1)

        return position

    def _apply_initial_position_constraint(self, params: torch.Tensor,
                                           init_pos: torch.Tensor) -> torch.Tensor:
        """Apply initial position constraint to parameters."""
        if not self.init_pos or init_pos is None:
            return params

        # params laid out as (dof, basis) flattened → (basis, dof) for editing
        reshaped = params.view(params.shape[0], self.num_dof, self.num_basis).transpose(1, 2).contiguous()
        reshaped[:, 0, :self.joint_dof] = init_pos[:, :self.joint_dof]
        return reshaped.transpose(1, 2).contiguous().view(params.shape[0], -1)

    @autocast_float32
    def compute_weights(self, demos: torch.Tensor) -> torch.Tensor:
        """
        Compute B-spline weights from demonstration trajectories.

        Args:
            demos: Demonstration trajectories [batch, seq_len, num_dof]

        Returns:
            B-spline weights [batch, num_params]
        """
        times = self._get_repeated_times(demos.shape[0])
        return self.bsp.learn_mp_params_from_trajs(times, demos)['params']

    def update_weights_bounds_per_batch(self, weights: torch.Tensor):
        """
        Update weight bounds based on batch statistics.

        Args:
            weights: Weights to analyze for bounds update
        """
        weights = weights.reshape(-1, self.num_dof * self.num_basis)
        batch_min = weights.min(dim=0)[0]
        batch_max = weights.max(dim=0)[0]

        tolerance = 1e-4
        smaller_mask = batch_min < (self.w_min - tolerance)
        larger_mask = batch_max > (self.w_max + tolerance)

        if torch.any(smaller_mask):
            self.w_min[smaller_mask] = batch_min[smaller_mask]
        if torch.any(larger_mask):
            self.w_max[larger_mask] = batch_max[larger_mask]

    def update_times(self, times: torch.Tensor):
        """Update the time grid."""
        self.times = times

    @torch.no_grad()
    @autocast_float32
    def encode_discrete(self, trajs: torch.Tensor, update_bounds: bool = True) -> torch.Tensor:
        """
        Encode trajectories to discrete tokens.

        Args:
            trajs: Input trajectories [batch, seq_len, num_dof]
            update_bounds: Update weight bounds from this batch

        Returns:
            Discrete tokens [batch, num_basis * num_dof] in range [0, vocab_size-1]
        """
        times = self._get_repeated_times(trajs.shape[0])
        params_dict = self._learn_trajectory_params(times, trajs)

        if update_bounds:
            self.update_weights_bounds_per_batch(params_dict['params'])

        params = torch.clamp(params_dict['params'], min=self.w_min, max=self.w_max)
        tokens = continuous_to_discrete(params, self.w_min, self.w_max, self.vocab_size)
        # (dof, basis) flat → (basis, dof) flat
        return tokens.view(tokens.shape[0], self.num_dof, self.num_basis).transpose(1, 2).contiguous().view(
            tokens.shape[0], -1
        )

    @torch.no_grad()
    @autocast_float32
    def decode_discrete(self, tokens: torch.Tensor, times: torch.Tensor = None,
                        init_pos: torch.Tensor = None) -> torch.Tensor:
        """
        Decode discrete tokens to trajectories.

        Args:
            tokens: Discrete tokens [batch, num_basis * num_dof]
            times: Custom time points (optional)
            init_pos: Initial position constraint (optional)

        Returns:
            Reconstructed trajectories [batch, seq_len, num_dof]
        """
        # (basis, dof) flat → (dof, basis) flat expected by reconstruction
        tokens = tokens.view(tokens.shape[0], self.num_basis, self.num_dof).transpose(1, 2).contiguous().view(
            tokens.shape[0], -1
        )
        params = discrete_to_continuous(tokens, self.w_min, self.w_max, self.vocab_size)

        if times is None:
            times = self._get_repeated_times(params.shape[0])

        params = self._apply_initial_position_constraint(params, init_pos)
        return self._reconstruct_trajectory(params, times)

    @torch.no_grad()
    @autocast_float32
    def encode_continuous(self, trajs: torch.Tensor, update_bounds: bool = True) -> torch.Tensor:
        """
        Encode trajectories to continuous normalized parameters.

        Args:
            trajs: Input trajectories [batch, seq_len, num_dof]
            update_bounds: Update weight bounds from this batch

        Returns:
            Normalized parameters [batch, num_params] in range [-1, 1]
        """
        times = self._get_repeated_times(trajs.shape[0])
        params_dict = self._learn_trajectory_params(times, trajs)

        if update_bounds:
            self.update_weights_bounds_per_batch(params_dict['params'])

        return normalize_tensor(params_dict['params'], self.w_min, self.w_max)

    @torch.no_grad()
    @autocast_float32
    def decode_continuous(self, params: torch.Tensor, times: torch.Tensor = None,
                          init_pos: torch.Tensor = None) -> torch.Tensor:
        """
        Decode continuous normalized parameters to trajectories.

        Args:
            params: Normalized parameters [batch, num_params] in range [-1, 1]
            times: Custom time points (optional)
            init_pos: Initial position constraint (optional)

        Returns:
            Reconstructed trajectories [batch, seq_len, num_dof]
        """
        params = denormalize_tensor(params, self.w_min, self.w_max)

        if times is None:
            times = self._get_repeated_times(params.shape[0])

        params = self._apply_initial_position_constraint(params, init_pos)
        return self._reconstruct_trajectory(params, times)

    @autocast_float32
    def compute_reconstruction_error(self, raw_traj: torch.Tensor) -> torch.Tensor:
        """
        Compute mean squared reconstruction error.

        Args:
            raw_traj: Original trajectory

        Returns:
            MSE between original and reconstructed trajectory
        """
        if len(raw_traj.shape) == 2:
            raw_traj = raw_traj.unsqueeze(-1)

        tokens = self.encode_discrete(raw_traj)
        reconstructed = self.decode_discrete(tokens)
        return torch.mean((raw_traj - reconstructed) ** 2)

    def _plot_trajectory_comparison(self, original: torch.Tensor, reconstructed: torch.Tensor,
                                    title_prefix: str = ""):
        """Plot comparison between original and reconstructed trajectories."""
        try:
            import matplotlib.pyplot as plt
        except ImportError as exc:
            raise ImportError("matplotlib is required for BEAST visualization helpers") from exc

        original = original.detach().cpu().numpy()
        reconstructed = reconstructed.detach().cpu().numpy()
        x_vals = np.linspace(0, 1.0, original.shape[1])

        batch_size, _, dof = original.shape

        for sample_idx in range(batch_size):
            _, axes = plt.subplots(dof, 1, figsize=(8, 2 * dof), sharex=True)
            if dof == 1:
                axes = [axes]

            for i in range(dof):
                axes[i].plot(x_vals, reconstructed[sample_idx, :, i],
                             marker='o', label='Reconstructed', linestyle='-', color='b')
                axes[i].plot(x_vals, original[sample_idx, :, i],
                             marker='*', label='Ground Truth', linestyle='--', color='r')
                axes[i].set_ylabel(f"DOF {i + 1}")
                axes[i].grid(True)
                axes[i].legend(loc="best")

            axes[-1].set_xlabel("Time (s)")
            plt.suptitle(f"{title_prefix}Trajectory Comparison - Sample {sample_idx}")
            plt.tight_layout(rect=[0, 0, 1, 0.96])
            plt.show()

    def visualize_reconstruction_error_discrete(self, raw_traj: torch.Tensor):
        """Visualize discrete encoding reconstruction error."""
        tokens = self.encode_discrete(raw_traj, update_bounds=True)
        reconstructed = self.decode_discrete(tokens)
        self._plot_trajectory_comparison(raw_traj, reconstructed, "Discrete ")

    def visualize_reconstruction_error_continuous(self, raw_traj: torch.Tensor):
        """Visualize continuous encoding reconstruction error."""
        raw_traj = raw_traj.to(torch.float32)
        if len(raw_traj.shape) == 2:
            raw_traj = raw_traj.unsqueeze(0)

        continuous_tokens = self.encode_continuous(raw_traj, update_bounds=True)
        reconstructed = self.decode_continuous(continuous_tokens)
        self._plot_trajectory_comparison(raw_traj, reconstructed, "Continuous ")


# =============================================================================
# StarVLA Action Header Wrapper
# =============================================================================

class _MLPResNetBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.ffn = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim), nn.ReLU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.ffn(x)


class _WeightRegressor(nn.Module):
    """MLP that maps VLM action-token hiddens to flattened BEAST weights."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_blocks: int = 2):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(input_dim)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.blocks = nn.ModuleList([_MLPResNetBlock(hidden_dim) for _ in range(num_blocks)])
        self.layer_norm2 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.fc1(self.layer_norm1(x)))
        for block in self.blocks:
            x = block(x)
        return self.fc2(self.layer_norm2(x))


class BeastActionHeader(nn.Module):
    """StarVLA action header built on BEAST continuous / discrete representations.

    Modes
    -----
    continuous
        ``encode`` → normalized weights ``[B, num_basis * num_dof]`` in ``[-1, 1]``.
        ``decode`` → action chunk ``[B, seq_len, num_dof]``.
        Optional MLP regressor predicts weights from VLM action-token embeddings
        (OFT-style). When enabled, ``predict_action`` returns decoded actions
        ``[B, seq_len, num_dof]``.

    discrete
        ``encode`` → integer tokens ``[B, num_basis * num_dof]`` in ``[0, vocab-1]``.
        ``decode`` → action chunk ``[B, seq_len, num_dof]``.
        No regressor; intended for AR / FAST-style frameworks that consume tokens.
    """

    def __init__(
        self,
        action_dim: int = 7,
        num_basis: int = 10,
        seq_len: int = 16,
        vocab_size: int = 256,
        degree_p: int = 4,
        gripper_zero_order: bool = True,
        gripper_dof: int = 1,
        mode: str = "continuous",
        with_regressor: bool = True,
        action_hidden_dim: int = 2560,
        device: str = "cpu",
    ) -> None:
        super().__init__()
        if mode not in ("continuous", "discrete"):
            raise ValueError(f"mode must be 'continuous' or 'discrete', got {mode!r}")

        self.mode = mode
        self.action_dim = int(action_dim)
        self.num_basis = int(num_basis)
        self.seq_len = int(seq_len)
        self.vocab_size = int(vocab_size)
        self.with_regressor = bool(with_regressor) and mode == "continuous"
        self.num_params = self.num_basis * self.action_dim

        self.tokenizer = BeastTokenizer(
            num_dof=self.action_dim,
            num_basis=self.num_basis,
            seq_len=self.seq_len,
            vocab_size=self.vocab_size,
            degree_p=degree_p,
            gripper_zero_order=gripper_zero_order,
            gripper_dof=gripper_dof,
            device=device,
        )

        self.regressor: Optional[_WeightRegressor] = None
        if self.with_regressor:
            # Pool chunk_len query tokens → one weight vector, then decode to seq_len actions.
            self.regressor = _WeightRegressor(
                input_dim=action_hidden_dim,
                hidden_dim=action_hidden_dim * 2,
                output_dim=self.num_params,
            )

    @property
    def action_horizon(self) -> int:
        return self.seq_len

    def encode(
        self,
        actions: torch.Tensor,
        update_bounds: bool = True,
    ) -> torch.Tensor:
        """Encode action chunks to BEAST targets.

        Args:
            actions: ``[B, seq_len, action_dim]`` (pads/truncates to ``seq_len``).
            update_bounds: Whether to refresh normalization / quantization bounds.

        Returns:
            continuous: ``[B, num_params]`` float in ``[-1, 1]``
            discrete: ``[B, num_params]`` long in ``[0, vocab_size-1]``
        """
        actions = self._ensure_chunk_length(actions)
        if self.mode == "continuous":
            return self.tokenizer.encode_continuous(actions, update_bounds=update_bounds)
        return self.tokenizer.encode_discrete(actions, update_bounds=update_bounds)

    def decode(
        self,
        encoded: torch.Tensor,
        init_pos: torch.Tensor | None = None,
        allow_grad: bool = False,
    ) -> torch.Tensor:
        """Decode BEAST targets back to action chunks ``[B, seq_len, action_dim]``.

        Args:
            allow_grad: If True (continuous only), skip ``@torch.no_grad`` decode so
                L1 on reconstructed actions can backprop into the regressor.
        """
        if self.mode == "continuous":
            if allow_grad:
                params = denormalize_tensor(encoded, self.tokenizer.w_min, self.tokenizer.w_max)
                times = self.tokenizer._get_repeated_times(params.shape[0])
                params = self.tokenizer._apply_initial_position_constraint(params, init_pos)
                return self.tokenizer._reconstruct_trajectory(params, times)
            return self.tokenizer.decode_continuous(encoded, init_pos=init_pos)
        return self.tokenizer.decode_discrete(encoded, init_pos=init_pos)

    def predict_action(self, actions_hidden_states: torch.Tensor) -> torch.Tensor:
        """OFT-style continuous path: hidden → weights → decoded actions.

        Args:
            actions_hidden_states: ``[B, chunk_len, hidden_dim]`` action-token embeddings.

        Returns:
            ``[B, seq_len, action_dim]`` decoded actions. Gradients flow through the
            linear B-spline reconstruction into the regressor.
        """
        if self.regressor is None:
            raise RuntimeError(
                "predict_action requires mode='continuous' and with_regressor=True"
            )
        # Mean-pool query tokens to a single conditioning vector.
        pooled = actions_hidden_states.mean(dim=1)  # [B, H]
        weights = torch.tanh(self.regressor(pooled))  # [B, num_params] in [-1, 1]
        return self.decode(weights, allow_grad=True)

    def predict_weights(self, actions_hidden_states: torch.Tensor) -> torch.Tensor:
        """Predict normalized continuous weights without decoding."""
        if self.regressor is None:
            raise RuntimeError(
                "predict_weights requires mode='continuous' and with_regressor=True"
            )
        pooled = actions_hidden_states.mean(dim=1)
        return torch.tanh(self.regressor(pooled))

    def _ensure_chunk_length(self, actions: torch.Tensor) -> torch.Tensor:
        """Pad / crop the time axis to ``seq_len``."""
        if actions.ndim != 3:
            raise ValueError(f"expected actions [B, T, D], got shape {tuple(actions.shape)}")
        bsz, t, dof = actions.shape
        if dof != self.action_dim:
            raise ValueError(f"action_dim mismatch: got {dof}, expected {self.action_dim}")
        if t == self.seq_len:
            return actions
        if t > self.seq_len:
            return actions[:, -self.seq_len :, :]
        pad = actions[:, -1:, :].expand(bsz, self.seq_len - t, dof)
        return torch.cat([actions, pad], dim=1)


def get_action_model(config=None) -> BeastActionHeader:
    """Factory: build ``BeastActionHeader`` from global framework config.

    Expected keys under ``config.framework.action_model``:
      - action_dim (or num_dof)
      - action_horizon / seq_len / future_action_window_size
      - num_basis, vocab_size, degree_p
      - gripper_zero_order, gripper_dof
      - mode: ``continuous`` | ``discrete``
      - with_regressor: bool (continuous only)
      - action_hidden_dim: VLM hidden size for the optional regressor
    """
    if config is None:
        return BeastActionHeader()

    cfg = config.framework.action_model
    action_dim = int(getattr(cfg, "action_dim", getattr(cfg, "num_dof", 7)))
    seq_len = int(
        getattr(
            cfg,
            "seq_len",
            getattr(cfg, "action_horizon", getattr(cfg, "future_action_window_size", 16)),
        )
    )
    mode = str(getattr(cfg, "mode", "continuous"))
    with_regressor = bool(getattr(cfg, "with_regressor", mode == "continuous"))
    action_hidden_dim = int(getattr(cfg, "action_hidden_dim", 2560))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    return BeastActionHeader(
        action_dim=action_dim,
        num_basis=int(getattr(cfg, "num_basis", 10)),
        seq_len=seq_len,
        vocab_size=int(getattr(cfg, "vocab_size", 256)),
        degree_p=int(getattr(cfg, "degree_p", 4)),
        gripper_zero_order=bool(getattr(cfg, "gripper_zero_order", True)),
        gripper_dof=int(getattr(cfg, "gripper_dof", 1)),
        mode=mode,
        with_regressor=with_regressor,
        action_hidden_dim=action_hidden_dim,
        device=device,
    )


def start_debugpy_once():
    """Start debugpy once (mirrors other ActionHeader helpers)."""
    import debugpy

    if getattr(start_debugpy_once, "_started", False):
        return
    debugpy.listen(("0.0.0.0", 10094))
    print("🔍 Waiting for VSCode attach on 0.0.0.0:10094 ...")
    debugpy.wait_for_client()
    start_debugpy_once._started = True


if __name__ == "__main__":
    import os

    if os.getenv("DEBUGPY_ENABLE", "0") == "1":
        start_debugpy_once()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    header = BeastActionHeader(
        action_dim=7,
        num_basis=8,
        seq_len=10,
        mode="continuous",
        with_regressor=True,
        action_hidden_dim=128,
        device=device,
    ).to(device)

    traj = torch.randn(2, 10, 7, device=device)
    weights = header.encode(traj, update_bounds=True)
    recon = header.decode(weights)
    print("continuous encode/decode", weights.shape, recon.shape, float((traj - recon).pow(2).mean()))

    header_d = BeastActionHeader(
        action_dim=7,
        num_basis=8,
        seq_len=10,
        mode="discrete",
        with_regressor=False,
        device=device,
    ).to(device)
    tokens = header_d.encode(traj, update_bounds=True)
    recon_d = header_d.decode(tokens)
    print("discrete encode/decode", tokens.shape, tokens.dtype, recon_d.shape)

    hidden = torch.randn(2, 10, 128, device=device)
    pred = header.predict_action(hidden)
    print("predict_action", pred.shape)
    processor.push_to_hub("zhouhongyi/beast", use_auth_token=True)
    print("Processor pushed to the hub.")
