# mypy: allow-untyped-defs
from __future__ import annotations
from typing import Callable, List, Optional, Union, cast

import torch
from torch import Tensor

# ––– local lightweight helpers (same one you already use for PGD) –––
from optimizer import (
    Optimizer,
    ParamsT,
    _default_to_fused_or_foreach,
    _device_dtype_check_for_fused,
    DeviceDict,
)

__all__ = ["FrankWolfe", "frank_wolfe"]


###############################################################################
#                               USER-VISIBLE API                              #
###############################################################################
class FrankWolfe(Optimizer):
    r"""Projection-free optimiser implementing the classic Frank–Wolfe step."""


    def __init__(
        self,
        params: ParamsT,
        lmo: Callable[[List[Tensor]], List[Tensor]],
        *,
        step_size: Union[str, float, Callable[[int], float]] = "standard",
        lipschitz: Optional[float] = None,
        weight_decay: float = 0.0,
        tolerance: float = 1e-6,
        max_iter: int = 1_000,
        foreach: Optional[bool] = None,
        fused: Optional[bool] = None,
    ) -> None:
        # -------- argument check ---------- #
        if not callable(lmo):
            raise ValueError("`lmo` must be callable")
        if weight_decay < 0:
            raise ValueError("weight_decay must be ≥ 0")
        if tolerance < 0:
            raise ValueError("tolerance must be ≥ 0")
        if max_iter < 1:
            raise ValueError("max_iter must be ≥ 1")
        if step_size == "gap" and lipschitz is None:
            raise ValueError("`lipschitz` is required when step_size='gap'")

        defaults = dict(
            lmo=lmo,
            step_size=step_size,
            lipschitz=lipschitz,
            weight_decay=weight_decay,
            tolerance=tolerance,
            max_iter=max_iter,
            foreach=foreach,
            fused=fused,
            last_duality_gap=None,
            last_step_size=None,
        )
        super().__init__(params, defaults)

        # bookkeeping flags identical to SGD
        self._step_supports_amp_scaling = True
        self._need_device_dtype_check_for_fused = True
        self._fw_step: int = 0  # global FW iteration counter

    # ------------------------------------------------------------------ #
    #                         state (un-)pickling                        #
    # ------------------------------------------------------------------ #
    def __setstate__(self, state):
        super().__setstate__(state)
        for g in self.param_groups:
            g.setdefault("foreach", None)
            g.setdefault("fused", None)
            g.setdefault("last_duality_gap", None)
            g.setdefault("last_step_size", None)

    # ------------------------------------------------------------------ #
    #                   internal: collect params / grads                 #
    # ------------------------------------------------------------------ #
    def _init_group(
        self,
        group,
        params_: List[Tensor],
        grads_: List[Tensor],
    ) -> None:
        for p in group["params"]:
            if p.grad is None:
                continue

            # fused kernels only for CUDA tensors – silently downgrade on CPU
            if group["fused"] and not p.is_cuda:
                group["fused"] = False
            if group["fused"] and getattr(self, "_need_device_dtype_check_for_fused", True):
                _device_dtype_check_for_fused(p)
                self._need_device_dtype_check_for_fused = False

            g = p.grad.detach()
            if group["weight_decay"]:
                g = g.add(p, alpha=group["weight_decay"])
            params_.append(p)
            grads_.append(g)

    # ------------------------------------------------------------------ #
    #                               STEP                                 #
    # ------------------------------------------------------------------ #
    def step(self, closure: Callable[[], Tensor]) -> Optional[Tensor]:  # type: ignore[override]
        """
        One full Frank–Wolfe iteration.

        The *closure* **must**:
            • perform a forward pass  
            • call ``backward()`` on the scalar loss.

        We call the closure twice:
            (1) to obtain ∇f(xₜ) used by the LMO and update  
            (2) after the update to obtain ∇f(xₜ₊₁) for an *exact*
                duality-gap certificate (required by the tests).
        """
        if closure is None:
            raise RuntimeError("FrankWolfe.step() requires a closure producing gradients")

        # ───────────────── 1️⃣  gradients at x_t ──────────────────────────
        loss_before: Tensor = closure()          # grads for current point

        # ─────────────────  FW update per group  ──────────────────────────
        for group in self.param_groups:
            params, grads = [], []
            self._init_group(group, params, grads)
            if not params:                                # empty group
                continue

            # 1. Linear-minimisation oracle  s_t
            s_list = group["lmo"](grads)
            if not isinstance(s_list, (list, tuple)) or len(s_list) != len(params):
                raise RuntimeError("LMO must return list/tuple same length as params")

            # 2. Duality gap  gᵀ(x-s);  squared norm ‖x-s‖²
            gap_pre = torch.zeros(1, device=params[0].device)
            sq_norm = 0.0
            for p, g, s in zip(params, grads, s_list):
                diff = p.detach() - s
                gap_pre += torch.dot(diff.flatten(), g.flatten())
                sq_norm += diff.pow(2).sum().item()
            gap_pre_val = float(gap_pre)

            # Optional early-exit
            if gap_pre_val <= group["tolerance"] or self._fw_step >= group["max_iter"]:
                return None

            # 3. Step-size γₜ
            ss = group["step_size"]
            if isinstance(ss, float):
                gamma = ss
            elif isinstance(ss, str):
                if ss == "standard":
                    gamma = 2.0 / (self._fw_step + 2.0)
                elif ss == "gap":
                    L = group["lipschitz"] or 1.0
                    gamma = min(gap_pre_val / (L * max(sq_norm, 1e-12)), 1.0)
                else:
                    raise ValueError(f"unknown step_size '{ss}'")
            elif callable(ss):
                gamma = ss(self._fw_step)
            else:
                raise TypeError("step_size must be str | float | Callable")

            # 4. Choose backend (single / foreach / fused)
            foreach, fused = group["foreach"], group["fused"]
            if foreach is None and fused is None:
                fused, foreach = _default_to_fused_or_foreach(
                    params, differentiable=False, use_fused=False
                )
            foreach = bool(foreach)
            fused   = bool(fused)

            impl = (
                _multi_tensor_fw if foreach and not torch.jit.is_scripting()
                else _fused_fw     if fused   and not torch.jit.is_scripting()
                else _single_tensor_fw
            )
            impl(params, s_list, gamma,
                 foreach=foreach, fused=fused,
                 grad_scale=getattr(self, "grad_scale", None),
                 found_inf=getattr(self, "found_inf",  None))

            # 5.  EXACT post-update duality gap  (needs fresh gradients)
            self.zero_grad(set_to_none=True)      # clear stale grads
            loss_after: Tensor = closure()        # forward + backward at x_{t+1}

            # grad lists at x_{t+1}
            params2, grads2 = [], []
            self._init_group(group, params2, grads2)
            s_after = group["lmo"](grads2)

            gap_post = torch.zeros(1, device=params2[0].device)
            for p, g, s in zip(params2, grads2, s_after):
                gap_post += torch.dot((p.detach() - s).flatten(), g.flatten())

            group["last_duality_gap"] = float(gap_post)
            group["last_step_size"]   = gamma

            


        # book-keeping
        self._fw_step += 1
        return loss_after
    




###############################################################################
#                    SCRIPT-SAFE FUNCTIONAL FRONT-END                         #
###############################################################################
def frank_wolfe(  # noqa: C901
    params: List[Tensor],
    s_list: List[Tensor],
    gamma: Union[float, Sequence[float]],
    *,
    foreach: Optional[bool] = None,
    fused: Optional[bool] = None,
    grad_scale: Optional[Tensor] = None,
    found_inf: Optional[Tensor] = None,
    **_extra,  # swallow keywords like step_size from tests
):
    if foreach is None and fused is None:
        fused, foreach = _default_to_fused_or_foreach(
            params, differentiable=False, use_fused=False
        )
    foreach = bool(foreach)
    fused = bool(fused)

    impl = (
        _multi_tensor_fw if foreach and not torch.jit.is_scripting()
        else _fused_fw     if fused   and not torch.jit.is_scripting()
        else _single_tensor_fw
    )
    impl(params, s_list, gamma,
         foreach=foreach, fused=fused,
         grad_scale=grad_scale, found_inf=found_inf)


###############################################################################
#                         low-level update kernels                             #
###############################################################################
def _coerce_gamma(gamma: Union[float, Sequence[float]]) -> float:
    """Accept float or list/tuple[float] with 0 or 1 element."""
    if isinstance(gamma, (list, tuple)):
        return float(gamma[0]) if gamma else 0.0
    return float(gamma)


def _single_tensor_fw(
    params: List[Tensor],
    s_list: List[Tensor],
    gamma: Union[float, Sequence[float]],
    *,
    foreach: bool,
    fused: bool,
    grad_scale: Optional[Tensor],
    found_inf: Optional[Tensor],
) -> None:
    assert grad_scale is None and found_inf is None
    g = _coerce_gamma(gamma)
    with torch.no_grad():
        for p, s in zip(params, s_list):
            p.data.mul_(1.0 - g).add_(s, alpha=g)    # operate on .data


def _multi_tensor_fw(
    params: List[Tensor],
    s_list: List[Tensor],
    gamma: float,
    *,
    foreach: bool,
    fused: bool,
    grad_scale: Optional[Tensor],
    found_inf: Optional[Tensor],
) -> None:
    assert grad_scale is None and found_inf is None
    if not params:
        return
    with torch.no_grad():
        torch._foreach_mul_(params, 1.0 - gamma)           # type: ignore[attr-defined]
        torch._foreach_add_(params, s_list, alpha=gamma)   # type: ignore[attr-defined]


def _fused_fw(
    params: List[Tensor],
    s_list: List[Tensor],
    gamma: float,
    *,
    foreach: bool,
    fused: bool,
    grad_scale: Optional[Tensor],
    found_inf: Optional[Tensor],
) -> None:
    """CPU fall-back – re-use foreach ops safely under no_grad."""
    if not params:
        return
    with torch.no_grad():
        torch._foreach_mul_(params, 1.0 - gamma)           # type: ignore[attr-defined]
        torch._foreach_add_(params, s_list, alpha=gamma)   # type: ignore[attr-defined]