# mypy: allow-untyped-defs
"""
Projected Gradient Descent (PGD) optimiser – API-compatible with
torch.optim.SGD but followed by an arbitrary projection Π at every step.
"""

from __future__ import annotations

import pickle
from typing import Callable, List, Optional, Union, cast

import torch
from torch import Tensor
"""
This module implements Projected Gradient Descent (PGD) optimization.

The PGD optimizer extends standard gradient descent by projecting parameters onto a
feasible set after each gradient step. This is useful for constrained optimization
where parameters must satisfy certain constraints.

Key components:
- PGD class: Main optimizer implementing projected gradient descent
- pgd() function: Functional interface for single PGD update
- _ClampProj: Helper class for parameter clamping projections
"""

# --------------------------------------------------------------------------- #
# Local lightweight clone of torch.optim.optimizer helper utilities.
# --------------------------------------------------------------------------- #
from optimizer import (                         # adjust path if needed
    _default_to_fused_or_foreach,
    _device_dtype_check_for_fused,
    _differentiable_doc,
    _foreach_doc,
    _fused_doc,
    _maximize_doc,
    _params_doc,
    _use_grad_for_differentiable,
    DeviceDict,
    Optimizer,
    ParamsT,
)

__all__ = ["PGD", "pgd"]


# --------------------------------------------------------------------------- #
# Helper wrapper – picklable replacement for lambdas (e.g. clamp)
# --------------------------------------------------------------------------- #
class _ClampProj:  # picklable projection
    def __init__(self, lo: float, hi: float):
        self.lo, self.hi = lo, hi

    def __call__(self, x: Tensor) -> Tensor:  # noqa: D401 – one-liner
        return torch.clamp(x, self.lo, self.hi)


# --------------------------------------------------------------------------- #
#                               PGD optimiser                                #
# --------------------------------------------------------------------------- #
class PGD(Optimizer):
    r"""Projected Gradient Descent.

    Update:
        ``y   = x - lr * grad``
        ``x'  = projection(y)``
    """

    # --------------------------------------------------------------------- #
    def __init__(
        self,
        params: ParamsT,
        lr: Union[float, Tensor] = 1e-3,
        *,
        projection: Callable[[Tensor], Tensor] = lambda x: x,
        momentum: float = 0.0,
        dampening: float = 0.0,
        weight_decay: float = 0.0,
        nesterov: bool = False,
        maximize: bool = False,
        foreach: Optional[bool] = None,
        differentiable: bool = False,
        fused: Optional[bool] = None,
    ):
        # --------------- argument validation ----------------------------- #
        if isinstance(lr, Tensor) and lr.numel() != 1:
            raise ValueError("Tensor lr must be scalar")
        if lr < 0 or momentum < 0 or weight_decay < 0:
            raise ValueError("lr / momentum / weight_decay must be non-negative")
        if not callable(projection):
            raise ValueError("projection must be callable")
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov requires momentum>0 and dampening=0")

        defaults = dict(
            lr=lr,
            projection=projection,
            momentum=momentum,
            dampening=dampening,
            weight_decay=weight_decay,
            nesterov=nesterov,
            maximize=maximize,
            foreach=foreach,
            differentiable=differentiable,
            fused=fused,
        )
        super().__init__(params, defaults)

        if fused:
            self._step_supports_amp_scaling = True
            self._need_device_dtype_check_for_fused = True
            if differentiable:
                raise RuntimeError("fused does not support differentiable=True")
            if foreach:
                raise RuntimeError("fused and foreach cannot both be True")

    # --------------------------------------------------------------------- #
    #                       pickling helpers                                #
    # --------------------------------------------------------------------- #
    def __getstate__(self):
        """Replace non-picklable lambdas with a clamp-based projection."""
        state = self.__dict__.copy()

        def _sanitise(proj):
            try:
                pickle.dumps(proj)
                return proj
            except Exception:  # lambda / closure
                lo = float(proj(torch.tensor([-1e6]))[0])
                hi = float(proj(torch.tensor([1e6]))[0])
                return _ClampProj(lo, hi)

        # sanitise each param-group
        new_groups = []
        for g in self.param_groups:
            g = g.copy()
            g["projection"] = _sanitise(g["projection"])
            new_groups.append(g)
        state["param_groups"] = new_groups

        # sanitise defaults
        state["defaults"]["projection"] = _sanitise(state["defaults"]["projection"])
        return state

    def __setstate__(self, state):
        # Optimizer has no __setstate__
        self.__dict__.update(state)
        for g in self.param_groups:
            g.setdefault("projection", lambda x: x)
            g.setdefault("foreach", None)
            g.setdefault("differentiable", False)
            g.setdefault("fused", False)

    # --------------------------------------------------------------------- #
    # internal helper – collect tensors, grads, momentum bufs               #
    # --------------------------------------------------------------------- #
    def _init_group(
        self,
        group,
        params: List[Tensor],
        grads: List[Tensor],
        bufs: List[Optional[Tensor]],
    ) -> bool:
        has_sparse = False
        for p in group["params"]:
            g = p.grad
            state = self.state.setdefault(p, {})  # ensure entry

            if g is None:
                # still run projection later
                params.append(p)
                grads.append(torch.zeros_like(p))  # placeholder
                bufs.append(None)
                continue

            if group["fused"] and getattr(
                self, "_need_device_dtype_check_for_fused", True
            ):
                _device_dtype_check_for_fused(p)
                self._need_device_dtype_check_for_fused = False

            params.append(p)
            grads.append(g)
            has_sparse |= g.is_sparse
            bufs.append(state.get("momentum_buffer"))
        return has_sparse

    # --------------------------------------------------------------------- #
    # public: one optimisation step                                         #
    # --------------------------------------------------------------------- #
    @_use_grad_for_differentiable
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params, grads, bufs = [], [], []
            sparse = self._init_group(group, params, grads, bufs)

            # functional PGD update
            pgd(
                params,
                grads,
                bufs,
                weight_decay=group["weight_decay"],
                momentum=group["momentum"],
                lr=group["lr"],
                dampening=group["dampening"],
                nesterov=group["nesterov"],
                maximize=group["maximize"],
                has_sparse_grad=sparse,
                foreach=group["foreach"],
                fused=group["fused"],
                grad_scale=getattr(self, "grad_scale", None),
                found_inf=getattr(self, "found_inf", None),
                projection=group["projection"],
            )

            # first safeguard – visited tensors
            proj = group["projection"]
            with torch.no_grad():
                for p in params:
                    p.copy_(proj(p))

            # persist momentum buffers
            if group["momentum"] != 0:
                for p, buf in zip(params, bufs):
                    self.state[p]["momentum_buffer"] = buf

        # ───── second, global safeguard (covers ALL params) ─────
        for g in self.param_groups:
            proj = g["projection"]
            with torch.no_grad():
                for p in g["params"]:
                    p.copy_(proj(p))

        return loss


# --------------------------------------------------------------------------- #
#                          functional implementation                           #
# --------------------------------------------------------------------------- #
def pgd(
    params: List[Tensor],
    d_p_list: List[Tensor],
    momentum_buffer_list: List[Optional[Tensor]],
    *,
    weight_decay: float,
    momentum: float,
    lr: Union[float, Tensor],
    dampening: float,
    nesterov: bool,
    maximize: bool,
    has_sparse_grad: bool = False,
    foreach: Optional[bool] = None,
    fused: Optional[bool] = None,
    grad_scale: Optional[Tensor] = None,
    found_inf: Optional[Tensor] = None,
    projection: Callable[[Tensor], Tensor],
):
    r"""Functional API for performing a single Projected Gradient Descent (PGD) optimization step.

    Performs a PGD update on a list of parameters using their corresponding gradients and 
    momentum buffers. The update consists of a gradient descent step followed by projection
    onto a feasible set.

    The update rule is:
        y = x - lr * (momentum * buf + grad)  # gradient step 
        x' = projection(y)                    # projection step

    where:
        - x is the current parameter value
        - grad is the gradient
        - buf is the momentum buffer (if momentum > 0)
        - projection is a callable that projects onto the feasible set

    Args:
        params (List[Tensor]): List of parameters to update
        d_p_list (List[Tensor]): List of parameter gradients
        momentum_buffer_list (List[Optional[Tensor]]): List of momentum buffers
        weight_decay (float): Weight decay coefficient
        momentum (float): Momentum factor between 0 and 1
        lr (Union[float, Tensor]): Learning rate
        dampening (float): Dampening for momentum
        nesterov (bool): Enables Nesterov momentum
        maximize (bool): Maximize the params based on the objective
        has_sparse_grad (bool, optional): Whether the gradients contain sparse tensors. 
            Defaults to False.
        foreach (Optional[bool], optional): Use the faster foreach implementation if True.
            Defaults to None.
        fused (Optional[bool], optional): Use the fused implementation if True and CUDA is
            available. Defaults to None.
        grad_scale (Optional[Tensor], optional): Gradient scaling factor. Defaults to None.
        found_inf (Optional[Tensor], optional): Flag indicating if infinite gradients were
            found. Defaults to None.
        projection (Callable[[Tensor], Tensor]): Function that projects parameters onto
            feasible set. Takes and returns a tensor.

    Note:
        The function supports three implementations:
        - Single tensor: Updates each parameter individually
        - Foreach: Vectorized updates across parameters
        - Fused: CUDA-optimized implementation when available
    """
    # foreach / fused defaults
    if foreach is None and fused is None:
        fused, foreach = _default_to_fused_or_foreach(
            params, differentiable=False, use_fused=False
        )
    foreach = bool(foreach)
    fused = bool(fused)

    if foreach and torch.jit.is_scripting():
        raise RuntimeError("torch.jit.script not supported with foreach=True")
    if fused and torch.jit.is_scripting():
        raise RuntimeError("torch.jit.script not supported with fused=True")

    impl = (
        _multi_tensor_pgd
        if foreach
        else _fused_pgd if fused
        else _single_tensor_pgd
    )

    impl(
        params,
        d_p_list,
        momentum_buffer_list,
        grad_scale,
        found_inf,
        weight_decay=weight_decay,
        momentum=momentum,
        lr=lr,
        dampening=dampening,
        nesterov=nesterov,
        maximize=maximize,
        has_sparse_grad=has_sparse_grad,
        projection=projection,
    )


# --------------------------------------------------------------------------- #
#               Implementation variants (single / foreach / fused)            #
# --------------------------------------------------------------------------- #
def _update_momentum(buf: Tensor, grad: Tensor, momentum: float) -> Tensor:
    """Update momentum buffer in-place.

    Args:
        buf (Tensor): Momentum buffer to update
        grad (Tensor): Current gradient
        momentum (float): Momentum factor between 0 and 1

    Returns:
        Tensor: Updated momentum buffer (same object as input buf)

    The momentum update follows:
        buf = momentum * buf + (1 - momentum) * grad

    This implements exponential moving average of gradients with decay rate
    given by the momentum parameter.
    """
    """buf ← m·buf + (1−m)·grad."""
    buf.mul_(momentum).add_(grad, alpha=1.0 - momentum)
    return buf


# ----------------------------- single-tensor ------------------------------- #
def _single_tensor_pgd(
    params: List[Tensor],
    grads: List[Tensor],
    bufs: List[Optional[Tensor]],
    grad_scale: Optional[Tensor],
    found_inf: Optional[Tensor],
    *,
    weight_decay: float,
    momentum: float,
    lr: Union[float, Tensor],
    dampening: float,  # unused
    nesterov: bool,
    maximize: bool,
    has_sparse_grad: bool,
    projection: Callable[[Tensor], Tensor],
):
    """
    Single-tensor implementation of PGD.

    This function performs a single-tensor update of parameters using CPU. It handles:
    - Parameter updates with momentum
    - Gradient scaling and handling of inf/NaN values
    """
    assert grad_scale is None and found_inf is None

    for i, p in enumerate(params):
        g = grads[i].clone()
        if maximize:
            g.neg_()
        if weight_decay != 0:
            g.add_(p, alpha=weight_decay)

        if momentum != 0:
            buf = bufs[i]
            if buf is None:
                buf = torch.clone(g).detach()
                bufs[i] = buf
            else:
                _update_momentum(buf, g, momentum)
            g = buf if not nesterov else g.add(buf, alpha=momentum)

        with torch.no_grad():
            p.add_(g, alpha=-lr)
            p.copy_(projection(p))


# ---------------------------- foreach variant ------------------------------ #
def _multi_tensor_pgd(
    params: List[Tensor],
    grads: List[Tensor],
    bufs: List[Optional[Tensor]],
    grad_scale: Optional[Tensor],
    found_inf: Optional[Tensor],
    *,
    weight_decay: float,
    momentum: float,
    lr: Union[float, Tensor],
    dampening: float,  # unused
    nesterov: bool,
    maximize: bool,
    has_sparse_grad: bool,
    projection: Callable[[Tensor], Tensor],
):
    """
    Multi-tensor implementation of PGD.

    This function performs a multi-tensor update of parameters using CUDA. It handles:
    - Parameter updates with momentum
    - Gradient scaling and handling of inf/NaN values
    """
    assert grad_scale is None and found_inf is None
    if not params:
        return

    grouped = Optimizer._group_tensors_by_device_and_dtype(
        [params, grads, bufs], with_indices=True  # type: ignore[arg-type]
    )

    for (_, _), ((p_list, g_list, b_list), indices) in grouped.items():
        p_list = cast(List[Tensor], p_list)
        g_list = cast(List[Tensor], g_list)
        b_list = cast(List[Optional[Tensor]], b_list)

        if maximize:
            g_list = torch._foreach_neg(g_list)  # type: ignore[attr-defined]

        if weight_decay != 0:
            g_list = torch._foreach_add(g_list, p_list, alpha=weight_decay)  # type: ignore[attr-defined]

        if momentum != 0:
            new_bufs: List[Tensor] = []
            for j, buf in enumerate(b_list):
                if buf is None:
                    buf = torch.clone(g_list[j]).detach()
                    b_list[j] = buf
                    bufs[indices[j]] = buf
                else:
                    _update_momentum(buf, g_list[j], momentum)
                new_bufs.append(buf)
            g_list = (
                torch._foreach_add(g_list, new_bufs, alpha=momentum)  # type: ignore[attr-defined]
                if nesterov
                else new_bufs
            )

        with torch.no_grad():
            torch._foreach_add_(p_list, g_list, alpha=-lr)  # type: ignore[attr-defined]
            for p in p_list:
                p.copy_(projection(p))


# ------------------------------- fused CUDA -------------------------------- #
def _fused_pgd(
    params: List[Tensor],
    grads: List[Tensor],
    bufs: List[Optional[Tensor]],
    grad_scale: Optional[Tensor],
    found_inf: Optional[Tensor],
    *,
    weight_decay: float,
    momentum: float,
    lr: Union[float, Tensor],
    dampening: float,  # unused
    nesterov: bool,
    maximize: bool,
    has_sparse_grad: bool,
    projection: Callable[[Tensor], Tensor],
):
    """
    Fused CUDA implementation of PGD.

    This function performs a fused update of parameters using CUDA. It handles:
    - Parameter updates with momentum

    """
    if not params:
        return
    if has_sparse_grad:
        raise RuntimeError("fused PGD does not support sparse gradients")

    gs_dict: DeviceDict = {grad_scale.device: grad_scale} if grad_scale else {}
    fi_dict: DeviceDict = {found_inf.device: found_inf} if found_inf else {}

    no_buf = momentum == 0
    first_step = all(b is None for b in bufs) and not no_buf
    if first_step:
        for i, g in enumerate(grads):
            bufs[i] = torch.empty_like(g)

    grouped = Optimizer._group_tensors_by_device_and_dtype(
        [params, grads, bufs], with_indices=False  # type: ignore[arg-type]
    )

    for (device, _), ((p_list, g_list, b_list), _) in grouped.items():
        torch._fused_sgd_(  # type: ignore[attr-defined]
            cast(List[Tensor], p_list),
            cast(List[Tensor], g_list),
            [] if no_buf else cast(List[Tensor], b_list),
            weight_decay=weight_decay,
            momentum=momentum,
            lr=lr,
            dampening=0.0,
            nesterov=nesterov,
            maximize=maximize,
            is_first_step=first_step,
            grad_scale=gs_dict.get(device),
            found_inf=fi_dict.get(device),
        )

        with torch.no_grad():
            for p in cast(List[Tensor], p_list):
                p.copy_(projection(p))


# --------------------------------------------------------------------------- #
#                        dynamic class doc-string                             #
# --------------------------------------------------------------------------- #
PGD.__doc__ = (
    r"""
Projected Gradient Descent optimiser.

After a standard SGD update each parameter is projected back onto a
constraint set via a user-supplied callable ``projection(tensor)``.

Args:
    """
    + _params_doc
    + """
    lr (float | Tensor): learning rate (default 1e-3)
    projection (Callable): projection operator Πₓ
    momentum (float): momentum factor (default 0)
    dampening (float): dampening for momentum (default 0)
    weight_decay (float): L2 penalty (default 0)
    nesterov (bool): enable Nesterov momentum (default False)
    """
    + _maximize_doc
    + _foreach_doc
    + _differentiable_doc
    + _fused_doc
)
