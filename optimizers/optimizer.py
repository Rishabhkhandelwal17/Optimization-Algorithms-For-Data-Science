# optim/optimizer.py
"""
Light-weight stand-in for ``torch.optim.optimizer``

Only the symbols required by *pgd_optimizer.py* and *frank_wolfe.py* are
implemented.  
"""

from __future__ import annotations

import itertools
import numbers
import warnings
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Tuple, Union

import torch
from torch import Tensor

__all__ = [
    # public Optimizer API
    "Optimizer",
    "ParamsT",
    # tiny helper aliases mirrored from torch
    "_default_to_fused_or_foreach",
    "_device_dtype_check_for_fused",
    "_use_grad_for_differentiable",
    # doc-strings that PGD / FW re-use
    "_differentiable_doc",
    "_foreach_doc",
    "_fused_doc",
    "_maximize_doc",
    "_params_doc",
    # typing niceties
    "DeviceDict",
]

###############################################################################
# Typing helpers
###############################################################################
ParamsT = Union[Iterable[Tensor], Iterable[dict[str, Any]]]
DeviceDict = Dict[torch.device, Any]  # used by fused / grad-scaler plumbing


###############################################################################
# Documentation snippets (lifted from PyTorch for consistency)
###############################################################################
_params_doc = """params (iterable): iterable of parameters to optimize or dicts defining
    parameter groups"""
_maximize_doc = """maximize (bool, optional): maximize the params based on the objective,
    instead of minimizing (default: False)"""
_foreach_doc = """foreach (bool, optional): whether foreach implementation of optimizer is used
    (default: None)"""
_fused_doc = """fused (bool, optional): whether fused implementation of optimizer is used
    (default: None)"""
_differentiable_doc = (
    "differentiable (bool, optional): "
    "whether autograd should record operations on the parameters in this optimizer. "
    "Setting to ``True`` allows higher-order optimization but slows down the code."
)

###############################################################################
# Tiny utility helpers
###############################################################################
def _default_to_fused_or_foreach(params: List[Tensor],
                                 *,
                                 differentiable: bool,
                                 use_fused: bool) -> Tuple[bool, bool]:
    """
    Mirror PyTorch’s heuristic:

    *If* we’re on a CUDA device, tensors are floats, and user didn't override
    flags, then enable ``foreach`` by default (but not ``fused`` unless
    ``use_fused=True``).

    Returns (fused, foreach)
    """
    if not params:
        return False, False

    first = params[0]
    is_cuda = first.is_cuda
    is_fp = first.dtype in (torch.float16, torch.float32, torch.bfloat16)
    if differentiable or not is_cuda or not is_fp:
        return False, False
    return (use_fused and torch.cuda.is_available()), (not use_fused)


def _device_dtype_check_for_fused(p: Tensor) -> None:
    """Quick sanity check—real PyTorch does much more."""
    if not p.is_cuda:
        raise RuntimeError("Fused optimizers require CUDA tensors")
    if p.dtype not in (torch.float16, torch.float32, torch.bfloat16):
        raise RuntimeError("Fused optimizers support only floating dtypes")


def _use_grad_for_differentiable(func):
    """
    Decorator used in PyTorch: forwards ``grad`` argument when
    ``differentiable=True``.  Our simplified version just calls through.
    """
    return func


###############################################################################
# Core Optimizer class (trimmed PyTorch clone)
###############################################################################
class Optimizer:
    """Base class for all optimizers.

    This serves as the foundation for all PyTorch-style optimizers. It provides the core
    functionality for parameter management, state tracking, and the optimization step interface.
    Specific optimizers (like SGD, Adam, etc.) inherit from this class and implement their
    own optimization algorithms.

    The class handles:
    - Parameter group management (multiple parameter groups with different settings)
    - Optimization state tracking (momentum buffers, running averages, etc.)
    - Basic parameter validation and type checking
    - Common optimizer utilities (zero_grad, step, state_dict, etc.)

    All optimizer implementations should inherit from this class and override the `step()`
    method to implement their specific optimization algorithm.
    

    Args:
        params:         iterable of parameters or parameter-group dicts
        defaults:       dict of default hyper-parameters (lr, weight_decay, …)
    """

    def __init__(self, params: ParamsT, defaults: Mapping[str, Any]) -> None:
        """Base class for all optimizers.

        .. warning::
            Parameters need to be specified as collections that have a deterministic
            ordering that is consistent between runs. Examples of objects that don't
            satisfy this requirement are sets and iterators over dictionaries.

        Args:
            params (iterable): an iterable of :class:`torch.Tensor` s or
                :class:`dict` s. Specifies what Tensors should be optimized.
            defaults: (dict): a dict containing default values of optimization
                options (used when a parameter group doesn't specify them).
        """
        self.defaults: Dict[str, Any] = dict(defaults)

        # list[dict] where each dict holds a 'params' list plus hyper-params
        self.param_groups: List[Dict[str, Any]] = []
        self.state: Dict[Tensor, Dict[str, Any]] = {}

        if isinstance(params, (list, tuple)):
            if len(params) == 0:
                raise ValueError("optimizer got an empty parameter list")
            if isinstance(params[0], Mapping):
                for group in params:
                    self.add_param_group(group)
            else:
                self.add_param_group({"params": params})
        else:
            raise TypeError("params argument given to the optimizer should be "
                            "an iterable of Tensors or dicts")

    # --------------------------------------------------------------------- #
    # Basic public API                                                      #
    # --------------------------------------------------------------------- #
    def __repr__(self) -> str:  # pragma: no cover
        """Return a string representation of the optimizer.

        The representation includes the optimizer class name and the number of parameter
        groups, showing how many parameters are in each group.

        Returns:
            str: String representation like "OptimizerName(N params, M params, ...)"
        """
        group_str = ", ".join(f"{len(g['params'])} params" for g in self.param_groups)
        return f"{self.__class__.__name__}({group_str})"

    def state_dict(self) -> Dict[str, Any]:
        """Return the optimizer's state dict.

        The state dict contains two entries:
            * state - a dict holding current optimization state. Its content
                differs between optimizer classes.
            * param_groups - a list containing all parameter groups where each
                parameter group is a dict containing the parameters and options

        Returns:
            dict: The state of the optimizer as a dict.
        """
        return {"state": self.state, "param_groups": self.param_groups}

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        """Load an optimizer state dict.

        Arguments:
            state_dict (dict): Optimizer state. Should be an object returned
                from a call to :meth:`state_dict`.

        The state dict contains two entries:
            * state - a dict holding current optimization state. Its content
                differs between optimizer classes.
            * param_groups - a list containing all parameter groups where each
                parameter group is a dict containing the parameters and options
        """
        self.state.update(state_dict["state"])
        self.param_groups[:] = state_dict["param_groups"]

    def zero_grad(self, set_to_none: bool = True) -> None:
        """Zero out the gradients of all parameters managed by the optimizer.

        Args:
            set_to_none (bool, optional): Instead of setting gradients to zero, set them
                to None. This can provide performance benefits, but requires changing model
                code to handle None gradients. Defaults to True.
        """
        for p in itertools.chain.from_iterable(g["params"] for g in self.param_groups):
            if p.grad is not None:
                if set_to_none:
                    p.grad = None
                else:
                    p.grad.detach_()
                    p.grad.zero_()

    # ------------------------------------------------------------------ #
    # Internal helpers                                                   #
    # ------------------------------------------------------------------ #
    def add_param_group(self, group: Dict[str, Any]) -> None:
        """Add a parameter group to the optimizer.

        This can be useful when fine tuning a pre-trained network as frozen layers can be made
        trainable and added to the optimizer as parameters groups.

        Arguments:
            group (dict): Specifies what Tensors should be optimized along with group
                specific optimization options.
        """
        if "params" not in group or not isinstance(group["params"], (list, tuple)):
            raise TypeError("param group must have a 'params' key")

        params = group["params"]
        if isinstance(params[0], (list, tuple)):
            params = list(params[0])

        for p in params:
            if not isinstance(p, torch.Tensor):
                raise TypeError("optimizer parameters must be Tensors")
            if not p.requires_grad:
                raise ValueError("optimize a parameter that doesn't require grad")

        # fill in defaults
        for k, v in self.defaults.items():
            group.setdefault(k, v)

        group["params"] = list(params)
        self.param_groups.append(group)

    # ------------------------------------------------------------------ #
    # Step must be implemented by subclasses                             #
    # ------------------------------------------------------------------ #
    def step(self, closure=None):
        """Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss. Optional for most optimizers.

        Returns:
            Optional[Tensor]: The loss value evaluated by the closure, if provided.

        Note:
            This method must be implemented by subclasses.
        """
        raise NotImplementedError


###############################################################################
# End of file
###############################################################################
