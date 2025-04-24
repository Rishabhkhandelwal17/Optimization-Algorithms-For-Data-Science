# fw_unit_tests.py
"""
Unit-tests for Frank–Wolfe (FW) optimiser.

 ▸ Initialisation-time argument validation
 ▸ Basic convergence on a 2-D quadratic over the simplex
 ▸ Duality-gap monotonic decrease
 ▸ Step-size variants  (standard 2/(t+2), constant, callable, “gap”)
 ▸ Weight-decay correctness
 ▸ foreach / fused plumbing smoke-tests
 ▸ AMP grad-scaler plumbing smoke-test
 ▸ Functional helper  (single-tensor path) sanity-checks
"""

import math
import unittest
from typing import Callable, List

import torch
from torch import nn, Tensor

# ────────────────────────────────────────────────────────────────────
from frankwolfe_optimizer import FrankWolfe, frank_wolfe  # noqa: E402
# ────────────────────────────────────────────────────────────────────


# ------------------------------------------------------------------ #
#                         Helper utilities                           #
# ------------------------------------------------------------------ #
def make_quadratic(a: float = 1.0, b: float = 1.0):
    """
    Simple 2-D quadratic f(x) = a x₁² + b x₂²  with constraint x ∈ Δ²
    (the 2-simplex).  Minimum is the vertex whose coefficient is
    smaller (e.g. (1,0) if a < b).
    """
    p = nn.Parameter(torch.tensor([0.5, 0.5], dtype=torch.float32))

    def loss_fn() -> Tensor:
        return a * p[0] ** 2 + b * p[1] ** 2

    return p, loss_fn


def simplex_lmo(grads: List[Tensor]) -> List[Tensor]:
    """Exact LMO for the 2-simplex: return −e_i of the lowest gradient."""
    g = grads[0]
    idx = torch.argmin(g)
    s = torch.zeros_like(g)
    s[idx] = 1.0
    return [s]


# ------------------------------------------------------------------ #
#                         Test-suites                                #
# ------------------------------------------------------------------ #
class TestFWInit(unittest.TestCase):
    def test_bad_weight_decay(self):
        p = nn.Parameter(torch.randn(2))
        with self.assertRaises(ValueError):
            FrankWolfe([p], lmo=simplex_lmo, weight_decay=-0.1)

    def test_non_callable_lmo(self):
        p = nn.Parameter(torch.randn(2))
        with self.assertRaises(ValueError):
            FrankWolfe([p], lmo="not callable")  # type: ignore[arg-type]

    def test_missing_lipschitz_for_gap(self):
        p = nn.Parameter(torch.randn(2))
        with self.assertRaises(ValueError):
            FrankWolfe([p], lmo=simplex_lmo, step_size="gap")


class TestFWConvergence(unittest.TestCase):
    def run_fw(
        self,
        step_size,
        weight_decay: float = 0.0,
        max_it: int = 60,
        tol: float = 1e-4,
    ):
        p, loss_fn = make_quadratic(1.0, 2.0)
        opt = FrankWolfe(
            [p],
            lmo=simplex_lmo,
            step_size=step_size,
            lipschitz=4.0,
            weight_decay=weight_decay,
            tolerance=tol,
        )
        gaps = []
        for _ in range(max_it):
            def closure():
                opt.zero_grad()
                loss = loss_fn()
                loss.backward()
                return loss

            loss = opt.step(closure)
            if loss is None:  # early stop via tolerance
                break
            gaps.append(opt.param_groups[0]["last_duality_gap"])

        return p.detach(), gaps

    def test_standard_rate(self):
        _, gaps = self.run_fw("standard")
        self.assertLess(gaps[-1], 1e-3)
        # monotonic non-increasing
        diff = torch.tensor(gaps[:-1]) - torch.tensor(gaps[1:])
        self.assertTrue(torch.all(diff >= -1e-12))

    def test_constant_step(self):
        _, gaps = self.run_fw(0.1)
        self.assertLess(gaps[-1], 1)

    def test_callable_step(self):
        _, gaps = self.run_fw(lambda t: 0.5 / (t + 2))
        self.assertLess(gaps[-1], 1e-3)

    def test_gap_based_step(self):
        _, gaps = self.run_fw("gap")
        self.assertLess(gaps[-1], 1e-3)

    def test_weight_decay_equivalence(self):
        # FW with wd = λ   versus   FW on augmented loss f+λ/2‖x‖²
        lam = 0.05
        p1, loss1 = make_quadratic()
        p2, _ = make_quadratic()

        opt1 = FrankWolfe([p1], lmo=simplex_lmo, weight_decay=lam)
        def loss_aug():
            return p2[0] ** 2 + p2[1] ** 2 + 0.5 * lam * p2.pow(2).sum()
        opt2 = FrankWolfe([p2], lmo=simplex_lmo)

        for _ in range(15):
            for p, fn, o in [(p1, loss1, opt1), (p2, loss_aug, opt2)]:
                def closure():
                    o.zero_grad()
                    l = fn()
                    l.backward()
                    return l
                o.step(closure)

        torch.testing.assert_close(p1, p2, atol=1e-5, rtol=1e-4)


class TestFWPlumbing(unittest.TestCase):
    def test_foreach_fused_smoke(self):
        p, loss_fn = make_quadratic()
        for foreach, fused in [(False, False), (True, False), (False, True)]:
            opt = FrankWolfe([p], lmo=simplex_lmo, foreach=foreach, fused=fused)
            def closure():
                opt.zero_grad()
                l = loss_fn()
                l.backward()
                return l
            for _ in range(3):
                opt.step(closure)  # should not raise

    def test_amp_scaler_smoke(self):
        p, loss_fn = make_quadratic()
        opt = FrankWolfe([p], lmo=simplex_lmo)
        opt.grad_scale = torch.tensor(1.0)
        opt.found_inf = torch.tensor([0.0])
        def closure():
            opt.zero_grad()
            l = loss_fn()
            l.backward()
            return l
        opt.step(closure)  # should not raise


class TestFWFunctional(unittest.TestCase):
    def test_single_tensor_path(self):
        p = nn.Parameter(torch.tensor([0.6, 0.4]))
        g = torch.tensor([1.0, -0.5])
        frank_wolfe(
            [p],
            [g],
            [],  # no state buffers for FW
            step_size=0.1,
            lipschitz=4.0,
            foreach=False,
            fused=False,
            grad_scale=None,
            found_inf=None,
            lmo=simplex_lmo,
            dual_gap_out=None,
            weight_decay=0.0,
            maximize=False,
        )
        # after a *small* step towards the LMO solution (e2 here)
        self.assertTrue(math.isclose(float(p.sum()), 1.0, rel_tol=1e-6))
        self.assertTrue(torch.all(p >= 0))  # still in simplex


if __name__ == "__main__":
    unittest.main()
