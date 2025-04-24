import unittest
import pickle

import torch
from torch import nn, Tensor

from pgd_optimizer import PGD, pgd


class TestPGDInitialization(unittest.TestCase):
    def test_invalid_lr(self):
        model = nn.Linear(2,1)
        with self.assertRaises(ValueError):
            PGD(model.parameters(), lr=-0.1)

    def test_invalid_momentum(self):
        model = nn.Linear(2,1)
        with self.assertRaises(ValueError):
            PGD(model.parameters(), momentum=-0.5)

    def test_invalid_weight_decay(self):
        model = nn.Linear(2,1)
        with self.assertRaises(ValueError):
            PGD(model.parameters(), weight_decay=-1.0)

    def test_non_callable_projection(self):
        model = nn.Linear(2,1)
        with self.assertRaises(ValueError):
            PGD(model.parameters(), projection="not a function")

    def test_bad_nesterov(self):
        model = nn.Linear(2,1)
        # momentum=0 but nesterov=True
        with self.assertRaises(ValueError):
            PGD(model.parameters(), momentum=0.0, nesterov=True)
        # dampening != 0 but nesterov=True
        with self.assertRaises(ValueError):
            PGD(model.parameters(), momentum=0.9, dampening=0.1, nesterov=True)


class TestPGDBasicFunctionality(unittest.TestCase):
    def setUp(self):
        # simple model with one parameter for clarity
        self.param = nn.Parameter(torch.tensor([0.5, -0.5], dtype=torch.float32))
        self.opt_sgd = torch.optim.SGD([self.param], lr=0.1)
        self.opt_pgd = PGD([self.param], lr=0.1, projection=lambda x: x)  # identity projection

    def test_identity_projection_matches_sgd(self):
        # set a fixed gradient
        self.param.grad = torch.tensor([1.0, -2.0])
        # one step of SGD
        self.opt_sgd.step()
        sgd_result = self.param.data.clone()

        # reset param
        self.param.data = torch.tensor([0.5, -0.5])
        self.param.grad = torch.tensor([1.0, -2.0])
        # one step of PGD (identity)
        self.opt_pgd.step()
        pgd_result = self.param.data

        self.assertTrue(torch.allclose(sgd_result, pgd_result))

    def test_custom_projection_clamp(self):
        # projection into [0,1]
        proj = lambda x: torch.clamp(x, 0.0, 1.0)
        opt = PGD([self.param], lr=0.2, projection=proj)
        # large negative gradient on first coord, large positive on second
        self.param.data = torch.tensor([0.1, 0.9])
        self.param.grad = torch.tensor([10.0, -10.0])
        opt.step()
        # after unconstrained step: [0.1 - 0.2*10, 0.9 - 0.2*(-10)] = [-1.9, 2.9]
        # after clamp: [0.0, 1.0]
        self.assertTrue(torch.all(self.param.data >= 0.0) and torch.all(self.param.data <= 1.0))
        self.assertTrue(torch.allclose(self.param.data, torch.tensor([0.0, 1.0])))

    def test_momentum_buffer_and_effect(self):
        # test that momentum buffer is created and influences update
        proj = lambda x: x
        opt = PGD([self.param], lr=0.1, momentum=0.9, projection=proj)
        # first step
        self.param.data = torch.tensor([0.0])
        self.param.grad = torch.tensor([1.0])
        opt.step()
        buf1 = opt.state[self.param]['momentum_buffer'].clone()
        self.assertTrue(torch.allclose(buf1, torch.tensor([1.0])))  # initial buf = grad

        # second step: grad again 1.0 => buf = 0.9*1 + 0.1*1 = 1, update uses momentum buf
        self.param.grad = torch.tensor([1.0])
        old = self.param.data.clone()
        opt.step()
        new = self.param.data.clone()
        # parameter moved by lr*buf = 0.1*1 = 0.1
        self.assertTrue(torch.allclose(new, old - 0.1))

    def test_per_group_projection_and_lr(self):
        # two parameters, two groups
        p1 = nn.Parameter(torch.tensor([0.5]))
        p2 = nn.Parameter(torch.tensor([-0.5]))
        # group1: lr=0.1, clamp≥0
        # group2: lr=0.2, identity
        proj1 = lambda x: torch.clamp(x, 0.0, 1.0)
        opt = PGD(
            [
                {'params': [p1], 'lr': 0.1, 'projection': proj1},
                {'params': [p2], 'lr': 0.2, 'projection': lambda x: x}
            ]
        )
        p1.grad = torch.tensor([10.0])
        p2.grad = torch.tensor([10.0])
        opt.step()
        # p1: unconstrained update 0.5 - 0.1*10 = -0.5 -> clamp -> 0
        self.assertTrue(torch.allclose(p1.data, torch.tensor([0.0])))
        # p2: unconstrained update -0.5 - 0.2*10 = -2.5
        self.assertTrue(torch.allclose(p2.data, torch.tensor([-2.5])))

    def test_closure_return(self):
        # ensure loss from closure is returned
        p = nn.Parameter(torch.tensor([1.0]), requires_grad=True)
        opt = PGD([p], lr=0.1, projection=lambda x: x)
        called = {'cnt': 0}
        def closure():
            called['cnt'] += 1
            # dummy loss = p^2
            loss = p * p
            loss.backward()
            return loss
        out = opt.step(closure)
        self.assertEqual(called['cnt'], 1)
        self.assertIsInstance(out, Tensor)
        self.assertTrue(torch.allclose(out, torch.tensor([1.0])))


class TestPGDFunctional(unittest.TestCase):
    def test_pgd_functional_identity(self):
        # single tensor, zero grad, identity projection
        p = torch.nn.Parameter(torch.tensor([1.0, -2.0]))
        g = torch.tensor([0.5, -0.5])
        # one‐shot functional PGD: no weight_decay, no momentum
        pgd([p], [g], [None],
            weight_decay=0.0, momentum=0.0, lr=0.1, dampening=0.0,
            nesterov=False, maximize=False, has_sparse_grad=False,
            foreach=False, fused=False, grad_scale=None, found_inf=None,
            projection=lambda x: x)
        # should be p = p - 0.1*g
        expected = torch.tensor([1.0 - 0.1*0.5, -2.0 - 0.1*(-0.5)])
        self.assertTrue(torch.allclose(p.data, expected))

    def test_pgd_functional_projection_only(self):
        # no gradient step, just projection
        p = torch.nn.Parameter(torch.tensor([2.0, -3.0]))
        g = torch.tensor([0.0, 0.0])
        proj = lambda x: torch.clamp(x, -1.0, 1.0)
        pgd([p], [g], [None],
            weight_decay=0.0, momentum=0.0, lr=0.1, dampening=0.0,
            nesterov=False, maximize=False, has_sparse_grad=False,
            foreach=False, fused=False, grad_scale=None, found_inf=None,
            projection=proj)
        self.assertTrue(torch.allclose(p.data, torch.tensor([1.0, -1.0])))

    def test_functional_momentum_buffer_update(self):
        p = torch.nn.Parameter(torch.tensor([0.0]))
        g = torch.tensor([1.0])
        # prepare a buffer list
        buf = None
        pgd([p], [g], [buf],
            weight_decay=0.0, momentum=0.5, lr=0.1, dampening=0.0,
            nesterov=False, maximize=False, has_sparse_grad=False,
            foreach=False, fused=False, grad_scale=None, found_inf=None,
            projection=lambda x: x)
        # buffer should be created and equal to grad
        # (we don't have direct access to the buffer list here, but we can infer via second call)
        # second call: buffer list must hold previous
        prev = p.data.clone()
        pgd([p], [g], [buf],
            weight_decay=0.0, momentum=0.5, lr=0.1, dampening=0.0,
            nesterov=False, maximize=False, has_sparse_grad=False,
            foreach=False, fused=False, grad_scale=None, found_inf=None,
            projection=lambda x: x)
        # parameter should move by 0.1 * (0.5*1 + 1*0.5) = 0.1*1.0 = 0.1
        self.assertAlmostEqual(float(prev - p.data), 0.1)

if __name__ == "__main__":
    unittest.main()
