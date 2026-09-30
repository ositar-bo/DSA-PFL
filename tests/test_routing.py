"""Numerical tests for the DSA-PFL routing operator."""

from __future__ import annotations

import unittest

import torch

from dsapfl.routing import Conceptor, _conceptor_matrix, conceptor_drift, dsapfl_fusion


class RoutingTests(unittest.TestCase):
    def test_spectral_solution_and_nonexpansiveness(self) -> None:
        local = {"layer.weight": torch.zeros(3, 2)}
        global_state = {"layer.weight": torch.ones(3, 2)}
        conceptor = Conceptor(
            "layer", 2, 3, 16, torch.diag(torch.tensor([0.0, 0.5, 1.0]))
        )
        fused, retention = dsapfl_fusion(
            global_state,
            local,
            {"layer": conceptor},
            {"layer": 0.0},
            {"layer.weight": "layer"},
            "cpu",
            tau=0.25,
            lambda_sub=1.0,
        )
        expected = torch.tensor([[1.0, 1.0], [2 / 3, 2 / 3], [0.5, 0.5]])
        self.assertTrue(torch.allclose(fused["layer.weight"], expected, atol=1e-6))
        self.assertGreater(retention, 0.0)
        self.assertLessEqual(retention, 1.0)

    def test_zero_drift(self) -> None:
        matrix = torch.diag(torch.tensor([0.2, 0.8]))
        left = Conceptor("x", 1, 2, 4, matrix)
        right = Conceptor("x", 2, 2, 4, matrix.clone())
        self.assertEqual(conceptor_drift(left, right), 0.0)

    def test_direct_and_dual_constructors_agree(self) -> None:
        generator = torch.Generator().manual_seed(7)
        features = torch.randn(8, 3, generator=generator)
        direct = _conceptor_matrix(features, aperture=10.0, solver="direct")
        dual = _conceptor_matrix(features, aperture=10.0, solver="auto")
        self.assertTrue(torch.allclose(direct, dual, atol=2e-5, rtol=2e-5))


if __name__ == "__main__":
    unittest.main()
