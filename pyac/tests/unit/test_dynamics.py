import numpy as np
import inspect
import importlib
from scipy.sparse import csr_matrix, isspmatrix_csr


dynamics = importlib.import_module("pyac.core.dynamics")
DynamicsStrategy = dynamics.DynamicsStrategy
FeedforwardStrategy = dynamics.FeedforwardStrategy
RecurrentStrategy = dynamics.RecurrentStrategy
RefractedStrategy = dynamics.RefractedStrategy


class TestDynamicsStrategyABC:
    def test_abc_declares_abstract_methods(self):
        assert inspect.isabstract(DynamicsStrategy)
        assert DynamicsStrategy.__abstractmethods__ == {
            "dynamics_contribution",
            "update_state",
        }


class TestFeedforwardStrategy:
    def test_returns_zero_contribution_for_any_activations(self):
        strategy = FeedforwardStrategy(n=8)

        contrib = strategy.dynamics_contribution(
            area_name="A",
            activations=np.array([1, 3, 5]),
            context={},
        )

        assert contrib.shape == (8,)
        assert contrib.dtype == np.float64
        assert np.all(contrib == 0.0)

    def test_update_state_is_no_op(self):
        strategy = FeedforwardStrategy(n=6)

        strategy.update_state(
            area_name="A",
            firing=np.array([0, 2, 4]),
            total_input=np.array([0.2, 0.1, 0.9, 0.4, 1.1, 0.0]),
            plasticity=0.3,
            context={},
        )

        contrib_after = strategy.dynamics_contribution(
            area_name="A",
            activations=np.array([0, 1]),
            context={},
        )
        assert np.all(contrib_after == 0.0)


class TestRecurrentStrategy:
    def test_dynamics_contribution_sums_active_rows_into_columns(self):
        weights = csr_matrix(
            [
                [0.0, 1.0, 1.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
            ],
            dtype=np.float64,
        )
        strategy = RecurrentStrategy(recurrent_weights=weights)

        contrib = strategy.dynamics_contribution(
            area_name="A",
            activations=np.array([0, 1]),
            context={},
        )

        assert contrib.shape == (4,)
        assert contrib.dtype == np.float64
        assert contrib[0] == 0.0
        assert contrib[1] == 1.0
        assert contrib[2] == 2.0
        assert contrib[3] == 0.0

    def test_dynamics_contribution_preserves_sparse_matrix_storage(self):
        weights = csr_matrix(
            [[0.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 0.0, 0.0]],
            dtype=np.float64,
        )
        original_nnz = weights.nnz
        strategy = RecurrentStrategy(recurrent_weights=weights)

        _ = strategy.dynamics_contribution(
            area_name="A",
            activations=np.array([1]),
            context={},
        )

        assert isspmatrix_csr(strategy.recurrent_weights)
        assert strategy.recurrent_weights.nnz == original_nnz

    def test_update_state_applies_hebbian_multiplicative_update(self):
        weights = csr_matrix(
            [
                [0.0, 1.0, 1.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
                [1.0, 0.0, 0.0, 0.0],
            ],
            dtype=np.float64,
        )
        strategy = RecurrentStrategy(recurrent_weights=weights)

        strategy.update_state(
            area_name="A",
            firing=np.array([2, 3]),
            total_input=np.array([0.0, 0.0, 0.0, 0.0]),
            plasticity=0.5,
            context={"pre_activations": np.array([0, 1])},
        )

        dense = strategy.recurrent_weights.toarray()
        assert dense[0, 2] == 1.5
        assert dense[0, 3] == 0.0
        assert dense[1, 2] == 1.5
        assert dense[1, 3] == 0.0
        assert dense[0, 1] == 1.0
        assert dense[2, 3] == 1.0
        assert dense[3, 0] == 1.0

    def test_update_state_defaults_pre_activations_to_firing(self):
        weights = csr_matrix(
            [[1.0, 1.0], [1.0, 0.0]],
            dtype=np.float64,
        )
        strategy = RecurrentStrategy(recurrent_weights=weights)

        strategy.update_state(
            area_name="A",
            firing=np.array([0]),
            total_input=np.array([0.0, 0.0]),
            plasticity=0.2,
            context={},
        )

        dense = strategy.recurrent_weights.toarray()
        assert dense[0, 0] == 1.2
        assert dense[0, 1] == 1.0


class TestRefractedStrategy:
    def test_initial_contribution_is_all_zeros(self):
        strategy = RefractedStrategy(n=5)

        contrib = strategy.dynamics_contribution(
            area_name="A",
            activations=np.array([0, 2]),
            context={},
        )

        assert contrib.shape == (5,)
        assert np.all(contrib == 0.0)

    def test_update_state_accumulates_negative_bias_for_firing_neurons(self):
        strategy = RefractedStrategy(n=10)
        total_input = np.array([0.0, 0.0, 3.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0])

        strategy.update_state(
            area_name="A",
            firing=np.array([2, 5]),
            total_input=total_input,
            plasticity=0.1,
            context={},
        )

        contrib = strategy.dynamics_contribution(
            area_name="A",
            activations=np.array([2, 5]),
            context={},
        )

        assert np.isclose(strategy.bias[2], 0.3)
        assert np.isclose(strategy.bias[5], 0.2)
        assert np.isclose(contrib[2], -0.3)
        assert np.isclose(contrib[5], -0.2)

    def test_update_state_accumulates_over_multiple_steps(self):
        strategy = RefractedStrategy(n=4)

        strategy.update_state(
            area_name="A",
            firing=np.array([1, 3]),
            total_input=np.array([0.0, 2.0, 0.0, 1.0]),
            plasticity=0.1,
            context={},
        )
        strategy.update_state(
            area_name="A",
            firing=np.array([1]),
            total_input=np.array([0.0, 5.0, 0.0, 0.0]),
            plasticity=0.1,
            context={},
        )

        assert np.allclose(strategy.bias, np.array([0.0, 0.7, 0.0, 0.1]))
        assert np.allclose(
            strategy.dynamics_contribution("A", np.array([1, 3]), {}),
            np.array([0.0, -0.7, 0.0, -0.1]),
        )
