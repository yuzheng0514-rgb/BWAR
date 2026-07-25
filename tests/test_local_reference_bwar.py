from __future__ import annotations

import dataclasses
import unittest
from unittest import mock

import numpy as np
from scipy import linalg as scipy_linalg

import bwar.paper_jcgs.local_reference_bwar as local_reference
from bwar.bwar_experiments import project_spd
from bwar.paper_jcgs.local_reference_bwar import DualRidgeAR1, fit_dual_ridge_ar1
from bwar.paper_jcgs.real_bwar_theory_matched import fit_var


TASK_2_API = (
    "ReferenceState",
    "RollingBuresReference",
    "bures_fixed_point_step",
    "fixed_point_residual",
)
TASK_3_API = (
    "LocalGeometry",
    "bures_transport_map",
    "local_bwar_encode",
    "build_local_bwar_geometry",
    "forecast_local_bwar",
)
ReferenceState = getattr(local_reference, "ReferenceState", None)
RollingBuresReference = getattr(local_reference, "RollingBuresReference", None)
bures_fixed_point_step = getattr(local_reference, "bures_fixed_point_step", None)
fixed_point_residual = getattr(local_reference, "fixed_point_residual", None)
LocalGeometry = getattr(local_reference, "LocalGeometry", None)
bures_transport_map = getattr(local_reference, "bures_transport_map", None)
local_bwar_encode = getattr(local_reference, "local_bwar_encode", None)
local_bwar_decode = getattr(local_reference, "local_bwar_decode", None)
build_local_bwar_geometry = getattr(local_reference, "build_local_bwar_geometry", None)
forecast_local_bwar = getattr(local_reference, "forecast_local_bwar", None)
common_source_indices = getattr(local_reference, "common_source_indices", None)
PrimalRidgeAR1 = getattr(local_reference, "PrimalRidgeAR1", None)
fit_primal_ridge_ar1 = getattr(local_reference, "fit_primal_ridge_ar1", None)
forecast_online_encoded = getattr(local_reference, "forecast_online_encoded", None)
forecast_online_raw_var_window = getattr(
    local_reference,
    "forecast_online_raw_var_window",
    None,
)
forecast_online_raw_var_mean = getattr(
    local_reference,
    "forecast_online_raw_var_mean",
    None,
)


class DualRidgeAR1Test(unittest.TestCase):
    def test_dual_predictions_match_existing_primal_full_ridge(self) -> None:
        rng = np.random.default_rng(20260713)
        coordinates = rng.normal(size=(18, 7))
        ridge = 0.1

        dual = fit_dual_ridge_ar1(coordinates, ridge=ridge)
        primal = fit_var(coordinates, len(coordinates), lam=ridge, model="full")

        for state in coordinates[[0, 4, 17]]:
            with self.subTest(state=state):
                np.testing.assert_allclose(
                    dual.predict(state),
                    np.r_[1.0, state] @ primal,
                    rtol=1e-10,
                    atol=1e-10,
                )

    def test_dual_fit_supports_more_coordinates_than_samples(self) -> None:
        rng = np.random.default_rng(8)
        coordinates = rng.normal(size=(12, 40))
        ridge = 1e-2

        model = fit_dual_ridge_ar1(coordinates, ridge=ridge)
        primal = fit_var(coordinates, len(coordinates), lam=ridge, model="full")
        out_of_sample_states = rng.normal(loc=0.37, scale=1.4, size=(3, 40))

        self.assertEqual(model.x_centered.shape, (11, 40))
        self.assertEqual(model.dual_coef.shape, (11, 40))
        for state in out_of_sample_states:
            with self.subTest(state_norm=np.linalg.norm(state)):
                prediction = model.predict(state)
                self.assertEqual(prediction.shape, (40,))
                np.testing.assert_allclose(
                    prediction,
                    np.r_[1.0, state] @ primal,
                    rtol=1e-10,
                    atol=1e-10,
                )

    def test_model_is_a_frozen_dataclass_with_expected_fields(self) -> None:
        model = fit_dual_ridge_ar1(np.arange(12, dtype=float).reshape(4, 3), ridge=0.1)

        self.assertTrue(dataclasses.is_dataclass(model))
        self.assertEqual(
            tuple(field.name for field in dataclasses.fields(model)),
            ("x_mean", "y_mean", "x_centered", "dual_coef"),
        )
        with self.assertRaises(dataclasses.FrozenInstanceError):
            model.x_mean = np.zeros(3)

    def test_model_defensively_copies_and_freezes_all_stored_arrays(self) -> None:
        source_arrays = {
            "x_mean": np.array([1.0, 2.0]),
            "y_mean": np.array([3.0, 4.0]),
            "x_centered": np.array([[1.0, -1.0], [-1.0, 1.0]]),
            "dual_coef": np.array([[0.5, 0.25], [-0.5, -0.25]]),
        }
        expected = {name: value.copy() for name, value in source_arrays.items()}

        model = DualRidgeAR1(**source_arrays)
        for value in source_arrays.values():
            value.fill(99.0)

        for name, original in source_arrays.items():
            with self.subTest(field=name):
                stored = getattr(model, name)
                np.testing.assert_array_equal(stored, expected[name])
                self.assertFalse(np.shares_memory(stored, original))
                self.assertFalse(stored.flags.writeable)
                with self.assertRaises(ValueError):
                    stored.flat[0] = 0.0
                with self.assertRaises(ValueError):
                    stored.setflags(write=True)

    def test_recursive_prediction_repeats_one_step_prediction(self) -> None:
        rng = np.random.default_rng(41)
        coordinates = rng.normal(size=(15, 5))
        model = fit_dual_ridge_ar1(coordinates, ridge=0.25)
        state = coordinates[-1]

        expected = state
        for _ in range(4):
            expected = model.predict(expected)

        np.testing.assert_allclose(
            model.predict_recursive(state, horizon=4),
            expected,
            rtol=0.0,
            atol=0.0,
        )

    def test_recursive_prediction_requires_positive_integer_horizon(self) -> None:
        model = fit_dual_ridge_ar1(np.arange(12, dtype=float).reshape(4, 3), ridge=0.1)

        invalid_horizons = (
            0,
            -1,
            True,
            False,
            1.0,
            3.5,
            np.float64(2.0),
            np.nan,
            np.inf,
            "2",
            np.array(2),
        )
        for horizon in invalid_horizons:
            with self.subTest(horizon=repr(horizon)):
                with self.assertRaisesRegex(ValueError, "integer at least one"):
                    model.predict_recursive(np.zeros(3), horizon=horizon)

        np.testing.assert_allclose(
            model.predict_recursive(np.zeros(3), horizon=np.int64(2)),
            model.predict_recursive(np.zeros(3), horizon=2),
        )

    def test_predict_rejects_state_with_nonexact_shape(self) -> None:
        model = fit_dual_ridge_ar1(np.arange(12, dtype=float).reshape(4, 3), ridge=0.1)

        for state in (np.zeros(2), np.zeros((1, 3)), np.zeros((3, 1))):
            with self.subTest(shape=state.shape):
                with self.assertRaisesRegex(ValueError, "state shape"):
                    model.predict(state)

    def test_predict_rejects_nonfinite_state(self) -> None:
        model = fit_dual_ridge_ar1(np.arange(12, dtype=float).reshape(4, 3), ridge=0.1)

        for value in (np.nan, np.inf, -np.inf):
            state = np.zeros(3)
            state[1] = value
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "state must be finite"):
                    model.predict(state)

    def test_predict_rejects_nonfinite_result_from_finite_extreme_state(self) -> None:
        model = DualRidgeAR1(
            x_mean=np.array([0.0]),
            y_mean=np.array([0.0]),
            x_centered=np.array([[1.0], [-1.0]]),
            dual_coef=np.array([[1e308], [-1e308]]),
        )
        state = np.array([1e308])

        self.assertTrue(np.isfinite(state).all())
        for field in dataclasses.fields(model):
            self.assertTrue(np.isfinite(getattr(model, field.name)).all())
        with self.assertRaisesRegex(ValueError, "prediction became nonfinite"):
            model.predict(state)

    def test_fit_and_predict_reject_complex_inputs(self) -> None:
        real_coordinates = np.arange(12, dtype=float).reshape(4, 3)
        model = fit_dual_ridge_ar1(real_coordinates, ridge=0.1)
        object_coordinates = real_coordinates.astype(object)
        object_coordinates[0, 0] = 1.0 + 0.0j
        object_state = np.ones(3, dtype=object)
        object_state[0] = 1.0 + 0.0j
        complex_cases = (
            (
                "coordinates_with_imaginary_part",
                lambda: fit_dual_ridge_ar1(real_coordinates + 1.0j, ridge=0.1),
            ),
            (
                "coordinates_complex_dtype",
                lambda: fit_dual_ridge_ar1(real_coordinates.astype(complex), ridge=0.1),
            ),
            (
                "coordinates_object_dtype",
                lambda: fit_dual_ridge_ar1(object_coordinates, ridge=0.1),
            ),
            ("state_with_imaginary_part", lambda: model.predict(np.ones(3) + 1.0j)),
            ("state_complex_dtype", lambda: model.predict(np.ones(3, dtype=complex))),
            ("state_object_dtype", lambda: model.predict(object_state)),
            (
                "recursive_state_complex_dtype",
                lambda: model.predict_recursive(np.ones(3, dtype=complex), horizon=1),
            ),
        )

        for name, operation in complex_cases:
            with self.subTest(argument=name):
                with self.assertRaisesRegex(ValueError, "complex values are not supported"):
                    operation()

    def test_huge_integer_conversion_failures_are_value_errors(self) -> None:
        huge_integer = 10**400
        model = fit_dual_ridge_ar1(np.arange(12, dtype=float).reshape(4, 3), ridge=0.1)
        cases = (
            (
                "coordinates",
                lambda: fit_dual_ridge_ar1(
                    np.array([[huge_integer], [huge_integer], [huge_integer]], dtype=object),
                    ridge=0.1,
                ),
                "coordinates must be a real numeric array",
            ),
            (
                "state",
                lambda: model.predict(np.array([huge_integer] * 3, dtype=object)),
                "state must be a real numeric array",
            ),
            (
                "recursive_state",
                lambda: model.predict_recursive(
                    np.array([huge_integer] * 3, dtype=object),
                    horizon=1,
                ),
                "state must be a real numeric array",
            ),
            (
                "ridge",
                lambda: fit_dual_ridge_ar1(np.ones((4, 2)), ridge=huge_integer),
                "ridge must be a finite real scalar greater than zero",
            ),
        )

        for name, operation, message in cases:
            with self.subTest(argument=name):
                with self.assertRaisesRegex(ValueError, message):
                    operation()

    def test_fit_rejects_insufficient_rows(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least three"):
            fit_dual_ridge_ar1(np.ones((2, 4)), ridge=0.1)

    def test_fit_rejects_nonfinite_coordinates(self) -> None:
        for value in (np.nan, np.inf, -np.inf):
            coordinates = np.ones((4, 2))
            coordinates[1, 0] = value
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "finite"):
                    fit_dual_ridge_ar1(coordinates, ridge=0.1)

    def test_fit_rejects_overflow_from_finite_extreme_coordinates(self) -> None:
        extreme_cases = {
            "intermediate means": np.array([[1e308], [1e308], [-1e308]]),
            "Gram matrix": np.array([[1e308], [-1e308], [0.0]]),
        }

        for stage, coordinates in extreme_cases.items():
            self.assertTrue(np.isfinite(coordinates).all())
            with self.subTest(stage=stage):
                with self.assertRaisesRegex(ValueError, f"nonfinite {stage}"):
                    fit_dual_ridge_ar1(coordinates, ridge=0.1)

    def test_fit_rejects_nonfinite_dual_coefficients(self) -> None:
        coordinates = np.array([[0.0], [0.0], [1.0]])
        smallest_positive_float = np.nextafter(0.0, 1.0)

        with self.assertRaisesRegex(ValueError, "nonfinite dual coefficients"):
            fit_dual_ridge_ar1(coordinates, ridge=smallest_positive_float)

    def test_fit_normalizes_subnormal_ridge_solve_failure(self) -> None:
        coordinates = np.array([[0.0], [1.0], [2.0]])
        smallest_positive_float = np.nextafter(0.0, 1.0)

        with self.assertRaisesRegex(ValueError, "dual ridge system could not be solved"):
            fit_dual_ridge_ar1(coordinates, ridge=smallest_positive_float)

    def test_fit_rejects_non_2d_input(self) -> None:
        for coordinates in (np.ones(4), np.ones((2, 2, 2))):
            with self.subTest(ndim=coordinates.ndim):
                with self.assertRaisesRegex(ValueError, "two-dimensional"):
                    fit_dual_ridge_ar1(coordinates, ridge=0.1)

    def test_fit_requires_finite_positive_real_scalar_ridge(self) -> None:
        invalid_ridges = (
            0.0,
            -0.1,
            True,
            False,
            np.nan,
            np.inf,
            -np.inf,
            "0.1",
            "ridge",
            1.0 + 0.0j,
            np.array(0.1),
            np.array([0.1]),
            np.array([0.1, 0.2]),
            np.array([[0.1]]),
        )
        for ridge in invalid_ridges:
            with self.subTest(ridge=repr(ridge)):
                with self.assertRaisesRegex(ValueError, "finite real scalar greater than zero"):
                    fit_dual_ridge_ar1(np.ones((4, 2)), ridge=ridge)

        for ridge in (1, 0.1, np.int64(1), np.float64(0.1)):
            with self.subTest(valid_ridge=repr(ridge)):
                fit_dual_ridge_ar1(np.ones((4, 2)), ridge=ridge)


class CommonSourceIndicesTest(unittest.TestCase):
    def test_returns_exact_common_support_and_one_source_boundary(self) -> None:
        self.assertIsNotNone(
            common_source_indices,
            "missing Task 4 API: common_source_indices",
        )

        np.testing.assert_array_equal(
            common_source_indices(90, 4),
            np.arange(47, 86),
        )
        np.testing.assert_array_equal(
            common_source_indices(52, 4, 48),
            np.array([47]),
        )

    def test_requires_strict_positive_integer_scalars(self) -> None:
        invalid_values = (True, False, 1.0, np.float64(2.0), "2", np.array(2))
        defaults = {
            "n_states": 90,
            "horizon": 4,
            "max_window_length": 48,
        }
        for argument in defaults:
            for invalid in invalid_values:
                values = defaults | {argument: invalid}
                with self.subTest(argument=argument, invalid=repr(invalid)):
                    with self.assertRaisesRegex(
                        ValueError,
                        rf"{argument} must be an integer at least 1",
                    ):
                        common_source_indices(**values)

        for argument in defaults:
            values = defaults | {argument: 0}
            with self.subTest(argument=argument, invalid="zero"):
                with self.assertRaisesRegex(
                    ValueError,
                    rf"{argument} must be an integer at least 1",
                ):
                    common_source_indices(**values)

        np.testing.assert_array_equal(
            common_source_indices(np.int64(52), np.int64(4), np.int64(48)),
            np.array([47]),
        )

    def test_rejects_stream_with_no_valid_source(self) -> None:
        for arguments in ((51, 4, 48), (4, 4, 1), (10, 3, 9)):
            with self.subTest(arguments=arguments):
                with self.assertRaisesRegex(ValueError, "no valid source"):
                    common_source_indices(*arguments)


class PrimalRidgeAR1Test(unittest.TestCase):
    def test_predictions_match_existing_full_var_in_and_out_of_sample(self) -> None:
        self.assertIsNotNone(
            fit_primal_ridge_ar1,
            "missing Task 4 API: fit_primal_ridge_ar1",
        )
        rng = np.random.default_rng(20260714)
        raw = rng.normal(size=(120, 5))
        ridge = 0.17

        primal = fit_primal_ridge_ar1(raw, ridge=ridge)
        existing = fit_var(raw, len(raw), lam=ridge, model="full")
        states = np.vstack((raw[[0, 37, -1]], rng.normal(size=(4, raw.shape[1]))))

        for state in states:
            with self.subTest(state_norm=float(np.linalg.norm(state))):
                np.testing.assert_allclose(
                    primal.predict(state),
                    np.r_[1.0, state] @ existing,
                    rtol=1e-11,
                    atol=1e-11,
                )

    def test_fit_solves_and_stores_only_dimension_sized_arrays(self) -> None:
        rng = np.random.default_rng(411)
        n_rows, dimension = 240, 4
        raw = rng.normal(size=(n_rows, dimension))

        with mock.patch.object(np.linalg, "solve", wraps=np.linalg.solve) as solve:
            model = fit_primal_ridge_ar1(raw, ridge=0.1)

        self.assertEqual(solve.call_count, 1)
        self.assertEqual(solve.call_args.args[0].shape, (dimension, dimension))
        self.assertEqual(solve.call_args.args[1].shape, (dimension, dimension))
        self.assertEqual(model.coef.shape, (dimension, dimension))
        self.assertEqual(model.x_mean.shape, (dimension,))
        self.assertEqual(model.y_mean.shape, (dimension,))
        for field in dataclasses.fields(model):
            self.assertNotIn(n_rows, getattr(model, field.name).shape)

    def test_model_is_frozen_and_owns_read_only_parameter_arrays(self) -> None:
        source = {
            "x_mean": np.array([1.0, 2.0]),
            "y_mean": np.array([3.0, 4.0]),
            "coef": np.array([[0.2, 0.3], [0.4, 0.5]]),
        }
        expected = {name: value.copy() for name, value in source.items()}

        model = PrimalRidgeAR1(**source)
        for value in source.values():
            value.fill(99.0)

        self.assertTrue(dataclasses.is_dataclass(model))
        with self.assertRaises(dataclasses.FrozenInstanceError):
            model.coef = np.eye(2)
        for name, original in source.items():
            with self.subTest(field=name):
                stored = getattr(model, name)
                np.testing.assert_array_equal(stored, expected[name])
                self.assertFalse(np.shares_memory(stored, original))
                self.assertFalse(stored.flags.writeable)
                with self.assertRaises(ValueError):
                    stored.flat[0] = 0.0
                with self.assertRaises(ValueError):
                    stored.setflags(write=True)

    def test_recursive_prediction_and_validation_match_dual_semantics(self) -> None:
        rng = np.random.default_rng(522)
        raw = rng.normal(size=(20, 3))
        model = fit_primal_ridge_ar1(raw, ridge=0.2)

        expected = raw[-1]
        for _ in range(5):
            expected = model.predict(expected)
        np.testing.assert_array_equal(
            model.predict_recursive(raw[-1], horizon=5),
            expected,
        )

        for horizon in (0, -1, True, 1.0, np.array(1)):
            with self.subTest(horizon=repr(horizon)):
                with self.assertRaisesRegex(ValueError, "integer at least one"):
                    model.predict_recursive(raw[-1], horizon=horizon)
        for state in (np.zeros(2), np.zeros((1, 3))):
            with self.subTest(shape=state.shape):
                with self.assertRaisesRegex(ValueError, "state shape"):
                    model.predict(state)
        with self.assertRaisesRegex(ValueError, "state must be finite"):
            model.predict(np.array([0.0, np.nan, 0.0]))

    def test_fit_rejects_malformed_coordinates_and_ridge(self) -> None:
        malformed_coordinates = (
            np.ones(4),
            np.ones((2, 3)),
            np.array([[1.0, 2.0], [3.0, np.nan], [4.0, 5.0]]),
            np.ones((3, 2), dtype=complex),
        )
        for coordinates in malformed_coordinates:
            with self.subTest(shape=coordinates.shape, dtype=str(coordinates.dtype)):
                with self.assertRaises(ValueError):
                    fit_primal_ridge_ar1(coordinates, ridge=0.1)

        for ridge in (0, -0.1, True, np.nan, np.inf, "0.1", np.array(0.1)):
            with self.subTest(ridge=repr(ridge)):
                with self.assertRaisesRegex(ValueError, "finite real scalar"):
                    fit_primal_ridge_ar1(np.ones((4, 2)), ridge=ridge)


class OnlineEncodedForecastTest(unittest.TestCase):
    def test_matches_independent_recent_window_fit_recursion_and_decode(self) -> None:
        self.assertIsNotNone(
            forecast_online_encoded,
            "missing Task 4 API: forecast_online_encoded",
        )
        rng = np.random.default_rng(633)
        coordinates = rng.normal(size=(20, 5))
        source_index = 14
        window_length = 7
        ridge = 0.13
        horizon = 3

        manual_model = fit_dual_ridge_ar1(
            coordinates[source_index - window_length + 1 : source_index + 1],
            ridge=ridge,
        )
        expected_coordinate = manual_model.predict_recursive(
            coordinates[source_index],
            horizon=horizon,
        )

        def decode(coordinate: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            return coordinate[:2], np.diag(np.exp(coordinate[2:4]))

        expected_mean, expected_covariance = decode(expected_coordinate)
        actual_mean, actual_covariance = forecast_online_encoded(
            coordinates,
            source_index,
            window_length,
            ridge,
            horizon,
            decode,
        )

        np.testing.assert_allclose(actual_mean, expected_mean, rtol=0.0, atol=0.0)
        np.testing.assert_allclose(
            actual_covariance,
            expected_covariance,
            rtol=1e-14,
            atol=1e-14,
        )

    def test_decoder_is_called_once_with_expected_recursive_coordinate(self) -> None:
        rng = np.random.default_rng(744)
        coordinates = rng.normal(size=(15, 4))
        source_index, window_length, horizon = 10, 6, 4
        model = fit_dual_ridge_ar1(
            coordinates[5:11],
            ridge=0.2,
        )
        expected_coordinate = model.predict_recursive(coordinates[10], horizon=4)
        decoded_coordinates: list[np.ndarray] = []

        def decode(coordinate: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            decoded_coordinates.append(coordinate.copy())
            return coordinate[:2], np.eye(2)

        forecast_online_encoded(
            coordinates,
            source_index,
            window_length,
            0.2,
            horizon,
            decode,
        )

        self.assertEqual(len(decoded_coordinates), 1)
        np.testing.assert_array_equal(decoded_coordinates[0], expected_coordinate)

    def test_future_coordinate_changes_cannot_affect_forecast(self) -> None:
        rng = np.random.default_rng(855)
        baseline = rng.normal(size=(22, 5))
        changed = baseline.copy()
        source_index = 13
        changed[source_index + 1 :] = rng.normal(
            loc=1e6,
            scale=1e4,
            size=changed[source_index + 1 :].shape,
        )

        def decode(coordinate: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            return coordinate[:2], np.diag(np.exp(coordinate[2:4]))

        baseline_forecast = forecast_online_encoded(
            baseline, source_index, 8, 0.1, 3, decode
        )
        changed_forecast = forecast_online_encoded(
            changed, source_index, 8, 0.1, 3, decode
        )

        np.testing.assert_array_equal(changed_forecast[0], baseline_forecast[0])
        np.testing.assert_array_equal(changed_forecast[1], baseline_forecast[1])

    def test_nonfinite_coordinates_are_checked_only_in_active_window(self) -> None:
        rng = np.random.default_rng(856)
        baseline = rng.normal(size=(15, 4))
        source_index = 8
        window_length = 5
        decode = lambda coordinate: (coordinate[:2], np.eye(2))
        expected = forecast_online_encoded(
            baseline,
            source_index,
            window_length,
            0.1,
            2,
            decode,
        )

        before_lookback = baseline.copy()
        before_lookback[1, 0] = np.nan
        after_source = baseline.copy()
        after_source[source_index + 1, 0] = np.inf
        for location, changed in (
            ("before_lookback", before_lookback),
            ("after_source", after_source),
        ):
            with self.subTest(location=location):
                actual = forecast_online_encoded(
                    changed,
                    source_index,
                    window_length,
                    0.1,
                    2,
                    decode,
                )
                np.testing.assert_array_equal(actual[0], expected[0])
                np.testing.assert_array_equal(actual[1], expected[1])

        active = baseline.copy()
        active[source_index - window_length + 1, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "active coordinates must be finite"):
            forecast_online_encoded(
                active,
                source_index,
                window_length,
                0.1,
                2,
                decode,
            )

    def test_decoder_output_is_finite_dimension_matched_and_spd_floored(self) -> None:
        coordinates = np.arange(40, dtype=float).reshape(10, 4)
        mean, covariance = forecast_online_encoded(
            coordinates,
            7,
            6,
            0.1,
            2,
            lambda coordinate: (
                coordinate[:2],
                np.array([[0.0, 0.0], [0.0, -2.0]]),
            ),
        )

        self.assertTrue(np.isfinite(mean).all())
        self.assertTrue(np.isfinite(covariance).all())
        self.assertGreaterEqual(
            np.linalg.eigvalsh(covariance).min(),
            1e-8 * (1.0 - 1e-8),
        )

        malformed_decoders = (
            (lambda _: np.zeros(2), "decode must return"),
            (lambda _: (np.array([np.nan]), np.eye(1)), "decoded mean must be finite"),
            (lambda _: (np.ones(2), np.eye(1)), "incompatible"),
            (lambda _: (np.ones(1), np.array([[np.inf]])), "decoded covariance must be finite"),
            (lambda _: (np.ones(1, dtype=complex), np.eye(1)), "real numeric array"),
        )
        for decoder, message in malformed_decoders:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    forecast_online_encoded(
                        coordinates,
                        7,
                        6,
                        0.1,
                        2,
                        decoder,
                    )

    def test_spd_floor_check_allows_eigendecomposition_roundoff(self) -> None:
        dimension = 30
        basis, _ = np.linalg.qr(
            np.random.default_rng(0).normal(size=(dimension, dimension))
        )
        covariance = (
            basis * np.geomspace(1e-8, 9.339386500557811, dimension)
        ) @ basis.T
        projected = project_spd(covariance, eps=1e-8)
        floor_ratio = np.linalg.eigvalsh(projected).min() / 1e-8
        self.assertLess(floor_ratio, 1.0 - 1e-8)
        self.assertGreater(floor_ratio, 1.0 - 1e-6)

        mean, actual = forecast_online_encoded(
            np.arange(400, dtype=float).reshape(10, 40),
            7,
            6,
            0.1,
            2,
            lambda _: (np.zeros(dimension), covariance),
        )

        np.testing.assert_array_equal(mean, np.zeros(dimension))
        self.assertGreater(np.linalg.eigvalsh(actual).min(), 0.0)

    def test_spd_floor_is_numerically_resolved_at_large_matrix_scale(self) -> None:
        dimension = 30
        basis, _ = np.linalg.qr(
            np.random.default_rng(0).normal(size=(dimension, dimension))
        )
        covariance = (
            basis * np.geomspace(1e-8, 1e6, dimension)
        ) @ basis.T

        mean, actual = forecast_online_encoded(
            np.arange(400, dtype=float).reshape(10, 40),
            7,
            6,
            0.1,
            2,
            lambda _: (np.zeros(dimension), covariance),
        )

        np.testing.assert_array_equal(mean, np.zeros(dimension))
        eigenvalues = np.linalg.eigvalsh(actual)
        numerical_resolution = (
            64.0
            * np.finfo(float).eps
            * dimension
            * max(float(np.abs(eigenvalues).max()), 1.0)
        )
        self.assertGreaterEqual(
            eigenvalues.min(),
            1e-8 + numerical_resolution,
        )

    def test_rejects_malformed_inputs_and_horizons(self) -> None:
        coordinates = np.arange(40, dtype=float).reshape(10, 4)
        decode = lambda coordinate: (coordinate[:2], np.eye(2))
        cases = (
            (np.ones(10), 7, 6, 0.1, 2, decode),
            (np.ones((10, 0)), 7, 6, 0.1, 2, decode),
            (np.ones((10, 4), dtype=bool), 7, 6, 0.1, 2, decode),
            (coordinates, True, 6, 0.1, 2, decode),
            (coordinates, 10, 6, 0.1, 2, decode),
            (coordinates, 4, 6, 0.1, 2, decode),
            (coordinates, 7, 2, 0.1, 2, decode),
            (coordinates, 7, 6.0, 0.1, 2, decode),
            (coordinates, 7, 6, 0.0, 2, decode),
            (coordinates, 7, 6, 0.1, 0, decode),
            (coordinates, 7, 6, 0.1, True, decode),
            (coordinates, 7, 6, 0.1, 2, None),
        )
        for arguments in cases:
            with self.subTest(arguments=tuple(map(repr, arguments[1:]))):
                with self.assertRaises(ValueError):
                    forecast_online_encoded(*arguments)

        nonfinite = coordinates.copy()
        nonfinite[2, 1] = np.nan
        with self.assertRaisesRegex(ValueError, "coordinates must be finite"):
            forecast_online_encoded(nonfinite, 7, 6, 0.1, 2, decode)


class OnlineRawVARForecastTest(unittest.TestCase):
    @staticmethod
    def _manual_target_window(
        raw: np.ndarray,
        source_state_index: int,
        state_window_size: int,
        window_length: int,
        ridge: float,
        horizon: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        source_stop = (source_state_index + 1) * state_window_size
        lookback = window_length * state_window_size
        training = raw[source_stop - lookback : source_stop]
        coefficients = fit_var(
            training,
            len(training),
            lam=ridge,
            model="full",
        )
        state = training[-1]
        predictions = []
        for _ in range(horizon * state_window_size):
            state = np.r_[1.0, state] @ coefficients
            predictions.append(state)
        return np.asarray(predictions[-state_window_size:]), training

    def test_window_matches_independent_exact_lookback_and_target_recursion(self) -> None:
        self.assertIsNotNone(
            forecast_online_raw_var_window,
            "missing Task 4 API: forecast_online_raw_var_window",
        )
        rng = np.random.default_rng(966)
        raw = rng.normal(size=(36, 3))
        arguments = {
            "source_state_index": 6,
            "state_window_size": 3,
            "window_length": 4,
            "ridge": 0.15,
            "horizon": 2,
        }
        expected_window, _ = self._manual_target_window(raw, **arguments)

        actual_window = forecast_online_raw_var_window(raw, **arguments)

        self.assertEqual(actual_window.shape, (3, 3))
        np.testing.assert_allclose(
            actual_window,
            expected_window,
            rtol=1e-11,
            atol=1e-11,
        )

    def test_mean_matches_independent_target_window_column_mean(self) -> None:
        self.assertIsNotNone(
            forecast_online_raw_var_mean,
            "missing Task 4 API: forecast_online_raw_var_mean",
        )
        rng = np.random.default_rng(1077)
        raw = rng.normal(size=(48, 2))
        arguments = {
            "source_state_index": 8,
            "state_window_size": 4,
            "window_length": 5,
            "ridge": 0.08,
            "horizon": 3,
        }
        expected_window, _ = self._manual_target_window(raw, **arguments)

        actual_mean = forecast_online_raw_var_mean(raw, **arguments)

        self.assertEqual(actual_mean.shape, (2,))
        np.testing.assert_allclose(
            actual_mean,
            expected_window.mean(axis=0),
            rtol=1e-11,
            atol=1e-11,
        )

    def test_fitter_receives_exact_last_raw_lookback_ending_at_source(self) -> None:
        raw = np.arange(120, dtype=float).reshape(40, 3)
        arguments = {
            "source_state_index": 6,
            "state_window_size": 4,
            "window_length": 3,
            "ridge": 0.11,
            "horizon": 2,
        }
        _, expected_training = self._manual_target_window(raw, **arguments)

        with mock.patch.object(
            local_reference,
            "fit_primal_ridge_ar1",
            wraps=fit_primal_ridge_ar1,
        ) as fitter:
            forecast_online_raw_var_window(raw, **arguments)

        fitter.assert_called_once()
        np.testing.assert_array_equal(fitter.call_args.args[0], expected_training)
        self.assertEqual(fitter.call_args.kwargs, {"ridge": 0.11})

    def test_rows_before_lookback_and_after_source_cannot_affect_output(self) -> None:
        rng = np.random.default_rng(1188)
        baseline = rng.normal(size=(52, 3))
        changed = baseline.copy()
        arguments = {
            "source_state_index": 7,
            "state_window_size": 4,
            "window_length": 4,
            "ridge": 0.09,
            "horizon": 3,
        }
        source_stop = 32
        training_start = 16
        changed[:training_start] = rng.normal(
            loc=-1e6,
            scale=1e3,
            size=changed[:training_start].shape,
        )
        changed[source_stop:] = rng.normal(
            loc=1e6,
            scale=1e3,
            size=changed[source_stop:].shape,
        )

        baseline_window = forecast_online_raw_var_window(baseline, **arguments)
        changed_window = forecast_online_raw_var_window(changed, **arguments)
        baseline_mean = forecast_online_raw_var_mean(baseline, **arguments)
        changed_mean = forecast_online_raw_var_mean(changed, **arguments)

        np.testing.assert_array_equal(changed_window, baseline_window)
        np.testing.assert_array_equal(changed_mean, baseline_mean)

    def test_nonfinite_raw_rows_are_checked_only_in_training_slice(self) -> None:
        rng = np.random.default_rng(1189)
        baseline = rng.normal(size=(40, 2))
        arguments = {
            "source_state_index": 5,
            "state_window_size": 4,
            "window_length": 3,
            "ridge": 0.1,
            "horizon": 2,
        }
        expected = forecast_online_raw_var_window(baseline, **arguments)

        before_lookback = baseline.copy()
        before_lookback[4, 0] = np.inf
        after_source_stop = baseline.copy()
        after_source_stop[24, 0] = np.nan
        for location, changed in (
            ("before_lookback", before_lookback),
            ("after_source_stop", after_source_stop),
        ):
            with self.subTest(location=location):
                actual = forecast_online_raw_var_window(changed, **arguments)
                np.testing.assert_array_equal(actual, expected)

        active = baseline.copy()
        active[12, 0] = np.inf
        with self.assertRaisesRegex(ValueError, "training raw rows must be finite"):
            forecast_online_raw_var_window(active, **arguments)

    def test_mean_calls_window_endpoint_and_averages_its_rows(self) -> None:
        raw = np.arange(24, dtype=float).reshape(12, 2)
        target_window = np.array([[1.0, 3.0], [5.0, 7.0], [9.0, 11.0]])

        with mock.patch.object(
            local_reference,
            "forecast_online_raw_var_window",
            return_value=target_window,
        ) as window_forecast:
            actual = forecast_online_raw_var_mean(raw, 3, 3, 2, 0.1, 4)

        window_forecast.assert_called_once_with(raw, 3, 3, 2, 0.1, 4)
        np.testing.assert_array_equal(actual, target_window.mean(axis=0))

    def test_rejects_malformed_raw_inputs_indices_and_horizons(self) -> None:
        raw = np.arange(80, dtype=float).reshape(40, 2)
        cases = (
            (np.ones(20), 5, 4, 3, 0.1, 2),
            (np.ones((20, 0)), 3, 4, 3, 0.1, 2),
            (np.ones((20, 2), dtype=bool), 3, 4, 3, 0.1, 2),
            (np.ones((20, 2), dtype=complex), 3, 4, 3, 0.1, 2),
            (raw, True, 4, 3, 0.1, 2),
            (raw, -1, 4, 3, 0.1, 2),
            (raw, 10, 4, 3, 0.1, 2),
            (raw, 5, 0, 3, 0.1, 2),
            (raw, 5, 4.0, 3, 0.1, 2),
            (raw, 2, 4, 4, 0.1, 2),
            (raw, 5, 4, 0, 0.1, 2),
            (raw, 5, 4, 3.0, 0.1, 2),
            (raw, 5, 1, 2, 0.1, 2),
            (raw, 5, 4, 3, 0.0, 2),
            (raw, 5, 4, 3, 0.1, 0),
            (raw, 5, 4, 3, 0.1, True),
            (raw, 5, 4, 3, 0.1, 2.0),
        )
        for arguments in cases:
            with self.subTest(arguments=tuple(map(repr, arguments[1:]))):
                with self.assertRaises(ValueError):
                    forecast_online_raw_var_window(*arguments)

        nonfinite = raw.copy()
        nonfinite[12, 0] = np.inf
        with self.assertRaisesRegex(ValueError, "training raw rows must be finite"):
            forecast_online_raw_var_window(nonfinite, 5, 4, 3, 0.1, 2)


class Task2APIPresenceTest(unittest.TestCase):
    def test_task_2_api_exists(self) -> None:
        missing = [name for name in TASK_2_API if not hasattr(local_reference, name)]
        self.assertEqual(missing, [], f"missing Task 2 API: {missing}")


@unittest.skipUnless(
    all(hasattr(local_reference, name) for name in TASK_2_API),
    "Task 2 API not implemented",
)
class RollingBuresReferenceTest(unittest.TestCase):
    @staticmethod
    def _stream(length: int = 12) -> tuple[np.ndarray, np.ndarray]:
        times = np.arange(length, dtype=float)
        means = np.column_stack((times, times**2, -0.5 * times))
        covs = np.array(
            [
                np.diag(
                    [
                        1.0 + 0.17 * time,
                        2.0 + 0.03 * time**2,
                        0.8 + 0.11 * (time % 4),
                    ]
                )
                for time in times
            ]
        )
        return means, covs

    @staticmethod
    def _conditioned_covariance(*, condition: float, rotate: bool) -> np.ndarray:
        dimension = 30
        spectrum = np.geomspace(1.0 / condition, 1.0, dimension)
        if not rotate:
            return np.diag(spectrum)
        rng = np.random.default_rng(20260713)
        basis, _ = np.linalg.qr(rng.normal(size=(dimension, dimension)))
        covariance = (basis * spectrum) @ basis.T
        return 0.5 * (covariance + covariance.T)

    @staticmethod
    def _closed_form_diagonal_barycenter(covs: np.ndarray) -> np.ndarray:
        diagonals = np.diagonal(covs, axis1=1, axis2=2)
        return np.diag(np.mean(np.sqrt(diagonals), axis=0) ** 2)

    def test_fixed_point_preserves_identical_condition_1e8_diagonal_covariances(self) -> None:
        covariance = self._conditioned_covariance(condition=1e8, rotate=False)
        covs = np.repeat(covariance[None, :, :], 4, axis=0)

        step = bures_fixed_point_step(covariance, covs)
        relative_error = np.linalg.norm(step - covariance, "fro") / np.linalg.norm(
            covariance,
            "fro",
        )

        self.assertLessEqual(relative_error, 1e-9)

    def test_fixed_point_preserves_identical_condition_1e8_rotated_covariances(self) -> None:
        covariance = self._conditioned_covariance(condition=1e8, rotate=True)
        covs = np.repeat(covariance[None, :, :], 4, axis=0)

        step = bures_fixed_point_step(covariance, covs)
        relative_error = np.linalg.norm(step - covariance, "fro") / np.linalg.norm(
            covariance,
            "fro",
        )

        self.assertLessEqual(relative_error, 1e-8)

    def test_fixed_point_retries_numpy_svd_failure_with_scipy_gesvd(self) -> None:
        covariance = self._conditioned_covariance(condition=1e8, rotate=True)
        covs = np.repeat(covariance[None, :, :], 4, axis=0)
        original_fallback = scipy_linalg.svd

        with (
            mock.patch.object(
                local_reference.np.linalg,
                "svd",
                side_effect=np.linalg.LinAlgError("forced numpy SVD failure"),
            ),
            mock.patch.object(
                scipy_linalg,
                "svd",
                wraps=original_fallback,
            ) as fallback,
        ):
            try:
                step = bures_fixed_point_step(covariance, covs)
            except Exception as exc:
                self.fail(f"gesvd fallback did not recover: {exc}")

        relative_error = np.linalg.norm(step - covariance, "fro") / np.linalg.norm(
            covariance,
            "fro",
        )
        self.assertLessEqual(relative_error, 1e-8)
        self.assertEqual(fallback.call_count, len(covs))
        for call in fallback.call_args_list:
            self.assertFalse(call.kwargs["full_matrices"])
            self.assertFalse(call.kwargs["check_finite"])
            self.assertEqual(call.kwargs["lapack_driver"], "gesvd")

    def test_first_reference_has_exact_window_bounds_mean_and_diagonal_barycenter(self) -> None:
        means, covs = self._stream()
        tracker = RollingBuresReference(window_length=4)

        state = tracker.reference_at(6, means, covs)
        expected_cov = self._closed_form_diagonal_barycenter(covs[3:7])

        self.assertEqual(state.origin, 6)
        self.assertEqual(state.window_start, 3)
        self.assertEqual(state.window_stop, 7)
        np.testing.assert_allclose(state.mean, means[3:7].mean(axis=0), rtol=0.0, atol=0.0)
        np.testing.assert_allclose(state.cov, expected_cov, rtol=1e-10, atol=1e-10)
        self.assertTrue(state.refreshed)
        self.assertFalse(state.fallback)

    def test_exact_solver_converges_for_condition_1e8_noncommuting_covariances(self) -> None:
        dimension = 30
        window_length = 4
        spectrum = np.geomspace(1e-8, 1.0, dimension)
        rng = np.random.default_rng(9127)
        covs = []
        for _ in range(window_length):
            basis, _ = np.linalg.qr(rng.normal(size=(dimension, dimension)))
            covariance = (basis * spectrum) @ basis.T
            covs.append(0.5 * (covariance + covariance.T))
        covs = np.array(covs)
        means = rng.normal(size=(window_length, dimension))

        state = RollingBuresReference(window_length).reference_at(
            window_length - 1,
            means,
            covs,
        )

        self.assertTrue(np.isfinite(state.cov).all())
        self.assertGreater(np.linalg.eigvalsh(state.cov).min(), 0.0)
        self.assertTrue(np.isfinite(state.residual))
        self.assertLessEqual(state.residual, 1e-9)

    def test_exact_solver_is_robust_across_large_window_condition_1e8_seeds(self) -> None:
        dimension = 30
        window_length = 48
        spectrum = np.geomspace(1e-8, 1.0, dimension)
        for seed in (47, 9127, 8675309):
            rng = np.random.default_rng(seed)
            covs = []
            for _ in range(window_length):
                basis, _ = np.linalg.qr(rng.normal(size=(dimension, dimension)))
                covariance = (basis * spectrum) @ basis.T
                covs.append(0.5 * (covariance + covariance.T))

            with self.subTest(seed=seed):
                state = RollingBuresReference(window_length).reference_at(
                    window_length - 1,
                    rng.normal(size=(window_length, dimension)),
                    np.array(covs),
                )
                self.assertTrue(np.isfinite(state.cov).all())
                self.assertGreater(np.linalg.eigvalsh(state.cov).min(), 0.0)
                self.assertTrue(np.isfinite(state.residual))
                self.assertLessEqual(state.residual, 1e-9)


    def test_anderson_safeguard_rejects_candidate_with_worse_residual(self) -> None:
        self.assertTrue(
            hasattr(local_reference, "_select_safeguarded_candidate"),
            "missing safeguarded Anderson candidate selector",
        )
        ordinary = np.eye(2)
        accelerated = 2.0 * np.eye(2)
        prepared = local_reference._PreparedCovariances(
            projected=np.array([np.eye(2)]),
            roots=np.array([np.eye(2)]),
        )

        def checked_residual(candidate, _prepared):
            if np.array_equal(candidate, ordinary):
                return 0.1, candidate
            return 0.2, candidate

        with mock.patch.object(
            local_reference,
            "_prepared_residual",
            side_effect=checked_residual,
        ) as checked:
            selected, residual, accepted = local_reference._select_safeguarded_candidate(
                ordinary,
                accelerated,
                prepared,
            )

        np.testing.assert_array_equal(selected, ordinary)
        self.assertEqual(residual, 0.1)
        self.assertFalse(accepted)
        self.assertEqual(checked.call_count, 2)

    def test_exact_solver_recovers_from_later_accelerated_candidate_failure(self) -> None:
        self.assertTrue(
            hasattr(local_reference, "_select_safeguarded_candidate"),
            "missing safeguarded Anderson candidate selector",
        )
        prepared = local_reference._PreparedCovariances(
            projected=np.array([[[1.0]]]),
            roots=np.array([[[1.0]]]),
        )

        def synthetic_residual(current, _prepared):
            value = float(current[0, 0])
            if np.isclose(value, 9.0):
                raise local_reference._BuresNumericalError(
                    "forced later accelerated failure"
                )
            if np.isclose(value, 1.0):
                return 1.0, np.array([[2.0]])
            if np.isclose(value, 2.0):
                return 1.0, np.array([[3.0]])
            if np.isclose(value, 3.0):
                return 0.0, np.array([[3.0]])
            raise AssertionError(f"unexpected synthetic iterate {value}")

        with (
            mock.patch.object(
                local_reference,
                "_prepared_residual",
                side_effect=synthetic_residual,
            ),
            mock.patch.object(
                local_reference,
                "_select_safeguarded_candidate",
                return_value=(np.array([[9.0]]), 0.5, True),
            ) as safeguard,
        ):
            covariance, residual = local_reference._solve_exact_barycenter(
                prepared,
                max_iter=5,
                tolerance=1e-9,
            )

        np.testing.assert_array_equal(covariance, [[3.0]])
        self.assertEqual(residual, 0.0)
        safeguard.assert_called_once()

    def test_future_observations_do_not_change_current_reference(self) -> None:
        means, covs = self._stream()
        changed_means = means.copy()
        changed_covs = covs.copy()
        changed_means[7:] += 1e6
        changed_covs[7:] *= 1e5

        baseline = RollingBuresReference(4).reference_at(6, means, covs)
        changed = RollingBuresReference(4).reference_at(6, changed_means, changed_covs)

        np.testing.assert_array_equal(changed.mean, baseline.mean)
        np.testing.assert_array_equal(changed.cov, baseline.cov)
        self.assertEqual(changed.residual, baseline.residual)

    def test_future_nonfinite_values_are_ignored_until_they_enter_active_window(self) -> None:
        means, covs = self._stream(7)
        baseline = RollingBuresReference(3).reference_at(2, means, covs)

        for target in ("means", "covs"):
            for value in (np.nan, np.inf, -np.inf):
                bad_means = means.copy()
                bad_covs = covs.copy()
                if target == "means":
                    bad_means[3, 0] = value
                else:
                    bad_covs[3, 0, 0] = value
                tracker = RollingBuresReference(3)
                with self.subTest(target=target, value=value, phase="future"):
                    current = tracker.reference_at(2, bad_means, bad_covs)
                    np.testing.assert_array_equal(current.mean, baseline.mean)
                    np.testing.assert_array_equal(current.cov, baseline.cov)
                with self.subTest(target=target, value=value, phase="active"):
                    with self.assertRaisesRegex(ValueError, "active window.*finite"):
                        tracker.reference_at(3, bad_means, bad_covs)

    def test_nonfinite_expired_values_do_not_change_first_reference(self) -> None:
        means, covs = self._stream(7)
        baseline = RollingBuresReference(3).reference_at(4, means, covs)
        changed_means = means.copy()
        changed_covs = covs.copy()
        changed_means[0, 0] = np.nan
        changed_covs[0, 0, 0] = np.inf

        changed = RollingBuresReference(3).reference_at(4, changed_means, changed_covs)

        np.testing.assert_array_equal(changed.mean, baseline.mean)
        np.testing.assert_array_equal(changed.cov, baseline.cov)

    def test_source_origin_changes_reference_but_later_observations_do_not(self) -> None:
        means, covs = self._stream()
        origin = 6
        source_means = means.copy()
        source_covs = covs.copy()
        future_means = means.copy()
        future_covs = covs.copy()
        source_means[origin] += np.array([50.0, -75.0, 125.0])
        source_covs[origin] *= 9.0
        future_means[origin + 1 :] += 1e6
        future_covs[origin + 1 :] *= 1e5

        baseline = RollingBuresReference(4).reference_at(origin, means, covs)
        source_changed = RollingBuresReference(4).reference_at(
            origin,
            source_means,
            source_covs,
        )
        future_changed = RollingBuresReference(4).reference_at(
            origin,
            future_means,
            future_covs,
        )

        self.assertFalse(np.allclose(source_changed.mean, baseline.mean))
        self.assertFalse(np.allclose(source_changed.cov, baseline.cov))
        np.testing.assert_array_equal(future_changed.mean, baseline.mean)
        np.testing.assert_array_equal(future_changed.cov, baseline.cov)

    def test_sequential_update_rejects_revised_append_only_history(self) -> None:
        means, covs = self._stream(7)
        for target in ("means", "covs"):
            tracker = RollingBuresReference(3)
            tracker.reference_at(2, means, covs)
            revised_means = means.copy()
            revised_covs = covs.copy()
            if target == "means":
                revised_means[1, 0] += 10.0
            else:
                revised_covs[1, 0, 0] += 10.0

            with self.subTest(target=target):
                with self.assertRaisesRegex(ValueError, "append-only history"):
                    tracker.reference_at(3, revised_means, revised_covs)

    def test_nonsequential_updates_are_rejected(self) -> None:
        means, covs = self._stream()
        tracker = RollingBuresReference(3)
        tracker.reference_at(5, means, covs)

        for origin in (5, 7, 3):
            with self.subTest(origin=origin):
                with self.assertRaisesRegex(ValueError, "sequential"):
                    tracker.reference_at(origin, means, covs)

    def test_incremental_mean_matches_direct_window_mean(self) -> None:
        means, covs = self._stream()
        tracker = RollingBuresReference(4, refresh_period=3, residual_threshold=1e6)

        for origin in range(3, 11):
            state = tracker.reference_at(origin, means, covs)
            with self.subTest(origin=origin):
                np.testing.assert_allclose(
                    state.mean,
                    means[origin - 3 : origin + 1].mean(axis=0),
                    rtol=1e-14,
                    atol=1e-14,
                )

    def test_scheduled_refresh_and_exact_number_of_warm_steps(self) -> None:
        means, covs = self._stream()
        tracker = RollingBuresReference(
            3,
            k_ref=2,
            refresh_period=3,
            residual_threshold=1e6,
        )

        initial = tracker.reference_at(5, means, covs)
        scheduled = tracker.reference_at(6, means, covs)
        warmed = tracker.reference_at(7, means, covs)

        expected_scheduled = project_spd(
            self._closed_form_diagonal_barycenter(covs[4:7]),
            eps=1e-8,
        )
        expected_warm = expected_scheduled
        for _ in range(2):
            expected_warm = bures_fixed_point_step(expected_warm, covs[5:8])
        expected_warm = project_spd(expected_warm, eps=1e-8)

        self.assertTrue(initial.refreshed)
        self.assertFalse(initial.fallback)
        self.assertTrue(scheduled.refreshed)
        self.assertFalse(scheduled.fallback)
        np.testing.assert_allclose(scheduled.cov, expected_scheduled, rtol=1e-12, atol=1e-12)
        self.assertFalse(warmed.refreshed)
        self.assertFalse(warmed.fallback)
        np.testing.assert_allclose(warmed.cov, expected_warm, rtol=1e-12, atol=1e-12)

    def test_active_covariance_window_is_projected_once_across_warm_steps(self) -> None:
        means, covs = self._stream(8)
        tracker = RollingBuresReference(
            3,
            k_ref=4,
            refresh_period=100,
            residual_threshold=1e6,
        )
        tracker.reference_at(2, means, covs)
        original_project = local_reference.project_spd

        with mock.patch.object(
            local_reference,
            "project_spd",
            wraps=original_project,
        ) as projected:
            state = tracker.reference_at(3, means, covs)

        self.assertFalse(state.refreshed)
        self.assertFalse(state.fallback)
        self.assertEqual(projected.call_count, 3 + 4 + 1)

    def test_tiny_residual_threshold_forces_unscheduled_exact_fallback(self) -> None:
        means, covs = self._stream()
        tracker = RollingBuresReference(
            3,
            k_ref=1,
            refresh_period=100,
            residual_threshold=1e-20,
        )
        tracker.reference_at(3, means, covs)

        state = tracker.reference_at(4, means, covs)
        expected = project_spd(
            self._closed_form_diagonal_barycenter(covs[2:5]),
            eps=1e-8,
        )

        self.assertFalse(state.refreshed)
        self.assertTrue(state.fallback)
        np.testing.assert_allclose(state.cov, expected, rtol=1e-12, atol=1e-12)

    def test_reference_state_is_frozen_owned_read_only_and_defensive(self) -> None:
        mean = np.array([1.0, 2.0])
        cov = np.diag([3.0, 4.0])
        state = ReferenceState(5, mean, cov, 2, 5, 0.01, True, False)
        mean.fill(99.0)
        cov.fill(99.0)

        self.assertTrue(dataclasses.is_dataclass(state))
        self.assertEqual(
            tuple(field.name for field in dataclasses.fields(state)),
            (
                "origin",
                "mean",
                "cov",
                "window_start",
                "window_stop",
                "residual",
                "refreshed",
                "fallback",
            ),
        )
        np.testing.assert_array_equal(state.mean, [1.0, 2.0])
        np.testing.assert_array_equal(state.cov, np.diag([3.0, 4.0]))
        self.assertFalse(state.mean.flags.writeable)
        self.assertFalse(state.cov.flags.writeable)
        self.assertFalse(np.shares_memory(state.mean, mean))
        self.assertFalse(np.shares_memory(state.cov, cov))
        with self.assertRaises(ValueError):
            state.mean[0] = 0.0
        with self.assertRaises(ValueError):
            state.cov[0, 0] = 0.0
        with self.assertRaises(ValueError):
            state.mean.setflags(write=True)
        with self.assertRaises(ValueError):
            state.cov.setflags(write=True)
        with self.assertRaises(dataclasses.FrozenInstanceError):
            state.origin = 6

    def test_tracker_result_is_defensive_against_source_mutation(self) -> None:
        means, covs = self._stream()
        tracker = RollingBuresReference(3)
        state = tracker.reference_at(5, means, covs)
        expected_mean = state.mean.copy()
        expected_cov = state.cov.copy()

        means[3:6] = -1e9
        covs[3:6] = np.eye(3) * 1e9

        np.testing.assert_array_equal(state.mean, expected_mean)
        np.testing.assert_array_equal(state.cov, expected_cov)

    def test_fixed_point_step_and_residual_follow_the_required_formula(self) -> None:
        current = np.diag([4.0, 9.0])
        covs = np.array([np.diag([1.0, 16.0]), np.diag([9.0, 4.0])])

        step = bures_fixed_point_step(current, covs)
        expected = np.diag([4.0, 9.0])
        expected_residual = np.linalg.norm(expected - current, "fro") / np.linalg.norm(current, "fro")

        np.testing.assert_allclose(step, expected, rtol=1e-12, atol=1e-12)
        self.assertAlmostEqual(fixed_point_residual(current, covs), expected_residual)

    def test_fixed_point_step_projects_inputs_to_the_required_floor(self) -> None:
        step = bures_fixed_point_step(
            np.diag([0.0, 1.0]),
            np.array([np.diag([-2.0, 3.0]), np.diag([0.0, 5.0])]),
        )

        self.assertTrue(np.isfinite(step).all())
        self.assertGreaterEqual(np.linalg.eigvalsh(step).min(), 1e-8 * (1.0 - 1e-10))

    def test_constructor_rejects_invalid_values_and_accepts_numpy_scalars(self) -> None:
        invalid_integer_values = (
            True,
            False,
            1.0,
            3.0,
            1.0 + 0.0j,
            "3",
            np.array(3),
            np.array([3]),
        )
        for parameter in ("window_length", "k_ref", "refresh_period"):
            minimum = 3 if parameter == "window_length" else 1
            for value in (*invalid_integer_values, minimum - 1, -1):
                kwargs = {"window_length": 3, "k_ref": 1, "refresh_period": 24}
                kwargs[parameter] = value
                with self.subTest(parameter=parameter, value=repr(value)):
                    with self.assertRaises(ValueError):
                        RollingBuresReference(**kwargs)

        for threshold in (
            -1.0,
            np.nan,
            np.inf,
            -np.inf,
            True,
            False,
            1.0 + 0.0j,
            "0.1",
            np.array(0.1),
            np.array([0.1]),
        ):
            with self.subTest(threshold=repr(threshold)):
                with self.assertRaises(ValueError):
                    RollingBuresReference(3, residual_threshold=threshold)

        RollingBuresReference(
            np.int64(3),
            k_ref=np.int64(1),
            refresh_period=np.int64(2),
            residual_threshold=np.float64(0.0),
        )

    def test_reference_at_rejects_malformed_inputs(self) -> None:
        means, covs = self._stream(7)
        malformed_cases = (
            ("origin_bool", True, means, covs),
            ("origin_float", 4.0, means, covs),
            ("incomplete_window", 1, means, covs),
            ("origin_at_data_length", 7, means, covs),
            ("origin_after_data", 8, means, covs),
            ("means_not_2d", 4, means[:, 0], covs),
            ("covs_not_3d", 4, means, covs[0]),
            ("covs_not_square", 4, means[:, :2], np.ones((7, 2, 3))),
            ("different_lengths", 4, means[:-1], covs),
            ("different_dimensions", 4, means[:, :2], covs),
            ("complex_means", 4, means.astype(complex), covs),
            ("complex_covs", 4, means, covs.astype(complex)),
        )
        for name, origin, bad_means, bad_covs in malformed_cases:
            with self.subTest(case=name):
                with self.assertRaises(ValueError):
                    RollingBuresReference(3).reference_at(origin, bad_means, bad_covs)

        for target in ("means", "covs"):
            for value in (np.nan, np.inf, -np.inf):
                bad_means = means.copy()
                bad_covs = covs.copy()
                if target == "means":
                    bad_means[4, 0] = value
                else:
                    bad_covs[4, 0, 0] = value
                with self.subTest(target=target, value=value):
                    with self.assertRaisesRegex(ValueError, "active window.*finite"):
                        RollingBuresReference(3).reference_at(4, bad_means, bad_covs)

    def test_zero_dimension_is_rejected_explicitly(self) -> None:
        with self.assertRaisesRegex(ValueError, "positive dimension"):
            bures_fixed_point_step(
                np.empty((0, 0)),
                np.empty((2, 0, 0)),
            )
        with self.assertRaisesRegex(ValueError, "positive dimension"):
            fixed_point_residual(
                np.empty((0, 0)),
                np.empty((2, 0, 0)),
            )
        with self.assertRaisesRegex(ValueError, "positive dimension"):
            RollingBuresReference(3).reference_at(
                2,
                np.empty((5, 0)),
                np.empty((5, 0, 0)),
            )

    def test_reference_at_accepts_first_complete_origin_and_rejects_data_length(self) -> None:
        means, covs = self._stream(7)

        first = RollingBuresReference(3).reference_at(2, means, covs)

        self.assertEqual(first.window_start, 0)
        self.assertEqual(first.window_stop, 3)
        np.testing.assert_array_equal(first.mean, means[:3].mean(axis=0))
        with self.assertRaisesRegex(ValueError, "complete reference window"):
            RollingBuresReference(3).reference_at(len(means), means, covs)

    def test_fixed_point_functions_reject_malformed_inputs(self) -> None:
        valid_current = np.eye(2)
        valid_covs = np.array([np.eye(2), np.eye(2)])
        cases = (
            (np.ones(2), valid_covs),
            (np.ones((2, 3)), valid_covs),
            (valid_current, np.eye(2)),
            (valid_current, np.ones((2, 2, 3))),
            (valid_current, np.ones((2, 3, 3))),
            (valid_current.astype(complex), valid_covs),
            (valid_current, valid_covs.astype(complex)),
            (np.array([[np.nan, 0.0], [0.0, 1.0]]), valid_covs),
            (valid_current, np.array([[[np.inf, 0.0], [0.0, 1.0]]])),
            (valid_current, np.empty((0, 2, 2))),
        )
        for current, candidate_covs in cases:
            with self.subTest(current_shape=current.shape, covs_shape=candidate_covs.shape):
                with self.assertRaises(ValueError):
                    bures_fixed_point_step(current, candidate_covs)
                with self.assertRaises(ValueError):
                    fixed_point_residual(current, candidate_covs)

    def test_all_returned_covariances_and_residuals_are_finite_spd(self) -> None:
        means, covs = self._stream()
        covs[1, 0, 0] = -0.25
        covs[2, 1, 1] = 0.0
        tracker = RollingBuresReference(
            3,
            k_ref=2,
            refresh_period=4,
            residual_threshold=1e-10,
        )

        for origin in range(2, len(means)):
            state = tracker.reference_at(origin, means, covs)
            with self.subTest(origin=origin):
                self.assertTrue(np.isfinite(state.cov).all())
                self.assertGreater(np.linalg.eigvalsh(state.cov).min(), 0.0)
                self.assertTrue(np.isfinite(state.residual))


class Task3APIPresenceTest(unittest.TestCase):
    def test_task_3_api_exists(self) -> None:
        missing = [name for name in TASK_3_API if not hasattr(local_reference, name)]
        self.assertEqual(missing, [], f"missing Task 3 API: {missing}")


class Task3QualityAPIPresenceTest(unittest.TestCase):
    def test_local_bwar_decode_api_exists(self) -> None:
        self.assertTrue(
            hasattr(local_reference, "local_bwar_decode"),
            "missing Task 3 quality API: local_bwar_decode",
        )


@unittest.skipUnless(
    all(hasattr(local_reference, name) for name in TASK_3_API),
    "Task 3 API not implemented",
)
class LocalBWARForecastTest(unittest.TestCase):
    @staticmethod
    def _stream(length: int = 14) -> tuple[np.ndarray, np.ndarray]:
        times = np.arange(length, dtype=float)
        means = np.column_stack((0.1 * times, -0.05 * times))
        angles = 0.03 * times
        covs = []
        for time, angle in zip(times, angles):
            rotation = np.array(
                [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
            )
            spectrum = np.diag([1.0 + 0.02 * time, 1.5 + 0.01 * time])
            covs.append(rotation @ spectrum @ rotation.T)
        return means, np.asarray(covs)

    @staticmethod
    def _conditioned_covariance(*, reverse: bool, rotate: bool) -> np.ndarray:
        spectrum = np.geomspace(1e-8, 1.0, 30)
        if reverse:
            spectrum = spectrum[::-1]
        if not rotate:
            return np.diag(spectrum)
        rng = np.random.default_rng(20260713 + int(reverse))
        basis, _ = np.linalg.qr(rng.normal(size=(30, 30)))
        covariance = (basis * spectrum) @ basis.T
        return 0.5 * (covariance + covariance.T)

    def test_local_geometry_is_frozen_defensive_and_irreversibly_read_only(self) -> None:
        means, covs = self._stream()
        reference = RollingBuresReference(6).reference_at(8, means, covs)
        source = np.arange(30, dtype=float).reshape(6, 5)
        expected = source.copy()

        geometry = LocalGeometry(8, reference, source)
        source.fill(-1.0)

        self.assertTrue(dataclasses.is_dataclass(geometry))
        self.assertEqual(
            tuple(field.name for field in dataclasses.fields(geometry)),
            ("origin", "reference", "coordinates"),
        )
        np.testing.assert_array_equal(geometry.coordinates, expected)
        self.assertFalse(np.shares_memory(geometry.coordinates, source))
        self.assertFalse(geometry.coordinates.flags.writeable)
        self.assertIsInstance(geometry.coordinates.base, np.ndarray)
        self.assertIsInstance(geometry.coordinates.base.base, bytes)
        with self.assertRaises(ValueError):
            geometry.coordinates[0, 0] = 0.0
        with self.assertRaises(ValueError):
            geometry.coordinates.setflags(write=True)
        with self.assertRaises(dataclasses.FrozenInstanceError):
            geometry.origin = 9

    def test_local_geometry_rejects_incompatible_metadata_and_shapes(self) -> None:
        reference = ReferenceState(
            origin=3,
            mean=np.zeros(2),
            cov=np.eye(2),
            window_start=0,
            window_stop=4,
            residual=0.0,
            refreshed=True,
            fallback=False,
        )
        valid_coordinates = np.zeros((4, 5))
        incompatible_reference = ReferenceState(
            origin=3,
            mean=np.zeros(2),
            cov=np.eye(3),
            window_start=0,
            window_stop=4,
            residual=0.0,
            refreshed=True,
            fallback=False,
        )
        cases = (
            ("reference_type", lambda: LocalGeometry(3, object(), valid_coordinates)),
            ("origin_bool", lambda: LocalGeometry(True, reference, valid_coordinates)),
            ("origin_float", lambda: LocalGeometry(3.0, reference, valid_coordinates)),
            ("origin_mismatch", lambda: LocalGeometry(2, reference, valid_coordinates)),
            ("row_count", lambda: LocalGeometry(3, reference, valid_coordinates[:-1])),
            ("coordinate_dimension", lambda: LocalGeometry(3, reference, np.zeros((4, 4)))),
            (
                "reference_dimensions",
                lambda: LocalGeometry(3, incompatible_reference, np.zeros((4, 5))),
            ),
        )
        for name, operation in cases:
            with self.subTest(case=name):
                with self.assertRaises(ValueError):
                    operation()

        geometry = LocalGeometry(np.int64(3), reference, valid_coordinates)
        self.assertEqual(geometry.origin, 3)

    def test_stable_transport_reproduces_condition_1e8_targets(self) -> None:
        for rotate in (False, True):
            reference = self._conditioned_covariance(reverse=False, rotate=rotate)
            target = self._conditioned_covariance(reverse=True, rotate=rotate)

            transport = bures_transport_map(reference, target)
            reproduced = transport @ reference @ transport
            relative_error = np.linalg.norm(
                reproduced - target,
                "fro",
            ) / np.linalg.norm(target, "fro")

            with self.subTest(rotate=rotate, relative_error=relative_error):
                self.assertTrue(np.isfinite(transport).all())
                self.assertGreater(np.linalg.eigvalsh(transport).min(), 0.0)
                self.assertLessEqual(relative_error, 1e-9)

    def test_stable_transport_rejects_spd_map_with_bad_congruence(self) -> None:
        identity = np.eye(2)
        invalid_svd = (identity, np.full(2, 2.0), identity)

        with mock.patch.object(
            local_reference,
            "_robust_svd",
            return_value=invalid_svd,
        ):
            with self.assertRaisesRegex(
                local_reference._BuresNumericalError,
                "congruence residual",
            ):
                bures_transport_map(identity, identity)

    def test_stable_transport_uses_robust_svd_fallback(self) -> None:
        reference = self._conditioned_covariance(reverse=False, rotate=True)
        target = self._conditioned_covariance(reverse=True, rotate=True)
        original_fallback = scipy_linalg.svd

        with (
            mock.patch.object(
                local_reference.np.linalg,
                "svd",
                side_effect=np.linalg.LinAlgError("forced numpy SVD failure"),
            ),
            mock.patch.object(
                scipy_linalg,
                "svd",
                wraps=original_fallback,
            ) as fallback,
        ):
            transport = bures_transport_map(reference, target)

        self.assertTrue(np.isfinite(transport).all())
        fallback.assert_called_once()
        self.assertFalse(fallback.call_args.kwargs["full_matrices"])
        self.assertFalse(fallback.call_args.kwargs["check_finite"])
        self.assertEqual(fallback.call_args.kwargs["lapack_driver"], "gesvd")

    def test_encode_reference_against_itself_has_zero_covariance_coordinates(self) -> None:
        covariance = self._conditioned_covariance(reverse=False, rotate=True)
        mean = np.linspace(-1.0, 1.0, 30)

        coordinate = local_bwar_encode(mean, covariance, mean, covariance)

        self.assertEqual(coordinate.shape, (30 + 30 * 31 // 2,))
        np.testing.assert_array_equal(coordinate[:30], np.zeros(30))
        np.testing.assert_allclose(coordinate[30:], 0.0, rtol=0.0, atol=1e-8)

    def test_all_window_coordinates_use_the_origin_reference(self) -> None:
        means, covs = self._stream()

        geometry = build_local_bwar_geometry(
            means,
            covs,
            window_length=6,
            source_indices=[8],
        )[8]
        direct = np.vstack(
            [
                local_bwar_encode(
                    means[index],
                    covs[index],
                    geometry.reference.mean,
                    geometry.reference.cov,
                )
                for index in range(3, 9)
            ]
        )

        self.assertEqual(geometry.origin, 8)
        self.assertEqual(geometry.reference.origin, 8)
        self.assertEqual(geometry.coordinates.shape, (6, 5))
        np.testing.assert_array_equal(geometry.coordinates, direct)

    def test_first_requested_origin_can_be_later_than_first_complete_window(self) -> None:
        means, covs = self._stream()

        geometries = build_local_bwar_geometry(
            means,
            covs,
            window_length=4,
            source_indices=np.array([7, 8, 9]),
            refresh_period=100,
            residual_threshold=1e6,
        )

        self.assertEqual(tuple(geometries), (7, 8, 9))
        self.assertTrue(geometries[7].reference.refreshed)
        self.assertFalse(geometries[8].reference.refreshed)
        self.assertFalse(geometries[9].reference.refreshed)
        for origin, geometry in geometries.items():
            self.assertEqual(geometry.reference.window_start, origin - 3)
            self.assertEqual(geometry.reference.window_stop, origin + 1)

    def test_future_data_cannot_change_geometry_or_h_step_forecast(self) -> None:
        means, covs = self._stream()
        changed_means = means.copy()
        changed_covs = covs.copy()
        changed_means[9:] += 500.0
        changed_covs[9:] *= 50.0

        baseline = build_local_bwar_geometry(means, covs, window_length=6, source_indices=[8])[8]
        changed = build_local_bwar_geometry(
            changed_means,
            changed_covs,
            window_length=6,
            source_indices=[8],
        )[8]
        baseline_forecast = forecast_local_bwar(baseline, ridge=0.1, horizon=3)
        changed_forecast = forecast_local_bwar(changed, ridge=0.1, horizon=3)

        np.testing.assert_array_equal(changed.reference.mean, baseline.reference.mean)
        np.testing.assert_array_equal(changed.reference.cov, baseline.reference.cov)
        np.testing.assert_array_equal(changed.coordinates, baseline.coordinates)
        np.testing.assert_array_equal(changed_forecast[0], baseline_forecast[0])
        np.testing.assert_array_equal(changed_forecast[1], baseline_forecast[1])

    def test_source_origin_data_changes_local_geometry(self) -> None:
        means, covs = self._stream()
        changed_means = means.copy()
        changed_covs = covs.copy()
        changed_means[8] += np.array([25.0, -40.0])
        changed_covs[8] *= 9.0

        baseline = build_local_bwar_geometry(means, covs, window_length=6, source_indices=[8])[8]
        changed = build_local_bwar_geometry(
            changed_means,
            changed_covs,
            window_length=6,
            source_indices=[8],
        )[8]

        self.assertFalse(np.allclose(changed.reference.mean, baseline.reference.mean))
        self.assertFalse(np.allclose(changed.reference.cov, baseline.reference.cov))
        self.assertFalse(np.allclose(changed.coordinates, baseline.coordinates))

    def test_forecast_has_exact_shapes_and_finite_spd_floor(self) -> None:
        means, covs = self._stream()
        geometry = build_local_bwar_geometry(
            means,
            covs,
            window_length=6,
            source_indices=[8],
        )[8]

        pred_mean, pred_cov = forecast_local_bwar(geometry, ridge=0.1, horizon=2)

        self.assertEqual(pred_mean.shape, (2,))
        self.assertEqual(pred_cov.shape, (2, 2))
        self.assertTrue(np.isfinite(pred_mean).all())
        self.assertTrue(np.isfinite(pred_cov).all())
        self.assertGreaterEqual(
            np.linalg.eigvalsh(pred_cov).min(),
            1e-8 * (1.0 - 1e-8),
        )

    def test_build_rejects_malformed_source_indices(self) -> None:
        means, covs = self._stream()
        malformed = (
            8,
            np.array([[8]]),
            [True],
            np.array([8, True], dtype=object),
            np.array([8.0]),
            np.array([np.nan]),
            np.array([np.inf]),
            [8, 8],
            [8, 10],
            [9, 8],
            [4],
            [len(means)],
        )
        for source_indices in malformed:
            with self.subTest(source_indices=repr(source_indices)):
                with self.assertRaises(ValueError):
                    build_local_bwar_geometry(
                        means,
                        covs,
                        window_length=6,
                        source_indices=source_indices,
                    )

        self.assertEqual(
            build_local_bwar_geometry(
                means,
                covs,
                window_length=6,
                source_indices=[],
            ),
            {},
        )
        try:
            object_integer_sources = build_local_bwar_geometry(
                means,
                covs,
                window_length=6,
                source_indices=np.array([8], dtype=object),
            )
        except ValueError as exc:
            self.fail(f"object array of exact integers was rejected: {exc}")
        self.assertEqual(tuple(object_integer_sources), (8,))

    def test_encode_and_build_reject_malformed_arrays(self) -> None:
        means, covs = self._stream()
        valid_encode = (means[0], covs[0], means[1], covs[1])
        malformed_encodes = (
            (means[0, 0], covs[0], means[1], covs[1]),
            (means[0], covs[0, 0], means[1], covs[1]),
            (means[0], covs[0], means[1, :1], covs[1]),
            (means[0], np.ones((2, 3)), means[1], covs[1]),
            (means[0].astype(complex), covs[0], means[1], covs[1]),
        )
        for arguments in malformed_encodes:
            with self.subTest(encode_shapes=tuple(np.shape(value) for value in arguments)):
                with self.assertRaises(ValueError):
                    local_bwar_encode(*arguments)
        for position in range(4):
            arguments = [np.array(value, copy=True) for value in valid_encode]
            arguments[position].flat[0] = np.nan
            with self.subTest(nonfinite_encode_position=position):
                with self.assertRaises(ValueError):
                    local_bwar_encode(*arguments)

        malformed_series = (
            (means[:, 0], covs),
            (means, covs[0]),
            (means, np.ones((len(means), 2, 3))),
            (means[:-1], covs),
            (means[:, :1], covs),
            (means.astype(complex), covs),
            (means, covs.astype(complex)),
        )
        for bad_means, bad_covs in malformed_series:
            with self.subTest(means_shape=bad_means.shape, covs_shape=bad_covs.shape):
                with self.assertRaises(ValueError):
                    build_local_bwar_geometry(
                        bad_means,
                        bad_covs,
                        window_length=6,
                        source_indices=[8],
                    )
        for target in ("means", "covs"):
            bad_means = means.copy()
            bad_covs = covs.copy()
            if target == "means":
                bad_means[8, 0] = np.inf
            else:
                bad_covs[8, 0, 0] = np.inf
            with self.subTest(nonfinite_series=target):
                with self.assertRaises(ValueError):
                    build_local_bwar_geometry(
                        bad_means,
                        bad_covs,
                        window_length=6,
                        source_indices=[8],
                    )


@unittest.skipUnless(
    hasattr(local_reference, "local_bwar_decode"),
    "local_bwar_decode not implemented",
)
class LocalBWARDecodeQualityTest(unittest.TestCase):
    @staticmethod
    def _conditioned_covariance(*, reverse: bool) -> np.ndarray:
        spectrum = np.geomspace(1e-8, 1.0, 30)
        if reverse:
            spectrum = spectrum[::-1]
        rng = np.random.default_rng(20260713 + int(reverse))
        basis, _ = np.linalg.qr(rng.normal(size=(30, 30)))
        covariance = (basis * spectrum) @ basis.T
        return 0.5 * (covariance + covariance.T)

    def test_spd_transport_below_legacy_floor_round_trips_without_contraction(self) -> None:
        reference_mean = np.array([1.5, -2.0])
        mean = np.array([-0.25, 3.0])
        reference_covariance = np.diag([3.0, 5.0])
        transport = np.diag([0.01, 2.0])
        target_covariance = transport @ reference_covariance @ transport
        expected_coordinate = np.concatenate(
            (
                mean - reference_mean,
                (transport - np.eye(2))[np.triu_indices(2)],
            )
        )

        encoded = local_bwar_encode(
            mean,
            target_covariance,
            reference_mean,
            reference_covariance,
        )
        decoded_mean, decoded_covariance = local_bwar_decode(
            expected_coordinate,
            reference_mean,
            reference_covariance,
        )
        relative_error = np.linalg.norm(
            decoded_covariance - target_covariance,
            "fro",
        ) / np.linalg.norm(target_covariance, "fro")
        decoded_scales = np.sqrt(
            np.diag(decoded_covariance) / np.diag(reference_covariance)
        )

        np.testing.assert_allclose(encoded, expected_coordinate, rtol=0.0, atol=1e-13)
        np.testing.assert_array_equal(decoded_mean, mean)
        np.testing.assert_allclose(decoded_scales, np.diag(transport), rtol=1e-13, atol=1e-15)
        self.assertLessEqual(relative_error, 1e-14)

    def test_decode_floor_check_allows_eigendecomposition_roundoff(self) -> None:
        dimension = 30
        basis, _ = np.linalg.qr(
            np.random.default_rng(0).normal(size=(dimension, dimension))
        )
        reference_covariance = (
            basis * np.geomspace(1e-8, 9.339386500557811, dimension)
        ) @ basis.T
        coordinate = np.zeros(dimension + dimension * (dimension + 1) // 2)

        decoded_mean, decoded_covariance = local_bwar_decode(
            coordinate,
            np.zeros(dimension),
            reference_covariance,
        )

        np.testing.assert_array_equal(decoded_mean, np.zeros(dimension))
        floor_ratio = np.linalg.eigvalsh(decoded_covariance).min() / 1e-8
        self.assertGreater(floor_ratio, 1.0 - 1e-6)
        self.assertGreater(np.linalg.eigvalsh(decoded_covariance).min(), 0.0)

    def test_forecast_matches_independent_recursive_coordinate_and_local_decode(self) -> None:
        reference = ReferenceState(
            origin=3,
            mean=np.array([1.0, -1.0]),
            cov=np.diag([3.0, 5.0]),
            window_start=0,
            window_stop=4,
            residual=0.0,
            refreshed=True,
            fallback=False,
        )
        coordinate = np.array([0.2, -0.1, -0.99, 0.0, 1.0])
        geometry = LocalGeometry(3, reference, np.repeat(coordinate[None, :], 4, axis=0))
        ridge = 0.1
        horizon = 3
        model = fit_dual_ridge_ar1(geometry.coordinates, ridge=ridge)
        recursive_coordinate = geometry.coordinates[-1]
        for _ in range(horizon):
            recursive_coordinate = model.predict(recursive_coordinate)
        expected = local_bwar_decode(
            recursive_coordinate,
            geometry.reference.mean,
            geometry.reference.cov,
        )

        with mock.patch.object(
            local_reference,
            "local_bwar_decode",
            wraps=local_bwar_decode,
        ) as decoder:
            actual = forecast_local_bwar(geometry, ridge=ridge, horizon=horizon)

        decoder.assert_called_once()
        self.assertIs(decoder.call_args.args[1], geometry.reference.mean)
        self.assertIs(decoder.call_args.args[2], geometry.reference.cov)
        np.testing.assert_array_equal(actual[0], expected[0])
        np.testing.assert_array_equal(actual[1], expected[1])

    def test_condition_1e8_d30_q495_encode_decode_round_trip(self) -> None:
        reference_covariance = self._conditioned_covariance(reverse=False)
        target_covariance = self._conditioned_covariance(reverse=True)
        reference_mean = np.linspace(-0.5, 0.5, 30)
        mean = np.linspace(1.0, 2.0, 30)

        coordinate = local_bwar_encode(
            mean,
            target_covariance,
            reference_mean,
            reference_covariance,
        )
        decoded_mean, decoded_covariance = local_bwar_decode(
            coordinate,
            reference_mean,
            reference_covariance,
        )
        relative_error = np.linalg.norm(
            decoded_covariance - target_covariance,
            "fro",
        ) / np.linalg.norm(target_covariance, "fro")

        self.assertEqual(coordinate.shape, (495,))
        np.testing.assert_allclose(decoded_mean, mean, rtol=0.0, atol=1e-15)
        self.assertLessEqual(relative_error, 1e-8)

    def test_decode_rejects_nonreal_nonfinite_and_inexact_dimensions(self) -> None:
        coordinate = np.zeros(5)
        reference_mean = np.zeros(2)
        reference_covariance = np.eye(2)
        malformed = (
            ("z_not_1d", np.zeros((1, 5)), reference_mean, reference_covariance),
            ("z_wrong_length", np.zeros(4), reference_mean, reference_covariance),
            ("z_complex", coordinate.astype(complex), reference_mean, reference_covariance),
            ("z_bool", coordinate.astype(bool), reference_mean, reference_covariance),
            ("z_string", coordinate.astype(str), reference_mean, reference_covariance),
            (
                "z_datetime",
                coordinate.astype("datetime64[D]"),
                reference_mean,
                reference_covariance,
            ),
            ("mean_not_1d", coordinate, np.zeros((1, 2)), reference_covariance),
            ("mean_complex", coordinate, reference_mean.astype(complex), reference_covariance),
            ("cov_wrong_shape", coordinate, reference_mean, np.eye(3)),
            ("cov_complex", coordinate, reference_mean, reference_covariance.astype(complex)),
        )
        for name, z, ref_mean, ref_cov in malformed:
            with self.subTest(case=name):
                with self.assertRaises(ValueError):
                    local_bwar_decode(z, ref_mean, ref_cov)

        for position in ("z", "mean", "cov"):
            z = coordinate.copy()
            ref_mean = reference_mean.copy()
            ref_cov = reference_covariance.copy()
            if position == "z":
                z[0] = np.nan
            elif position == "mean":
                ref_mean[0] = np.inf
            else:
                ref_cov[0, 0] = -np.inf
            with self.subTest(nonfinite=position):
                with self.assertRaises(ValueError):
                    local_bwar_decode(z, ref_mean, ref_cov)


if __name__ == "__main__":
    unittest.main()
