from typing import Any

import numpy as np
import pytest
from numpy.typing import ArrayLike, NDArray
from pydantic import ValidationError
from ropt.config import EnOptConfig
from ropt.enums import EventType
from ropt.plan import BasicOptimizer, Event
from ropt.plugins import PluginManager
from ropt.results import FunctionResults
from ropt.transforms import OptModelTransforms
from ropt.transforms.base import NonLinearConstraintTransform, ObjectiveTransform

initial_values = [0.2, 0.0, 0.1]


@pytest.fixture(name="enopt_config")
def enopt_config_fixture() -> dict[str, Any]:
    return {
        "variables": {
            "variable_count": len(initial_values),
            "lower_bounds": [-1.0, -1.0, -1.0],
            "upper_bounds": [1.0, 1.0, 1.0],
        },
        "optimizer": {
            "method": "nomad/default",
            "max_iterations": 7,
        },
        "objectives": {
            "weights": [0.75, 0.25],
        },
    }


def test_nomad_invalid_options(enopt_config: Any) -> None:
    enopt_config["optimizer"]["method"] = "mads"
    enopt_config["optimizer"]["options"] = [
        "BB_MAX_BLOCK_SIZE 10",
        "BB_OUTPUT_TYPE OBJ",
    ]
    PluginManager().get_plugin("optimizer", "mads").validate_options(
        "mads", enopt_config["optimizer"]["options"]
    )

    enopt_config["optimizer"]["options"] = [
        "BB_MAX_BLOCK_SIZE 10",
        "BB_OUTPUT_TYPE FOO",
    ]
    with pytest.raises(
        ValueError, match=r"Option Error: First argument of BB_OUTPUT_TYPE must be OBJ"
    ):
        PluginManager().get_plugin("optimizer", "mads").validate_options(
            "mads", enopt_config["optimizer"]["options"]
        )

    enopt_config["optimizer"]["options"] = [
        "BB_MAX_BLOCK_SIZE 10",
        "BB_OUTPUT_TYPE OBJ FOO",
    ]
    with pytest.raises(
        ValueError, match=r"Invalid output type\(s\) in BB_OUTPUT_TYPE: {'FOO'}"
    ):
        PluginManager().get_plugin("optimizer", "mads").validate_options(
            "mads", enopt_config["optimizer"]["options"]
        )

    enopt_config["optimizer"]["options"] = [
        "DIMENSION 10",
        "FOO FOO",
    ]
    with pytest.raises(
        ValueError, match=r"Unknown or unsupported option\(s\): `DIMENSION`, `FOO`"
    ):
        PluginManager().get_plugin("optimizer", "mads").validate_options(
            "mads", enopt_config["optimizer"]["options"]
        )


@pytest.mark.parametrize("parallel", [False, True])
@pytest.mark.parametrize(
    "external", ["", pytest.param("external/", marks=pytest.mark.external)]
)
def test_nomad_bound_constraints(
    enopt_config: dict[str, Any], evaluator: Any, parallel: bool, external: str
) -> None:
    enopt_config["optimizer"]["method"] = f"{external}nomad/default"
    enopt_config["variables"]["lower_bounds"] = [0.15, -1.0, -1.0]
    enopt_config["variables"]["upper_bounds"] = [1.0, 1.0, 0.2]
    enopt_config["optimizer"]["max_iterations"] = 3
    enopt_config["optimizer"]["parallel"] = parallel
    if parallel:
        enopt_config["optimizer"]["options"] = ["BB_MAX_BLOCK_SIZE 4"]
    variables = BasicOptimizer(enopt_config, evaluator()).run(initial_values).variables
    assert variables is not None
    assert np.allclose(variables, [0.15, 0.0, 0.2], atol=0.02)


def test_nomad_bound_constraints_block_size_one(
    enopt_config: dict[str, Any], evaluator: Any
) -> None:
    enopt_config["variables"]["lower_bounds"] = [0.15, -1.0, -1.0]
    enopt_config["variables"]["upper_bounds"] = [1.0, 1.0, 0.2]
    enopt_config["optimizer"]["max_iterations"] = 3
    enopt_config["optimizer"]["parallel"] = True
    enopt_config["optimizer"]["options"] = ["BB_MAX_BLOCK_SIZE 1"]
    variables = BasicOptimizer(enopt_config, evaluator()).run(initial_values).variables
    assert variables is not None
    assert np.allclose(variables, [0.15, 0.0, 0.2], atol=0.02)


@pytest.mark.parametrize("parallel", [False, True])
@pytest.mark.parametrize(
    ("lower_bounds", "upper_bounds"), [(-np.inf, 0.4), (-0.4, np.inf)]
)
def test_nomad_ineq_nonlinear_constraints(
    enopt_config: dict[str, Any],
    lower_bounds: Any,
    upper_bounds: Any,
    evaluator: Any,
    parallel: bool,
    test_functions: Any,
) -> None:
    enopt_config["nonlinear_constraints"] = {
        "lower_bounds": lower_bounds,
        "upper_bounds": upper_bounds,
    }
    enopt_config["optimizer"]["parallel"] = parallel
    if parallel:
        enopt_config["optimizer"]["options"] = ["BB_MAX_BLOCK_SIZE 4"]

    weight = 1.0 if upper_bounds == 0.4 else -1.0
    test_functions = (
        *test_functions,
        lambda variables: weight * variables[0] + weight * variables[2],
    )
    variables = (
        BasicOptimizer(enopt_config, evaluator(test_functions))
        .run(initial_values)
        .variables
    )
    assert variables is not None
    assert np.allclose(variables, [-0.05, 0.0, 0.45], atol=0.02)


@pytest.mark.parametrize("parallel", [False, True])
def test_nomad_eq_nonlinear_constraints(
    enopt_config: dict[str, Any],
    evaluator: Any,
    parallel: bool,
    test_functions: Any,
) -> None:
    enopt_config["nonlinear_constraints"] = {
        "lower_bounds": 1.0,
        "upper_bounds": 1.0,
    }
    enopt_config["optimizer"]["parallel"] = parallel
    if parallel:
        enopt_config["optimizer"]["options"] = ["BB_MAX_BLOCK_SIZE 4"]

    test_functions = (
        *test_functions,
        lambda variables: variables[0] + variables[2],
    )
    with pytest.raises(
        ValueError,
        match="Equality constraints are not supported by NOMAD",
    ):
        BasicOptimizer(enopt_config, evaluator(test_functions)).run(initial_values)


@pytest.mark.parametrize("parallel", [False, True])
def test_nomad_ineq_nonlinear_constraints_two_sided(
    enopt_config: Any,
    parallel: bool,
    evaluator: Any,
    test_functions: Any,
) -> None:
    enopt_config["nonlinear_constraints"] = {
        "lower_bounds": [0.0],
        "upper_bounds": [0.3],
    }
    enopt_config["optimizer"]["parallel"] = parallel
    if parallel:
        enopt_config["optimizer"]["options"] = ["BB_MAX_BLOCK_SIZE 4"]
    test_functions = (
        *test_functions,
        lambda variables: variables[0] + variables[2],
    )

    variables = (
        BasicOptimizer(enopt_config, evaluator(test_functions))
        .run(initial_values)
        .variables
    )
    assert variables is not None
    assert np.allclose(variables, [-0.1, 0.0, 0.4], atol=0.02)


@pytest.mark.parametrize("parallel", [False, True])
def test_nomad_le_ge_linear_constraints(
    enopt_config: dict[str, Any], evaluator: Any, parallel: bool
) -> None:
    enopt_config["linear_constraints"] = {
        "coefficients": [[1, 0, 1]],
        "lower_bounds": [-np.inf],
        "upper_bounds": [0.4],
    }
    enopt_config["optimizer"]["parallel"] = parallel
    if parallel:
        enopt_config["optimizer"]["options"] = ["BB_MAX_BLOCK_SIZE 4"]

    variables = BasicOptimizer(enopt_config, evaluator()).run(initial_values).variables
    assert variables is not None
    assert np.allclose(variables, [-0.05, 0.0, 0.45], atol=0.02)


@pytest.mark.parametrize("parallel", [False, True])
def test_nomad_eq_linear_constraints(
    enopt_config: dict[str, Any], evaluator: Any, parallel: bool
) -> None:
    enopt_config["linear_constraints"] = {
        "coefficients": [[1, 0, 1], [0, 1, 1]],
        "lower_bounds": [1.0, 0.75],
        "upper_bounds": [1.0, 0.75],
    }
    enopt_config["optimizer"]["parallel"] = parallel
    if parallel:
        enopt_config["optimizer"]["options"] = ["BB_MAX_BLOCK_SIZE 4"]

    with pytest.raises(
        ValueError, match="Equality constraints are not supported by NOMAD"
    ):
        BasicOptimizer(enopt_config, evaluator()).run(initial_values)


@pytest.mark.parametrize("parallel", [False, True])
def test_nomad_le_ge_linear_constraints_two_sided(
    enopt_config: Any, evaluator: Any, parallel: bool
) -> None:
    enopt_config["linear_constraints"] = {
        "coefficients": [[1, 0, 1], [1, 0, 1]],
        "lower_bounds": [-np.inf, 0.0],
        "upper_bounds": [0.3, np.inf],
    }
    enopt_config["optimizer"]["parallel"] = parallel
    if parallel:
        enopt_config["optimizer"]["options"] = ["BB_MAX_BLOCK_SIZE 4"]

    variables = BasicOptimizer(enopt_config, evaluator()).run(initial_values).variables
    assert variables is not None
    assert np.allclose(variables, [-0.1, 0.0, 0.4], atol=0.02)

    enopt_config["linear_constraints"] = {
        "coefficients": [[1, 0, 1]],
        "lower_bounds": [0.0],
        "upper_bounds": [0.3],
    }

    variables = BasicOptimizer(enopt_config, evaluator()).run(initial_values).variables
    assert variables is not None
    assert np.allclose(variables, [-0.1, 0.0, 0.4], atol=0.02)


def test_nomad_dimension_keyword(enopt_config: dict[str, Any], evaluator: Any) -> None:
    enopt_config["optimizer"]["options"] = ["DIMENSION 4"]
    with pytest.raises(
        ValidationError, match=r"Unknown or unsupported option\(s\): `DIMENSION`"
    ):
        BasicOptimizer(enopt_config, evaluator()).run(initial_values)


def test_nomad_max_iterations_keyword(
    enopt_config: dict[str, Any], evaluator: Any
) -> None:
    enopt_config["optimizer"]["options"] = ["MAX_ITERATIONS 4"]
    with pytest.raises(
        ValidationError,
        match=r"Unknown or unsupported option\(s\): `MAX_ITERATIONS`",
    ):
        BasicOptimizer(enopt_config, evaluator()).run(initial_values)


@pytest.mark.parametrize(
    ("lower_bounds", "upper_bounds"), [(-np.inf, 0.4), (-0.4, np.inf)]
)
def test_nomad_bb_output_type(
    enopt_config: dict[str, Any],
    lower_bounds: Any,
    upper_bounds: Any,
    evaluator: Any,
    test_functions: Any,
) -> None:
    enopt_config["nonlinear_constraints"] = {
        "lower_bounds": lower_bounds,
        "upper_bounds": upper_bounds,
    }
    enopt_config["optimizer"]["options"] = ["BB_OUTPUT_TYPE OBJ PB"]

    weight = 1.0 if upper_bounds == 0.4 else -1.0
    test_functions = (
        *test_functions,
        lambda variables: weight * variables[0] + weight * variables[2],
    )
    variables = (
        BasicOptimizer(enopt_config, evaluator(test_functions))
        .run(initial_values)
        .variables
    )
    assert variables is not None
    assert np.allclose(variables, [-0.05, 0.0, 0.45], atol=0.02)

    enopt_config["optimizer"]["options"] = ["BB_OUTPUT_TYPE OBJ PB PB"]
    with pytest.raises(
        ValueError,
        match="Option Error: BB_OUTPUT_TYPE specifies incorrect number of outputs",
    ):
        BasicOptimizer(enopt_config, evaluator()).run(initial_values)


def test_nomad_bb_max_block_size_no_parallel(
    enopt_config: dict[str, Any], evaluator: Any
) -> None:
    enopt_config["optimizer"]["options"] = ["BB_MAX_BLOCK_SIZE 4"]
    with pytest.raises(
        ValueError,
        match="Option Error: BB_MAX_BLOCK_SIZE may only be specified",
    ):
        BasicOptimizer(enopt_config, evaluator()).run(initial_values)


def test_nomad_parallel_no_bb_max_block_size(
    enopt_config: dict[str, Any], evaluator: Any
) -> None:
    enopt_config["optimizer"]["parallel"] = True
    with pytest.raises(
        ValueError,
        match="Option Error: BB_MAX_BLOCK_SIZE must be specified",
    ):
        BasicOptimizer(enopt_config, evaluator()).run(initial_values)


@pytest.mark.parametrize("parallel", [False, True])
def test_nomad_evaluation_failure(
    enopt_config: dict[str, Any], evaluator: Any, parallel: bool, test_functions: Any
) -> None:
    enopt_config["variables"]["lower_bounds"] = [0.15, -0.5, -1.0]
    enopt_config["variables"]["upper_bounds"] = [1.0, 0.5, 0.2]
    enopt_config["optimizer"]["max_iterations"] = 4
    enopt_config["realizations"] = {"realization_min_success": 0}
    enopt_config["optimizer"]["parallel"] = parallel
    if parallel:
        enopt_config["optimizer"]["options"] = ["BB_MAX_BLOCK_SIZE 4"]

    variables1 = BasicOptimizer(enopt_config, evaluator()).run(initial_values).variables
    assert variables1 is not None
    assert np.allclose(variables1, [0.15, 0.0, 0.2], atol=0.02)

    counter = 0

    def _add_nan(x: Any) -> Any:
        nonlocal counter
        counter += 1
        if counter == 2:
            counter = 0
            return np.nan
        return test_functions[0](x)

    variables2 = (
        BasicOptimizer(enopt_config, evaluator((_add_nan, test_functions[1])))
        .run(initial_values)
        .variables
    )
    assert variables2 is not None
    assert np.allclose(variables2, [0.15, 0.0, 0.2], atol=0.02)
    assert not np.all(np.equal(variables1, variables2))


class ObjectiveScaler(ObjectiveTransform):
    def __init__(self, scales: ArrayLike) -> None:
        self._scales = np.asarray(scales, dtype=np.float64)
        self._set = True

    def set_scales(self, scales: ArrayLike) -> None:
        if self._set:
            self._scales = np.asarray(scales, dtype=np.float64)
            self._set = False

    def to_optimizer(self, objectives: NDArray[np.float64]) -> NDArray[np.float64]:
        return objectives / self._scales

    def from_optimizer(self, objectives: NDArray[np.float64]) -> NDArray[np.float64]:
        return objectives * self._scales


@pytest.mark.parametrize("parallel", [False, True])
def test_nomad_objective_with_scaler(
    enopt_config: Any, evaluator: Any, parallel: bool, test_functions: Any
) -> None:
    enopt_config["optimizer"]["parallel"] = parallel
    if parallel:
        enopt_config["optimizer"]["options"] = ["BB_MAX_BLOCK_SIZE 4"]
    results1 = BasicOptimizer(enopt_config, evaluator()).run(initial_values).results
    assert results1 is not None
    assert results1.functions is not None
    variables1 = results1.evaluations.variables
    objectives1 = results1.functions.objectives
    assert np.allclose(variables1, [0.0, 0.0, 0.5], atol=0.02)
    assert np.allclose(objectives1, [0.5, 4.5], atol=0.02)

    def function1(variables: NDArray[np.float64]) -> float:
        return float(test_functions[0](variables))

    def function2(variables: NDArray[np.float64]) -> float:
        return float(test_functions[1](variables))

    init1 = test_functions[1](initial_values)
    transforms = OptModelTransforms(
        objectives=ObjectiveScaler(np.array([init1, init1]))
    )

    checked = False

    def check_value(event: Event) -> None:
        nonlocal checked
        results = event.data.get("results", ())
        for item in results:
            if isinstance(item, FunctionResults) and not checked:
                checked = True
                assert item.functions is not None
                assert item.functions.objectives is not None
                assert np.allclose(item.functions.objectives[-1], 1.0)
                transformed = item.transform_from_optimizer(
                    event.data["config"], event.data["transforms"]
                )
                assert transformed.functions is not None
                assert transformed.functions.objectives is not None
                assert np.allclose(transformed.functions.objectives[-1], init1)

    optimizer = BasicOptimizer(
        enopt_config, evaluator([function1, function2]), transforms=transforms
    )
    optimizer._observers.append(  # noqa: SLF001
        (EventType.FINISHED_EVALUATION, check_value)
    )
    results2 = optimizer.run(initial_values).results
    assert results2 is not None
    assert np.allclose(results2.evaluations.variables, variables1, atol=0.02)
    assert results2.functions is not None
    assert np.allclose(objectives1, results2.functions.objectives, atol=0.025)


@pytest.mark.parametrize("parallel", [False, True])
def test_nomad_objective_with_lazy_scaler(
    enopt_config: Any, evaluator: Any, parallel: bool, test_functions: Any
) -> None:
    enopt_config["optimizer"]["parallel"] = parallel
    if parallel:
        enopt_config["optimizer"]["options"] = ["BB_MAX_BLOCK_SIZE 4"]
    results1 = BasicOptimizer(enopt_config, evaluator()).run(initial_values).results
    assert results1 is not None
    assert results1.functions is not None
    variables1 = results1.evaluations.variables
    objectives1 = results1.functions.objectives
    assert np.allclose(variables1, [0.0, 0.0, 0.5], atol=0.02)
    assert np.allclose(objectives1, [0.5, 4.5], atol=0.02)

    objective_transform = ObjectiveScaler(np.array([1.0, 1.0]))
    transforms = OptModelTransforms(objectives=objective_transform)

    init1 = test_functions[1](initial_values)

    def function1(variables: NDArray[np.float64]) -> float:
        objective_transform.set_scales([init1, init1])
        return float(test_functions[0](variables))

    def function2(variables: NDArray[np.float64]) -> float:
        return float(test_functions[1](variables))

    checked = False

    def check_value(event: Event) -> None:
        nonlocal checked
        results = event.data.get("results", ())
        for item in results:
            if isinstance(item, FunctionResults) and not checked:
                checked = True
                assert item.functions is not None
                assert item.functions.objectives is not None
                assert np.allclose(item.functions.objectives[-1], 1.0)
                transformed = item.transform_from_optimizer(
                    event.data["config"], event.data["transforms"]
                )
                assert transformed.functions is not None
                assert transformed.functions.objectives is not None
                assert np.allclose(transformed.functions.objectives[-1], init1)

    optimizer = BasicOptimizer(
        enopt_config, evaluator([function1, function2]), transforms=transforms
    )
    optimizer._observers.append(  # noqa: SLF001
        (EventType.FINISHED_EVALUATION, check_value)
    )
    results2 = optimizer.run(initial_values).results
    assert results2 is not None
    assert np.allclose(results2.evaluations.variables, variables1, atol=0.02)
    assert results2.functions is not None
    assert np.allclose(objectives1, results2.functions.objectives, atol=0.025)


class ConstraintScaler(NonLinearConstraintTransform):
    def __init__(self, scales: ArrayLike) -> None:
        self._scales = np.asarray(scales, dtype=np.float64)
        self._set = True

    def set_scales(self, scales: ArrayLike) -> None:
        if self._set:
            self._scales = np.asarray(scales, dtype=np.float64)
            self._set = False

    def bounds_to_optimizer(
        self, lower_bounds: NDArray[np.float64], upper_bounds: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        return lower_bounds / self._scales, upper_bounds / self._scales

    def to_optimizer(self, constraints: NDArray[np.float64]) -> NDArray[np.float64]:
        return constraints / self._scales

    def from_optimizer(self, constraints: NDArray[np.float64]) -> NDArray[np.float64]:
        return constraints * self._scales

    def nonlinear_constraint_diffs_from_optimizer(
        self, lower_diffs: NDArray[np.float64], upper_diffs: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        return lower_diffs * self._scales, upper_diffs * self._scales


@pytest.mark.parametrize("parallel", [False, True])
@pytest.mark.parametrize(
    "external", ["", pytest.param("external/", marks=pytest.mark.external)]
)
def test_nomad_nonlinear_constraint_with_scaler(
    enopt_config: Any,
    evaluator: Any,
    parallel: bool,
    test_functions: Any,
    external: str,
) -> None:
    enopt_config["optimizer"]["method"] = f"{external}nomad/default"
    enopt_config["nonlinear_constraints"] = {
        "lower_bounds": 0.0,
        "upper_bounds": 0.4,
    }
    enopt_config["optimizer"]["parallel"] = parallel
    if parallel:
        enopt_config["optimizer"]["options"] = ["BB_MAX_BLOCK_SIZE 4"]

    functions = (
        *test_functions,
        lambda variables: variables[0] + variables[2],
    )

    results1 = (
        BasicOptimizer(enopt_config, evaluator(functions)).run(initial_values).results
    )
    assert results1 is not None
    assert results1.evaluations.variables[[0, 2]].sum() > 0.0 - 1e-5
    assert results1.evaluations.variables[[0, 2]].sum() < 0.4 + 1e-5

    scales = np.array(functions[-1](initial_values), ndmin=1)
    transforms = OptModelTransforms(nonlinear_constraints=ConstraintScaler(scales))
    config = EnOptConfig.model_validate(enopt_config, context=transforms)
    assert config.nonlinear_constraints is not None
    assert config.nonlinear_constraints.upper_bounds == 0.4
    assert transforms.nonlinear_constraints is not None
    bounds = transforms.nonlinear_constraints.bounds_to_optimizer(
        config.nonlinear_constraints.lower_bounds,
        config.nonlinear_constraints.upper_bounds,
    )
    assert bounds is not None
    assert bounds[1] == 0.4 / scales

    check = True

    def check_constraints(event: Event) -> None:
        nonlocal check
        results = event.data.get("results", ())
        for item in results:
            if isinstance(item, FunctionResults) and check:
                check = False
                assert item.functions is not None
                assert item.functions.constraints is not None
                assert np.allclose(item.functions.constraints, 1.0)
                transformed = item.transform_from_optimizer(
                    event.data["config"], event.data["transforms"]
                )
                assert transformed.functions is not None
                assert transformed.functions.constraints is not None
                assert np.allclose(transformed.functions.constraints, scales)

    optimizer = BasicOptimizer(
        enopt_config, evaluator(functions), transforms=transforms
    )
    optimizer._observers.append(  # noqa: SLF001
        (EventType.FINISHED_EVALUATION, check_constraints)
    )
    results2 = optimizer.run(initial_values).results
    assert results2 is not None
    assert np.allclose(
        results2.evaluations.variables, results1.evaluations.variables, atol=0.02
    )
    assert results1.functions is not None
    assert results2.functions is not None
    assert np.allclose(
        results1.functions.objectives, results2.functions.objectives, atol=0.025
    )


@pytest.mark.parametrize("parallel", [False, True])
@pytest.mark.parametrize(
    "external", ["", pytest.param("external/", marks=pytest.mark.external)]
)
def test_nomad_nonlinear_constraint_with_lazy_scaler(
    enopt_config: Any,
    evaluator: Any,
    parallel: bool,
    test_functions: Any,
    external: str,
) -> None:
    enopt_config["optimizer"]["method"] = f"{external}nomad/default"
    enopt_config["nonlinear_constraints"] = {
        "lower_bounds": 0.0,
        "upper_bounds": 0.4,
    }
    enopt_config["optimizer"]["parallel"] = parallel
    if parallel:
        enopt_config["optimizer"]["options"] = ["BB_MAX_BLOCK_SIZE 4"]

    functions = (
        *test_functions,
        lambda variables: variables[0] + variables[2],
    )

    results1 = (
        BasicOptimizer(enopt_config, evaluator(functions)).run(initial_values).results
    )
    assert results1 is not None
    assert results1.evaluations.variables[[0, 2]].sum() > 0.0 - 1e-5
    assert results1.evaluations.variables[[0, 2]].sum() < 0.4 + 1e-5

    scales = np.array(functions[-1](initial_values), ndmin=1)
    scaler = ConstraintScaler([1.0])
    transforms = OptModelTransforms(nonlinear_constraints=scaler)

    config = EnOptConfig.model_validate(enopt_config, context=transforms)
    assert config.nonlinear_constraints is not None
    assert config.nonlinear_constraints.upper_bounds == 0.4
    assert transforms.nonlinear_constraints is not None
    bounds = transforms.nonlinear_constraints.bounds_to_optimizer(
        config.nonlinear_constraints.lower_bounds,
        config.nonlinear_constraints.upper_bounds,
    )
    assert bounds is not None
    assert bounds[1] == 0.4

    def constraint_function(variables: NDArray[np.float64]) -> float:
        scaler.set_scales(scales)
        return float(variables[0] + variables[2])

    functions = (*test_functions, constraint_function)

    check = True

    def check_constraints(event: Event) -> None:
        nonlocal check
        config = event.data["config"]
        results = event.data.get("results", ())
        for item in results:
            if isinstance(item, FunctionResults) and check:
                check = False
                assert config.nonlinear_constraints is not None
                assert transforms.nonlinear_constraints is not None
                _, upper_bounds = transforms.nonlinear_constraints.bounds_to_optimizer(
                    config.nonlinear_constraints.lower_bounds,
                    config.nonlinear_constraints.upper_bounds,
                )
                assert np.allclose(upper_bounds, 0.4 / scales)
                assert item.functions is not None
                assert item.functions.constraints is not None
                assert np.allclose(item.functions.constraints, 1.0)
                transformed = item.transform_from_optimizer(
                    event.data["config"], event.data["transforms"]
                )
                assert transformed.functions is not None
                assert transformed.functions.constraints is not None
                assert np.allclose(transformed.functions.constraints, scales)

    optimizer = BasicOptimizer(
        enopt_config, evaluator(functions), transforms=transforms
    )
    optimizer._observers.append(  # noqa: SLF001
        (EventType.FINISHED_EVALUATION, check_constraints)
    )
    results2 = optimizer.run(initial_values).results
    assert results2 is not None
    assert np.allclose(
        results2.evaluations.variables, results1.evaluations.variables, atol=0.02
    )
    assert results1.functions is not None
    assert results2.functions is not None
    assert np.allclose(
        results1.functions.objectives, results2.functions.objectives, atol=0.025
    )
