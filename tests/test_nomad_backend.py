# ruff: file-ignore[float-equality-comparison]

from typing import Any

import numpy as np
import pytest
from pydantic import ValidationError
from ropt.simple import optimize
from ropt.workflow import validate_backend_options

initial_values = [0.2, 0.0, 0.1]

# ruff: file-ignore[boolean-type-hint-positional-argument]


@pytest.fixture(name="config")
def config_fixture() -> dict[str, Any]:
    return {
        "variables": {
            "variable_count": len(initial_values),
            "lower_bounds": [-1.0, -1.0, -1.0],
            "upper_bounds": [1.0, 1.0, 1.0],
        },
        "backend": {
            "method": "nomad/default",
            "max_iterations": 7,
        },
        "objectives": {
            "weights": [0.75, 0.25],
        },
    }


def test_nomad_invalid_options(config: Any) -> None:
    config["backend"]["method"] = "mads"
    config["backend"]["options"] = [
        "BB_MAX_BLOCK_SIZE 10",
        "BB_OUTPUT_TYPE OBJ",
    ]
    validate_backend_options("mads", config["backend"]["options"])

    config["backend"]["options"] = [
        "BB_MAX_BLOCK_SIZE 10",
        "BB_OUTPUT_TYPE FOO",
    ]
    with pytest.raises(
        ValueError, match=r"Option Error: First argument of BB_OUTPUT_TYPE must be OBJ"
    ):
        validate_backend_options("mads", config["backend"]["options"])

    config["backend"]["options"] = [
        "BB_MAX_BLOCK_SIZE 10",
        "BB_OUTPUT_TYPE OBJ FOO",
    ]
    with pytest.raises(
        ValueError, match=r"Invalid output type\(s\) in BB_OUTPUT_TYPE: {'FOO'}"
    ):
        validate_backend_options("mads", config["backend"]["options"])

    config["backend"]["options"] = [
        "DIMENSION 10",
        "FOO FOO",
    ]
    with pytest.raises(
        ValueError, match=r"Unknown or unsupported option\(s\): `DIMENSION`, `FOO`"
    ):
        validate_backend_options("mads", config["backend"]["options"])


@pytest.mark.parametrize("parallel", [False, True])
@pytest.mark.parametrize(
    "external", ["", pytest.param("external/", marks=pytest.mark.external)]
)
def test_nomad_bound_constraints(
    config: dict[str, Any], eval_func: Any, parallel: bool, external: str
) -> None:
    config["backend"]["method"] = f"{external}nomad/default"
    config["variables"]["lower_bounds"] = [0.15, -1.0, -1.0]
    config["variables"]["upper_bounds"] = [1.0, 1.0, 0.2]
    config["backend"]["max_iterations"] = 3
    config["backend"]["parallel"] = parallel
    if parallel:
        config["backend"]["options"] = ["BB_MAX_BLOCK_SIZE 4"]
    result = optimize(config, initial_values, eval_func())
    assert result.variables is not None
    assert np.allclose(result.variables, [0.15, 0.0, 0.2], atol=0.02)


def test_nomad_bound_constraints_block_size_one(
    config: dict[str, Any], eval_func: Any
) -> None:
    config["variables"]["lower_bounds"] = [0.15, -1.0, -1.0]
    config["variables"]["upper_bounds"] = [1.0, 1.0, 0.2]
    config["backend"]["max_iterations"] = 3
    config["backend"]["parallel"] = True
    config["backend"]["options"] = ["BB_MAX_BLOCK_SIZE 1"]
    result = optimize(config, initial_values, eval_func())
    assert result.variables is not None
    assert np.allclose(result.variables, [0.15, 0.0, 0.2], atol=0.02)


@pytest.mark.parametrize("parallel", [False, True])
@pytest.mark.parametrize(
    ("lower_bounds", "upper_bounds"), [(-np.inf, 0.4), (-0.4, np.inf)]
)
def test_nomad_ineq_nonlinear_constraints(  # ruff: ignore[too-many-positional-arguments]
    config: dict[str, Any],
    lower_bounds: Any,
    upper_bounds: Any,
    eval_func: Any,
    parallel: bool,
    test_functions: Any,
) -> None:
    config["nonlinear_constraints"] = {
        "lower_bounds": lower_bounds,
        "upper_bounds": upper_bounds,
    }
    config["backend"]["parallel"] = parallel
    if parallel:
        config["backend"]["options"] = ["BB_MAX_BLOCK_SIZE 4"]

    weight = 1.0 if upper_bounds == 0.4 else -1.0

    def constraint_function(variables: Any, _: Any) -> Any:
        return weight * float(variables[0] + variables[2])

    result = optimize(
        config, initial_values, eval_func(test_functions, [constraint_function])
    )
    assert result.variables is not None
    assert np.allclose(result.variables, [-0.05, 0.0, 0.45], atol=0.02)


@pytest.mark.parametrize("parallel", [False, True])
def test_nomad_eq_nonlinear_constraints(
    config: dict[str, Any],
    eval_func: Any,
    parallel: bool,
    test_functions: Any,
) -> None:
    config["nonlinear_constraints"] = {
        "lower_bounds": 1.0,
        "upper_bounds": 1.0,
    }
    config["backend"]["parallel"] = parallel
    if parallel:
        config["backend"]["options"] = ["BB_MAX_BLOCK_SIZE 4"]

    def constraint_function(variables: Any, _: Any) -> Any:
        return float(variables[0] + variables[2])

    with pytest.raises(
        ValueError,
        match="Equality constraints are not supported by NOMAD",
    ):
        optimize(
            config, initial_values, eval_func(test_functions, [constraint_function])
        )


@pytest.mark.parametrize("parallel", [False, True])
def test_nomad_ineq_nonlinear_constraints_two_sided(
    config: Any,
    parallel: bool,
    eval_func: Any,
    test_functions: Any,
) -> None:
    config["nonlinear_constraints"] = {
        "lower_bounds": [0.0],
        "upper_bounds": [0.3],
    }
    config["backend"]["parallel"] = parallel
    if parallel:
        config["backend"]["options"] = ["BB_MAX_BLOCK_SIZE 4"]

    def constraint_function(variables: Any, _: Any) -> Any:
        return float(variables[0] + variables[2])

    result = optimize(
        config, initial_values, eval_func(test_functions, [constraint_function])
    )
    assert result.variables is not None
    assert np.allclose(result.variables, [-0.1, 0.0, 0.4], atol=0.02)


@pytest.mark.parametrize("parallel", [False, True])
def test_nomad_le_ge_linear_constraints(
    config: dict[str, Any], eval_func: Any, parallel: bool
) -> None:
    config["linear_constraints"] = {
        "coefficients": [[1, 0, 1]],
        "lower_bounds": [-np.inf],
        "upper_bounds": [0.4],
    }
    config["backend"]["parallel"] = parallel
    if parallel:
        config["backend"]["options"] = ["BB_MAX_BLOCK_SIZE 4"]

    result = optimize(config, initial_values, eval_func())
    assert result.variables is not None
    assert np.allclose(result.variables, [-0.05, 0.0, 0.45], atol=0.02)


@pytest.mark.parametrize("parallel", [False, True])
def test_nomad_eq_linear_constraints(
    config: dict[str, Any], eval_func: Any, parallel: bool
) -> None:
    config["linear_constraints"] = {
        "coefficients": [[1, 0, 1], [0, 1, 1]],
        "lower_bounds": [1.0, 0.75],
        "upper_bounds": [1.0, 0.75],
    }
    config["backend"]["parallel"] = parallel
    if parallel:
        config["backend"]["options"] = ["BB_MAX_BLOCK_SIZE 4"]

    with pytest.raises(
        ValueError, match="Equality constraints are not supported by NOMAD"
    ):
        optimize(config, initial_values, eval_func())


@pytest.mark.parametrize("parallel", [False, True])
def test_nomad_le_ge_linear_constraints_two_sided(
    config: Any, eval_func: Any, parallel: bool
) -> None:
    config["linear_constraints"] = {
        "coefficients": [[1, 0, 1], [1, 0, 1]],
        "lower_bounds": [-np.inf, 0.0],
        "upper_bounds": [0.3, np.inf],
    }
    config["backend"]["parallel"] = parallel
    if parallel:
        config["backend"]["options"] = ["BB_MAX_BLOCK_SIZE 4"]

    result = optimize(config, initial_values, eval_func())
    assert result.variables is not None
    assert np.allclose(result.variables, [-0.1, 0.0, 0.4], atol=0.02)

    config["linear_constraints"] = {
        "coefficients": [[1, 0, 1]],
        "lower_bounds": [0.0],
        "upper_bounds": [0.3],
    }

    result = optimize(config, initial_values, eval_func())
    assert result.variables is not None
    assert np.allclose(result.variables, [-0.1, 0.0, 0.4], atol=0.02)


def test_nomad_dimension_keyword(config: dict[str, Any], eval_func: Any) -> None:
    config["backend"]["options"] = ["DIMENSION 4"]
    with pytest.raises(
        ValidationError, match=r"Unknown or unsupported option\(s\): `DIMENSION`"
    ):
        optimize(config, initial_values, eval_func())


def test_nomad_max_iterations_keyword(config: dict[str, Any], eval_func: Any) -> None:
    config["backend"]["options"] = ["MAX_ITERATIONS 4"]
    with pytest.raises(
        ValidationError,
        match=r"Unknown or unsupported option\(s\): `MAX_ITERATIONS`",
    ):
        optimize(config, initial_values, eval_func())


@pytest.mark.parametrize(
    ("lower_bounds", "upper_bounds"), [(-np.inf, 0.4), (-0.4, np.inf)]
)
def test_nomad_bb_output_type(
    config: dict[str, Any],
    lower_bounds: Any,
    upper_bounds: Any,
    eval_func: Any,
    test_functions: Any,
) -> None:
    config["nonlinear_constraints"] = {
        "lower_bounds": lower_bounds,
        "upper_bounds": upper_bounds,
    }
    config["backend"]["options"] = ["BB_OUTPUT_TYPE OBJ PB"]

    weight = 1.0 if upper_bounds == 0.4 else -1.0

    def constraint_function(variables: Any, _: Any) -> Any:
        return weight * float(variables[0] + variables[2])

    result = optimize(
        config, initial_values, eval_func(test_functions, [constraint_function])
    )
    assert result.variables is not None
    assert np.allclose(result.variables, [-0.05, 0.0, 0.45], atol=0.02)

    config["backend"]["options"] = ["BB_OUTPUT_TYPE OBJ PB PB"]
    with pytest.raises(
        ValueError,
        match="Option Error: BB_OUTPUT_TYPE specifies incorrect number of outputs",
    ):
        optimize(config, initial_values, eval_func())


def test_nomad_bb_max_block_size_no_parallel(
    config: dict[str, Any], eval_func: Any
) -> None:
    config["backend"]["options"] = ["BB_MAX_BLOCK_SIZE 4"]
    with pytest.raises(
        ValueError,
        match="Option Error: BB_MAX_BLOCK_SIZE may only be specified",
    ):
        optimize(config, initial_values, eval_func())


def test_nomad_parallel_no_bb_max_block_size(
    config: dict[str, Any], eval_func: Any
) -> None:
    config["backend"]["parallel"] = True
    with pytest.raises(
        ValueError,
        match="Option Error: BB_MAX_BLOCK_SIZE must be specified",
    ):
        optimize(config, initial_values, eval_func())


@pytest.mark.parametrize("parallel", [False, True])
def test_nomad_evaluation_failure(
    config: dict[str, Any], eval_func: Any, parallel: bool, test_functions: Any
) -> None:
    config["variables"]["lower_bounds"] = [0.15, -0.5, -1.0]
    config["variables"]["upper_bounds"] = [1.0, 0.5, 0.2]
    config["backend"]["max_iterations"] = 4
    config["realizations"] = {"realization_min_success": 0}
    config["backend"]["parallel"] = parallel
    if parallel:
        config["backend"]["options"] = ["BB_MAX_BLOCK_SIZE 4"]

    result1 = optimize(config, initial_values, eval_func())
    assert result1.variables is not None
    assert np.allclose(result1.variables, [0.15, 0.0, 0.2], atol=0.02)

    counter = 0

    def _add_nan(x: Any, _: int) -> Any:
        nonlocal counter
        counter += 1
        if counter == 2:
            counter = 0
            return np.nan
        return test_functions[0](x, 0)

    result2 = optimize(config, initial_values, eval_func((_add_nan, test_functions[1])))
    assert result2.variables is not None
    assert np.allclose(result2.variables, [0.15, 0.0, 0.2], atol=0.02)
    assert not np.all(np.equal(result1.variables, result2.variables))
