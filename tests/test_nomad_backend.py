from typing import Any

import numpy as np
import pytest
from pydantic import ValidationError
from ropt.exceptions import ConfigError
from ropt.plan import BasicOptimizer
from ropt.plugins import PluginManager


@pytest.fixture(name="enopt_config")
def enopt_config_fixture() -> dict[str, Any]:
    return {
        "variables": {
            "initial_values": [0.2, 0.0, 0.1],
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
    with pytest.raises(ValidationError, match=r"Input should be 'EB',"):
        PluginManager().get_plugin("optimizer", "mads").validate_options(
            "mads", enopt_config["optimizer"]["options"]
        )

    enopt_config["optimizer"]["options"] = [
        "DIMENSION 10",
        "FOO FOO",
    ]
    with pytest.raises(
        ValidationError, match=r"Unknown or unsupported option\(s\): `DIMENSION`, `FOO`"
    ):
        PluginManager().get_plugin("optimizer", "mads").validate_options(
            "mads", enopt_config["optimizer"]["options"]
        )


@pytest.mark.parametrize("parallel", [False, True])
def test_nomad_bound_constraints(
    enopt_config: dict[str, Any], evaluator: Any, parallel: bool
) -> None:
    enopt_config["variables"]["lower_bounds"] = [0.15, -1.0, -1.0]
    enopt_config["variables"]["upper_bounds"] = [1.0, 1.0, 0.2]
    enopt_config["optimizer"]["max_iterations"] = 3
    enopt_config["optimizer"]["parallel"] = parallel
    if parallel:
        enopt_config["optimizer"]["options"] = ["BB_MAX_BLOCK_SIZE 4"]
    variables = BasicOptimizer(enopt_config, evaluator()).run().variables
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
    variables = BasicOptimizer(enopt_config, evaluator()).run().variables
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
    enopt_config["variables"]["lower_bounds"] = [-1.0, -1.0, -1.0]
    enopt_config["variables"]["upper_bounds"] = [1.0, 1.0, 1.0]
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
    variables = BasicOptimizer(enopt_config, evaluator(test_functions)).run().variables
    assert variables is not None
    assert np.allclose(variables, [-0.05, 0.0, 0.45], atol=0.02)


@pytest.mark.parametrize("parallel", [False, True])
def test_nomad_eq_nonlinear_constraints(
    enopt_config: dict[str, Any],
    evaluator: Any,
    parallel: bool,
    test_functions: Any,
) -> None:
    enopt_config["variables"]["lower_bounds"] = [-1.0, -1.0, -1.0]
    enopt_config["variables"]["upper_bounds"] = [1.0, 1.0, 1.0]
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
        ConfigError,
        match="Equality constraints are not supported by NOMAD",
    ):
        BasicOptimizer(enopt_config, evaluator(test_functions)).run()


@pytest.mark.parametrize("parallel", [False, True])
def test_nomad_ineq_nonlinear_constraints_two_sided(
    enopt_config: Any,
    parallel: bool,
    evaluator: Any,
    test_functions: Any,
) -> None:
    enopt_config["variables"]["lower_bounds"] = [-1.0, -1.0, -1.0]
    enopt_config["variables"]["upper_bounds"] = [1.0, 1.0, 1.0]
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

    variables = BasicOptimizer(enopt_config, evaluator(test_functions)).run().variables
    assert variables is not None
    assert np.allclose(variables, [-0.1, 0.0, 0.4], atol=0.02)


@pytest.mark.parametrize("parallel", [False, True])
def test_nomad_le_ge_linear_constraints(
    enopt_config: dict[str, Any], evaluator: Any, parallel: bool
) -> None:
    enopt_config["variables"]["lower_bounds"] = [-1.0, -1.0, -1.0]
    enopt_config["variables"]["upper_bounds"] = [1.0, 1.0, 1.0]
    enopt_config["linear_constraints"] = {
        "coefficients": [[1, 0, 1]],
        "lower_bounds": [-np.inf],
        "upper_bounds": [0.4],
    }
    enopt_config["optimizer"]["parallel"] = parallel
    if parallel:
        enopt_config["optimizer"]["options"] = ["BB_MAX_BLOCK_SIZE 4"]

    variables = BasicOptimizer(enopt_config, evaluator()).run().variables
    assert variables is not None
    assert np.allclose(variables, [-0.05, 0.0, 0.45], atol=0.02)


@pytest.mark.parametrize("parallel", [False, True])
def test_nomad_eq_linear_constraints(
    enopt_config: dict[str, Any], evaluator: Any, parallel: bool
) -> None:
    enopt_config["variables"]["lower_bounds"] = [-1.0, -1.0, -1.0]
    enopt_config["variables"]["upper_bounds"] = [1.0, 1.0, 1.0]
    enopt_config["linear_constraints"] = {
        "coefficients": [[1, 0, 1], [0, 1, 1]],
        "lower_bounds": [1.0, 0.75],
        "upper_bounds": [1.0, 0.75],
    }
    enopt_config["optimizer"]["parallel"] = parallel
    if parallel:
        enopt_config["optimizer"]["options"] = ["BB_MAX_BLOCK_SIZE 4"]

    with pytest.raises(
        ConfigError, match="Equality constraints are not supported by NOMAD"
    ):
        BasicOptimizer(enopt_config, evaluator()).run()


@pytest.mark.parametrize("parallel", [False, True])
def test_nomad_le_ge_linear_constraints_two_sided(
    enopt_config: Any, evaluator: Any, parallel: bool
) -> None:
    enopt_config["variables"]["lower_bounds"] = [-1.0, -1.0, -1.0]
    enopt_config["variables"]["upper_bounds"] = [1.0, 1.0, 1.0]
    enopt_config["linear_constraints"] = {
        "coefficients": [[1, 0, 1], [1, 0, 1]],
        "lower_bounds": [-np.inf, 0.0],
        "upper_bounds": [0.3, np.inf],
    }
    enopt_config["optimizer"]["parallel"] = parallel
    if parallel:
        enopt_config["optimizer"]["options"] = ["BB_MAX_BLOCK_SIZE 4"]

    variables = BasicOptimizer(enopt_config, evaluator()).run().variables
    assert variables is not None
    assert np.allclose(variables, [-0.1, 0.0, 0.4], atol=0.02)

    enopt_config["linear_constraints"] = {
        "coefficients": [[1, 0, 1]],
        "lower_bounds": [0.0],
        "upper_bounds": [0.3],
    }

    variables = BasicOptimizer(enopt_config, evaluator()).run().variables
    assert variables is not None
    assert np.allclose(variables, [-0.1, 0.0, 0.4], atol=0.02)


def test_nomad_dimension_keyword(enopt_config: dict[str, Any], evaluator: Any) -> None:
    enopt_config["variables"]["lower_bounds"] = [-1.0, -1.0, -1.0]
    enopt_config["variables"]["upper_bounds"] = [1.0, 1.0, 1.0]
    enopt_config["optimizer"]["options"] = ["DIMENSION 4"]
    with pytest.raises(ConfigError, match="Option not supported: DIMENSION 4"):
        BasicOptimizer(enopt_config, evaluator()).run()


def test_nomad_max_iterations_keyword(
    enopt_config: dict[str, Any], evaluator: Any
) -> None:
    enopt_config["variables"]["lower_bounds"] = [-1.0, -1.0, -1.0]
    enopt_config["variables"]["upper_bounds"] = [1.0, 1.0, 1.0]
    enopt_config["optimizer"]["options"] = ["MAX_ITERATIONS 4"]
    with pytest.raises(
        ConfigError,
        match="Option not supported: MAX_ITERATIONS 4",
    ):
        BasicOptimizer(enopt_config, evaluator()).run()


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
    enopt_config["variables"]["lower_bounds"] = [-1.0, -1.0, -1.0]
    enopt_config["variables"]["upper_bounds"] = [1.0, 1.0, 1.0]
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
    variables = BasicOptimizer(enopt_config, evaluator(test_functions)).run().variables
    assert variables is not None
    assert np.allclose(variables, [-0.05, 0.0, 0.45], atol=0.02)

    enopt_config["optimizer"]["options"] = ["BB_OUTPUT_TYPE OBJ PB PB"]
    with pytest.raises(
        ConfigError,
        match="Option Error: BB_OUTPUT_TYPE specifies incorrect number of outputs",
    ):
        BasicOptimizer(enopt_config, evaluator()).run()


def test_nomad_bb_max_block_size_no_parallel(
    enopt_config: dict[str, Any], evaluator: Any
) -> None:
    enopt_config["variables"]["lower_bounds"] = [-1.0, -1.0, -1.0]
    enopt_config["variables"]["upper_bounds"] = [1.0, 1.0, 1.0]
    enopt_config["optimizer"]["options"] = ["BB_MAX_BLOCK_SIZE 4"]
    with pytest.raises(
        ConfigError,
        match="Option Error: BB_MAX_BLOCK_SIZE may only be specified",
    ):
        BasicOptimizer(enopt_config, evaluator()).run()


def test_nomad_parallel_no_bb_max_block_size(
    enopt_config: dict[str, Any], evaluator: Any
) -> None:
    enopt_config["variables"]["lower_bounds"] = [-1.0, -1.0, -1.0]
    enopt_config["variables"]["upper_bounds"] = [1.0, 1.0, 1.0]
    enopt_config["optimizer"]["parallel"] = True
    with pytest.raises(
        ConfigError,
        match="Option Error: BB_MAX_BLOCK_SIZE must be specified",
    ):
        BasicOptimizer(enopt_config, evaluator()).run()


@pytest.mark.parametrize("parallel", [False, True])
def test_nomad_evaluation_failure(
    enopt_config: dict[str, Any], evaluator: Any, parallel: bool, test_functions: Any
) -> None:
    enopt_config["variables"]["lower_bounds"] = [0.15, -1.0, -1.0]
    enopt_config["variables"]["upper_bounds"] = [1.0, 1.0, 0.2]
    enopt_config["optimizer"]["max_iterations"] = 3
    enopt_config["optimizer"]["parallel"] = parallel
    enopt_config["realizations"] = {"realization_min_success": 0}
    if parallel:
        enopt_config["optimizer"]["options"] = ["BB_MAX_BLOCK_SIZE 4"]

    variables1 = BasicOptimizer(enopt_config, evaluator()).run().variables
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
        .run()
        .variables
    )
    assert variables2 is not None
    assert np.allclose(variables2, [0.15, 0.0, 0.2], atol=0.02)
    assert not np.all(np.equal(variables1, variables2))
