# pylint: disable=protected-access

from typing import Any, Dict, cast

import numpy as np
import pytest
from numpy.typing import NDArray
from ropt.enums import ConstraintType
from ropt.exceptions import ConfigError
from ropt.optimization import EnsembleOptimizer


@pytest.fixture(name="enopt_config")
def enopt_config_fixture() -> Dict[str, Any]:
    return {
        "variables": {
            "initial_values": [0.2, 0.0, 0.1],
        },
        "optimizer": {
            "backend": "nomad",
            "max_iterations": 7,
        },
        "objective_functions": {
            "weights": [0.75, 0.25],
        },
    }


@pytest.mark.parametrize("parallel", [False, True])
def test_nomad_bound_constraints(
    enopt_config: Dict[str, Any], evaluator: Any, parallel: bool
) -> None:
    enopt_config["variables"]["lower_bounds"] = [0.15, -1.0, -1.0]
    enopt_config["variables"]["upper_bounds"] = [1.0, 1.0, 0.2]
    enopt_config["optimizer"]["max_iterations"] = 3
    enopt_config["optimizer"]["parallel"] = parallel
    if parallel:
        enopt_config["optimizer"]["options"] = ["BB_MAX_BLOCK_SIZE 4"]
    optimizer = EnsembleOptimizer(evaluator())
    result = optimizer.start_optimization(
        plan=[
            {"config": enopt_config},
            {"optimizer": {"id": "opt"}},
            {"tracker": {"id": "optimum", "source": "opt"}},
        ],
    )
    assert result is not None
    assert np.allclose(result.evaluations.variables, [0.15, 0.0, 0.2], atol=0.02)


def test_nomad_bound_constraints_block_size_one(
    enopt_config: Dict[str, Any], evaluator: Any
) -> None:
    enopt_config["variables"]["lower_bounds"] = [0.15, -1.0, -1.0]
    enopt_config["variables"]["upper_bounds"] = [1.0, 1.0, 0.2]
    enopt_config["optimizer"]["max_iterations"] = 3
    enopt_config["optimizer"]["parallel"] = True
    enopt_config["optimizer"]["options"] = ["BB_MAX_BLOCK_SIZE 1"]
    optimizer = EnsembleOptimizer(evaluator())
    result = optimizer.start_optimization(
        plan=[
            {"config": enopt_config},
            {"optimizer": {"id": "opt"}},
            {"tracker": {"id": "optimum", "source": "opt"}},
        ],
    )
    assert result is not None
    assert np.allclose(result.evaluations.variables, [0.15, 0.0, 0.2], atol=0.02)


@pytest.mark.parametrize("parallel", [False, True])
@pytest.mark.parametrize("bound_type", [ConstraintType.LE, ConstraintType.GE])
def test_nomad_ineq_nonlinear_constraints(
    enopt_config: Dict[str, Any],
    bound_type: ConstraintType,
    evaluator: Any,
    parallel: bool,
    test_functions: Any,
) -> None:
    enopt_config["variables"]["lower_bounds"] = [-1.0, -1.0, -1.0]
    enopt_config["variables"]["upper_bounds"] = [1.0, 1.0, 1.0]
    enopt_config["nonlinear_constraints"] = {
        "rhs_values": 0.4 if bound_type == ConstraintType.LE else -0.4,
        "types": [bound_type],
    }
    enopt_config["optimizer"]["parallel"] = parallel
    if parallel:
        enopt_config["optimizer"]["options"] = ["BB_MAX_BLOCK_SIZE 4"]

    weight = 1.0 if bound_type == ConstraintType.LE else -1.0
    test_functions = (
        *test_functions,
        lambda variables: cast(
            NDArray[np.float64], weight * variables[0] + weight * variables[2]
        ),
    )
    optimizer = EnsembleOptimizer(evaluator(test_functions))
    result = optimizer.start_optimization(
        plan=[
            {"config": enopt_config},
            {"optimizer": {"id": "opt"}},
            {
                "tracker": {
                    "id": "optimum",
                    "source": "opt",
                    "constraint_tolerance": 1e-4,
                }
            },
        ],
    )
    assert result is not None
    assert np.allclose(result.evaluations.variables, [-0.05, 0.0, 0.45], atol=0.02)


@pytest.mark.parametrize("parallel", [False, True])
def test_nomad_eq_nonlinear_constraints(
    enopt_config: Dict[str, Any],
    evaluator: Any,
    parallel: bool,
    test_functions: Any,
) -> None:
    enopt_config["variables"]["lower_bounds"] = [-1.0, -1.0, -1.0]
    enopt_config["variables"]["upper_bounds"] = [1.0, 1.0, 1.0]
    enopt_config["nonlinear_constraints"] = {
        "rhs_values": 1.0,
        "types": [ConstraintType.EQ],
    }
    enopt_config["optimizer"]["parallel"] = parallel
    if parallel:
        enopt_config["optimizer"]["options"] = ["BB_MAX_BLOCK_SIZE 4"]

    test_functions = (
        *test_functions,
        lambda variables: cast(NDArray[np.float64], variables[0] + variables[2]),
    )
    optimizer = EnsembleOptimizer(evaluator(test_functions))
    with pytest.raises(
        ConfigError,
        match="Equality constraints are not supported by NOMAD",
    ):
        optimizer.start_optimization(plan=[{"config": enopt_config}, {"optimizer": {}}])


@pytest.mark.parametrize("parallel", [False, True])
def test_nomad_le_ge_linear_constraints(
    enopt_config: Dict[str, Any], evaluator: Any, parallel: bool
) -> None:
    enopt_config["variables"]["lower_bounds"] = [-1.0, -1.0, -1.0]
    enopt_config["variables"]["upper_bounds"] = [1.0, 1.0, 1.0]
    enopt_config["linear_constraints"] = {
        "coefficients": [[1, 0, 1], [-1, 0, -1]],
        "rhs_values": [0.4, -0.4],
        "types": [ConstraintType.LE, ConstraintType.GE],
    }
    enopt_config["optimizer"]["parallel"] = parallel
    if parallel:
        enopt_config["optimizer"]["options"] = ["BB_MAX_BLOCK_SIZE 4"]

    optimizer = EnsembleOptimizer(evaluator())
    result = optimizer.start_optimization(
        plan=[
            {"config": enopt_config},
            {"optimizer": {"id": "opt"}},
            {
                "tracker": {
                    "id": "optimum",
                    "source": "opt",
                    "constraint_tolerance": 1e-4,
                }
            },
        ],
    )
    assert result is not None
    assert np.allclose(result.evaluations.variables, [-0.05, 0.0, 0.45], atol=0.02)


@pytest.mark.parametrize("parallel", [False, True])
def test_nomad_eq_linear_constraints(
    enopt_config: Dict[str, Any], evaluator: Any, parallel: bool
) -> None:
    enopt_config["variables"]["lower_bounds"] = [-1.0, -1.0, -1.0]
    enopt_config["variables"]["upper_bounds"] = [1.0, 1.0, 1.0]
    enopt_config["linear_constraints"] = {
        "coefficients": [[1, 0, 1], [0, 1, 1]],
        "rhs_values": [1.0, 0.75],
        "types": [ConstraintType.EQ, ConstraintType.EQ],
    }
    enopt_config["optimizer"]["parallel"] = parallel
    if parallel:
        enopt_config["optimizer"]["options"] = ["BB_MAX_BLOCK_SIZE 4"]

    optimizer = EnsembleOptimizer(evaluator())
    with pytest.raises(
        ConfigError, match="Equality constraints are not supported by NOMAD"
    ):
        optimizer.start_optimization(
            plan=[
                {"config": enopt_config},
                {"optimizer": {"id": "opt"}},
                {
                    "tracker": {
                        "id": "optimum",
                        "source": "opt",
                        "constraint_tolerance": 1e-4,
                    }
                },
            ],
        )


def test_nomad_dimension_keyword(enopt_config: Dict[str, Any], evaluator: Any) -> None:
    enopt_config["variables"]["lower_bounds"] = [-1.0, -1.0, -1.0]
    enopt_config["variables"]["upper_bounds"] = [1.0, 1.0, 1.0]
    enopt_config["optimizer"]["options"] = ["DIMENSION 4"]
    optimizer = EnsembleOptimizer(evaluator())
    with pytest.raises(ConfigError, match="Option Error: DIMENSION cannot be used"):
        optimizer.start_optimization(plan=[{"config": enopt_config}, {"optimizer": {}}])


def test_nomad_max_iterations_keyword(
    enopt_config: Dict[str, Any], evaluator: Any
) -> None:
    enopt_config["variables"]["lower_bounds"] = [-1.0, -1.0, -1.0]
    enopt_config["variables"]["upper_bounds"] = [1.0, 1.0, 1.0]
    enopt_config["optimizer"]["options"] = ["MAX_ITERATIONS 4"]
    optimizer = EnsembleOptimizer(evaluator())
    with pytest.raises(
        ConfigError,
        match="Option Error: MAX_ITERATIONS, maximum iterations already configured",
    ):
        optimizer.start_optimization(plan=[{"config": enopt_config}, {"optimizer": {}}])


@pytest.mark.parametrize("bound_type", [ConstraintType.LE, ConstraintType.GE])
def test_nomad_bb_output_type(
    enopt_config: Dict[str, Any],
    bound_type: ConstraintType,
    evaluator: Any,
    test_functions: Any,
) -> None:
    enopt_config["variables"]["lower_bounds"] = [-1.0, -1.0, -1.0]
    enopt_config["variables"]["upper_bounds"] = [1.0, 1.0, 1.0]
    enopt_config["nonlinear_constraints"] = {
        "rhs_values": 0.4 if bound_type == ConstraintType.LE else -0.4,
        "types": [bound_type],
    }
    enopt_config["optimizer"]["options"] = ["BB_OUTPUT_TYPE OBJ PB"]

    weight = 1.0 if bound_type == ConstraintType.LE else -1.0
    test_functions = (
        *test_functions,
        lambda variables: cast(
            NDArray[np.float64], weight * variables[0] + weight * variables[2]
        ),
    )
    optimizer = EnsembleOptimizer(evaluator(test_functions))
    result = optimizer.start_optimization(
        plan=[
            {"config": enopt_config},
            {"optimizer": {"id": "opt"}},
            {
                "tracker": {
                    "id": "optimum",
                    "source": "opt",
                    "constraint_tolerance": 1e-4,
                }
            },
        ],
    )
    assert result is not None
    assert np.allclose(result.evaluations.variables, [-0.05, 0.0, 0.45], atol=0.02)

    enopt_config["optimizer"]["options"] = ["BB_OUTPUT_TYPE OBJ PB PB"]
    optimizer = EnsembleOptimizer(evaluator(test_functions))
    with pytest.raises(
        ConfigError,
        match="Option Error: BB_OUTPUT_TYPE specifies incorrect number of outputs",
    ):
        optimizer.start_optimization(
            plan=[
                {"config": enopt_config},
                {"optimizer": {"id": "opt"}},
                {
                    "tracker": {
                        "id": "optimum",
                        "source": "opt",
                        "constraint_tolerance": 1e-4,
                    }
                },
            ],
        )


def test_nomad_bb_max_block_size_no_parallel(
    enopt_config: Dict[str, Any], evaluator: Any
) -> None:
    enopt_config["variables"]["lower_bounds"] = [-1.0, -1.0, -1.0]
    enopt_config["variables"]["upper_bounds"] = [1.0, 1.0, 1.0]
    enopt_config["optimizer"]["options"] = ["BB_MAX_BLOCK_SIZE 4"]
    optimizer = EnsembleOptimizer(evaluator())
    with pytest.raises(
        ConfigError,
        match="Option Error: BB_MAX_BLOCK_SIZE may only be specified",
    ):
        optimizer.start_optimization(plan=[{"config": enopt_config}, {"optimizer": {}}])


def test_nomad_parallel_no_bb_max_block_size(
    enopt_config: Dict[str, Any], evaluator: Any
) -> None:
    enopt_config["variables"]["lower_bounds"] = [-1.0, -1.0, -1.0]
    enopt_config["variables"]["upper_bounds"] = [1.0, 1.0, 1.0]
    enopt_config["optimizer"]["parallel"] = True
    optimizer = EnsembleOptimizer(evaluator())
    with pytest.raises(
        ConfigError,
        match="Option Error: BB_MAX_BLOCK_SIZE must be specified",
    ):
        optimizer.start_optimization(plan=[{"config": enopt_config}, {"optimizer": {}}])


@pytest.mark.parametrize("parallel", [False, True])
def test_nomad_evaluation_failure(
    enopt_config: Dict[str, Any], evaluator: Any, parallel: bool, test_functions: Any
) -> None:
    enopt_config["variables"]["lower_bounds"] = [0.15, -1.0, -1.0]
    enopt_config["variables"]["upper_bounds"] = [1.0, 1.0, 0.2]
    enopt_config["optimizer"]["max_iterations"] = 3
    enopt_config["optimizer"]["parallel"] = parallel
    enopt_config["realizations"] = {"realization_min_success": 0}
    if parallel:
        enopt_config["optimizer"]["options"] = ["BB_MAX_BLOCK_SIZE 4"]

    optimizer = EnsembleOptimizer(evaluator())
    result1 = optimizer.start_optimization(
        plan=[
            {"config": enopt_config},
            {"optimizer": {"id": "opt"}},
            {"tracker": {"id": "optimum", "source": "opt"}},
        ],
    )
    assert result1 is not None
    assert np.allclose(result1.evaluations.variables, [0.15, 0.0, 0.2], atol=0.02)

    counter = 0

    def _add_nan(x: Any) -> Any:
        nonlocal counter
        counter += 1
        if counter == 2:
            counter = 0
            return np.nan
        return test_functions[0](x)

    optimizer = EnsembleOptimizer(evaluator((_add_nan, test_functions[1])))
    result2 = optimizer.start_optimization(
        plan=[
            {"config": enopt_config},
            {"optimizer": {"id": "opt"}},
            {"tracker": {"id": "optimum", "source": "opt"}},
        ],
    )
    assert result2 is not None
    assert np.allclose(result2.evaluations.variables, [0.15, 0.0, 0.2], atol=0.02)
    assert not np.all(
        np.equal(result1.evaluations.variables, result2.evaluations.variables)
    )
