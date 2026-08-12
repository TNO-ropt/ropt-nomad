"""Discrete optimization example."""

from typing import Any

import numpy as np
from numpy.typing import NDArray
from ropt.enums import VariableType
from ropt.simple import EvaluateResult, EvaluationFunctionContext, optimize

initial_values = 2 * [0.0]

CONFIG: dict[str, Any] = {
    "variables": {
        "variable_count": 2,
        "lower_bounds": [0.0, 0.0],
        "upper_bounds": [10.0, 10.0],
        "types": VariableType.INTEGER,
    },
    "backend": {
        "method": "nomad/default",
        "options": ["MAX_EVAL 100"],
    },
    "optimizer": {
        "output_dir": ".",
    },
    "nonlinear_constraints": {
        "lower_bounds": [-np.inf],
        "upper_bounds": [0.0],
    },
}


def function(
    variables: NDArray[np.float64], _: EvaluationFunctionContext
) -> list[float]:
    """Evaluate the function.

    Args:
        variables: The variables to evaluate

    Returns:
        Calculated objectives and constraints.
    """
    x, y = variables
    objective = -min(3 * x, y)
    constraint = x + y - 10
    return [float(objective), float(constraint)]


def report(result: EvaluateResult) -> None:
    """Report results of an evaluation.

    Args:
        result: The result of a single function evaluation.
    """
    if result.target_objective is not None:
        print(f"  variables: {result.results.evaluations.variables}")
        print(f"  objective: {result.target_objective}\n")


def run_optimization(config: dict[str, Any]) -> None:
    """Run the optimization."""
    result = optimize(config, initial_values, function, report=report)
    assert result.variables is not None
    assert np.all(np.equal(result.variables, [3, 7]))
    print(f"  variables: {result.variables}")
    print(f"  objective: {result.target_objective}\n")


def main() -> None:
    """Main function."""
    run_optimization(CONFIG)


if __name__ == "__main__":
    main()
