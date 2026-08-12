"""Rosenbrock example."""

from typing import Any

import numpy as np
from numpy.typing import NDArray
from ropt.simple import EvaluateResult, EvaluationFunctionContext, optimize

initial_values = 2 * [0.4]

CONFIG: dict[str, Any] = {
    "variables": {
        "variable_count": len(initial_values),
        "lower_bounds": [0.4, 0.3],
        "upper_bounds": [1.7, 1.8],
    },
    "backend": {
        "method": "nomad/default",
        "max_iterations": 10,
    },
    "optimizer": {
        "output_dir": ".",
    },
}


def rosenbrock(variables: NDArray[np.float64], _: EvaluationFunctionContext) -> float:
    """Evaluate the rosenbrock function.

    Args:
        variables: The variables to evaluate

    Returns:
        Calculated objectives.
    """
    x, y = variables
    return float((1.0 - x) ** 2 + 100 * (y - x * x) ** 2)


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
    result = optimize(config, initial_values, rosenbrock, report=report)
    assert result.variables is not None
    assert np.allclose(result.variables, 1.0, atol=0.01)
    print(f"  variables: {result.variables}")
    print(f"  objective: {result.target_objective}\n")


def main() -> None:
    """Main function."""
    run_optimization(CONFIG)


if __name__ == "__main__":
    main()
