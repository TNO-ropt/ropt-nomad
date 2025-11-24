"""Discrete optimization example."""

from typing import Any

import numpy as np
from numpy.typing import NDArray
from ropt.enums import VariableType
from ropt.evaluator import EvaluatorContext, EvaluatorResult
from ropt.results import FunctionResults, Results
from ropt.workflow import BasicOptimizer

initial_values = 2 * [0.0]

CONFIG: dict[str, Any] = {
    "variables": {
        "variable_count": 2,
        "lower_bounds": [0.0, 0.0],
        "upper_bounds": [10.0, 10.0],
        "types": VariableType.INTEGER,
    },
    "optimizer": {
        "method": "nomad/default",
        "options": ["MAX_EVAL 100"],
        "output_dir": ".",
    },
    "nonlinear_constraints": {
        "lower_bounds": [-np.inf],
        "upper_bounds": [0.0],
    },
}


def function(variables: NDArray[np.float64], _: EvaluatorContext) -> EvaluatorResult:
    """Evaluate the function.

    Args:
        variables: The variables to evaluate
        context:   Evaluator context

    Returns:
        Calculated objectives and constraints.
    """
    x, y = variables[0, :]
    objectives = np.array(-min(3 * x, y), ndmin=2, dtype=np.float64)
    constraints = np.array(x + y - 10, ndmin=2, dtype=np.float64)
    return EvaluatorResult(objectives=objectives, constraints=constraints)


def report(results: tuple[Results, ...]) -> None:
    """Report results of an evaluation.

    Args:
        results: The results.
    """
    for item in results:
        if isinstance(item, FunctionResults) and item.functions is not None:
            print(f"  variables: {item.evaluations.variables}")
            print(f"  objective: {item.functions.weighted_objective}\n")


def run_optimization(config: dict[str, Any]) -> None:
    """Run the optimization."""
    optimizer = BasicOptimizer(config, function)
    optimizer.set_results_callback(report)
    optimizer.run(initial_values)
    assert optimizer.results is not None
    assert optimizer.results.functions is not None
    assert np.all(np.equal(optimizer.results.evaluations.variables, [3, 7]))
    print(f"  variables: {optimizer.results.evaluations.variables}")
    print(f"  objective: {optimizer.results.functions.weighted_objective}\n")


def main() -> None:
    """Main function."""
    run_optimization(CONFIG)


if __name__ == "__main__":
    main()
