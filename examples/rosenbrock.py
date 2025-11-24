"""Rosenbrock example."""

from typing import Any

import numpy as np
from numpy.typing import NDArray
from ropt.evaluator import EvaluatorContext, EvaluatorResult
from ropt.results import FunctionResults, Results
from ropt.workflow import BasicOptimizer

initial_values = 2 * [0.4]

CONFIG: dict[str, Any] = {
    "variables": {
        "variable_count": len(initial_values),
        "lower_bounds": [0.4, 0.3],
        "upper_bounds": [1.7, 1.8],
    },
    "optimizer": {
        "method": "nomad/default",
        "max_iterations": 10,
        "output_dir": ".",
    },
}


def rosenbrock(variables: NDArray[np.float64], _: EvaluatorContext) -> EvaluatorResult:
    """Evaluate the rosenbrock function.

    Args:
        variables: The variables to evaluate
        context:   Evaluator context

    Returns:
        Calculated objectives.
    """
    objectives = np.zeros((variables.shape[0], 1), dtype=np.float64)
    for idx in range(variables.shape[0]):
        x, y = variables[idx, :]
        objectives[idx, 0] = (1.0 - x) ** 2 + 100 * (y - x * x) ** 2
    return EvaluatorResult(
        objectives=objectives,
    )


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
    optimizer = BasicOptimizer(config, rosenbrock)
    optimizer.set_results_callback(report)
    optimizer.run(initial_values)
    assert optimizer.results is not None
    assert optimizer.results.functions is not None
    assert np.allclose(optimizer.results.evaluations.variables, 1.0, atol=0.01)
    print(f"  variables: {optimizer.results.evaluations.variables}")
    print(f"  objective: {optimizer.results.functions.weighted_objective}\n")


def main() -> None:
    """Main function."""
    run_optimization(CONFIG)


if __name__ == "__main__":
    main()
