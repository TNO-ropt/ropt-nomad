"""Rosenbrock example."""

from typing import Any

import numpy as np
from numpy.typing import NDArray
from ropt.evaluator import EvaluatorContext, EvaluatorResult
from ropt.plan import BasicOptimizer
from ropt.results import FunctionResults, Results

CONFIG: dict[str, Any] = {
    "variables": {
        "initial_values": 2 * [0.3],
        "lower_bounds": [0.2, 0.1],
        "upper_bounds": [2.1, 2.2],
    },
    "optimizer": {
        "method": "nomad/default",
        "max_iterations": 20,
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
    optimal_result = (
        BasicOptimizer(config, rosenbrock).set_results_callback(report).run().results
    )
    assert optimal_result is not None
    assert optimal_result.functions is not None
    assert np.allclose(optimal_result.evaluations.variables, 1.0, atol=0.01)
    print(f"  variables: {optimal_result.evaluations.variables}")
    print(f"  objective: {optimal_result.functions.weighted_objective}\n")


def main() -> None:
    """Main function."""
    run_optimization(CONFIG)


if __name__ == "__main__":
    main()
