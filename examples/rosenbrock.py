from typing import Any, Dict  # noqa: INP001

import numpy as np
from numpy.typing import NDArray
from ropt.config.enopt import EnOptConfig
from ropt.enums import EventType
from ropt.evaluator import EvaluatorContext, EvaluatorResult
from ropt.events import OptimizationEvent
from ropt.optimization import EnsembleOptimizer
from ropt.results import FunctionResults

CONFIG: Dict[str, Any] = {
    "variables": {
        "initial_values": 2 * [0.3],
        "lower_bounds": [0.2, 0.1],
        "upper_bounds": [2.1, 2.2],
    },
    "optimizer": {
        "backend": "nomad",
        "max_iterations": 20,
        "output_dir": ".",
    },
}


def rosenbrock(variables: NDArray[np.float64], _: EvaluatorContext) -> EvaluatorResult:
    objectives = np.zeros((variables.shape[0], 1), dtype=np.float64)
    for idx in range(variables.shape[0]):
        x, y = variables[idx, :]
        objectives[idx, 0] = (1.0 - x) ** 2 + 100 * (y - x * x) ** 2
    return EvaluatorResult(
        objectives=objectives,
    )


def report(event: OptimizationEvent) -> None:
    assert event.results is not None
    for item in event.results:
        if isinstance(item, FunctionResults):
            assert item.functions is not None
            print(f"evaluation: {item.result_id}")
            print(f"  variables: {item.evaluations.variables}")
            print(f"  objective: {item.functions.weighted_objective}\n")


def run_optimization(config: Dict[str, Any]) -> None:
    optimizer = EnsembleOptimizer(rosenbrock)
    optimizer.add_observer(EventType.FINISHED_EVALUATION, report)
    optimal_result = optimizer.start_optimization(
        plan=[
            {"config": EnOptConfig.model_validate(config)},
            {"optimizer": {"id": "opt"}},
            {"tracker": {"id": "optimum", "source": "opt"}},
        ],
    )

    assert optimal_result is not None
    assert optimal_result.functions is not None
    assert np.allclose(optimal_result.evaluations.variables, 1.0, atol=0.01)
    print(f"BEST RESULT: {optimal_result.result_id}")
    print(f"  variables: {optimal_result.evaluations.variables}")
    print(f"  objective: {optimal_result.functions.weighted_objective}\n")


def main() -> None:
    run_optimization(CONFIG)


if __name__ == "__main__":
    main()
