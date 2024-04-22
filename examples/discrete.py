from typing import Any, Dict  # noqa: INP001

import numpy as np
from numpy.typing import NDArray
from ropt.config.enopt import EnOptConfig
from ropt.enums import ConstraintType, EventType, VariableType
from ropt.evaluator import EvaluatorContext, EvaluatorResult
from ropt.events import OptimizationEvent
from ropt.optimization import EnsembleOptimizer
from ropt.results import FunctionResults

CONFIG: Dict[str, Any] = {
    "variables": {
        "initial_values": 2 * [0.0],
        "lower_bounds": [0.0, 0.0],
        "upper_bounds": [10.0, 10.0],
        "types": VariableType.INTEGER,
    },
    "optimizer": {
        "backend": "nomad",
        "options": ["MAX_EVAL 100"],
        "output_dir": ".",
    },
    "nonlinear_constraints": {
        "types": [ConstraintType.LE],
        "rhs_values": [0.0],
    },
}


def function(variables: NDArray[np.float64], _: EvaluatorContext) -> EvaluatorResult:
    x, y = variables[0, :]
    objectives = np.array(-min(3 * x, y), ndmin=2, dtype=np.float64)
    constraints = np.array(x + y - 10, ndmin=2, dtype=np.float64)
    return EvaluatorResult(objectives=objectives, constraints=constraints)


def report(event: OptimizationEvent) -> None:
    assert event.results is not None
    for item in event.results:
        if isinstance(item, FunctionResults):
            assert item.functions is not None
            print(f"evaluation: {item.result_id}")
            print(f"  variables: {item.evaluations.variables}")
            print(f"  objective: {item.functions.weighted_objective}\n")


def run_optimization(config: Dict[str, Any]) -> None:
    optimizer = EnsembleOptimizer(function)
    optimizer.add_observer(EventType.FINISHED_EVALUATION, report)
    optimal_result = optimizer.start_optimization(
        plan=[
            {"config": EnOptConfig.model_validate(config)},
            {"optimizer": {"id": "opt"}},
            {
                "tracker": {
                    "id": "optimum",
                    "source": "opt",
                    "constraint_tolerance": 1e-10,
                }
            },
        ],
    )

    assert optimal_result is not None
    assert optimal_result.functions is not None
    assert np.all(np.equal(optimal_result.evaluations.variables, [3, 7]))
    print(f"BEST RESULT: {optimal_result.result_id}")
    print(f"  variables: {optimal_result.evaluations.variables}")
    print(f"  objective: {optimal_result.functions.weighted_objective}\n")


def main() -> None:
    run_optimization(CONFIG)


if __name__ == "__main__":
    main()
