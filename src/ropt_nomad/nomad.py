"""This module implements the NOMAD optimization plugin."""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

import numpy as np
import PyNomad
from ropt.enums import VariableType
from ropt.exceptions import ConfigError, OptimizationAborted
from ropt.plugins.optimizer.base import Optimizer, OptimizerCallback, OptimizerPlugin
from ropt.plugins.optimizer.utils import (
    NormalizedConstraints,
    get_masked_linear_constraints,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from ropt.config.enopt import EnOptConfig

_SUPPORTED_METHODS: Final = {"mads"}


class NomadOptimizer(Optimizer):
    """Backend class for optimization via nomad."""

    def __init__(
        self,
        enopt_config: EnOptConfig,
        optimizer_callback: OptimizerCallback,
    ) -> None:
        """Initialize the optimizer implemented by the nomad plugin.

        See the [ropt.plugins.optimizer.base.Optimizer][] abstract base class.

        # noqa
        """
        self._config = enopt_config
        self._optimizer_callback = optimizer_callback
        self._bounds = self._get_bounds()
        self._normalized_constraints = self._init_constraints()
        self._parameters = self._get_parameters(self._normalized_constraints)
        self._cached_variables: NDArray[np.float64] | None = None
        self._cached_function: NDArray[np.float64] | None = None
        self._exception: OptimizationAborted | None = None

        _, _, self._method = self._config.optimizer.method.lower().rpartition("/")
        if self._method == "default":
            self._method = "mads"
        if self._method not in _SUPPORTED_METHODS:
            msg = f"NOMAD optimizer algorithm {self._method} is not supported"
            raise NotImplementedError(msg)

    @property
    def is_parallel(self) -> bool:
        """Whether the current run is parallel.

        See the [ropt.plugins.optimizer.base.Optimizer][] abstract base class.

        # noqa
        """
        return self._config.optimizer.parallel

    def start(self, initial_values: NDArray[np.float64]) -> None:
        """Start the optimization.

        See the [ropt.plugins.optimizer.base.Optimizer][] abstract base class.

        # noqa
        """
        self._cached_variables = None
        self._cached_function = None

        if self._config.variables.mask is not None:
            initial_values = initial_values[self._config.variables.mask]

        PyNomad.optimize(
            self._evaluate,
            initial_values.tolist(),
            self._bounds[0],
            self._bounds[1],
            self._parameters,
        )
        if self._exception is not None:
            raise self._exception

    @property
    def allow_nan(self) -> bool:
        """Whether NaN is allowed.

        See the [ropt.plugins.optimizer.base.Optimizer][] abstract base class.

        # noqa
        """
        return True

    def _get_bounds(self) -> tuple[list[float], list[float]]:
        lower_bounds = self._config.variables.lower_bounds
        upper_bounds = self._config.variables.upper_bounds
        if self._config.variables.mask is not None:
            lower_bounds = lower_bounds[self._config.variables.mask]
            upper_bounds = upper_bounds[self._config.variables.mask]
        return lower_bounds.tolist(), upper_bounds.tolist()

    def _evaluate(
        self,
        block_or_eval_point: PyNomad.PyNomadEvalPoint | PyNomad.PyNomadBlock,
    ) -> int | list[int]:
        if isinstance(block_or_eval_point, PyNomad.PyNomadEvalPoint):
            eval_points = [block_or_eval_point]
        else:
            eval_points = [
                block_or_eval_point.get_x(block_idx)
                for block_idx in range(block_or_eval_point.size())
            ]
        variables = np.vstack(
            [
                np.fromiter(
                    (eval_point.get_coord(idx) for idx in range(eval_point.size())),
                    dtype=np.float64,
                )
                for eval_point in eval_points
            ],
        )
        try:
            objectives = self._calculate_objective(variables)
            constraints = self._calculate_constraints(variables)
        except OptimizationAborted as exc:
            self._exception = exc
            return (
                0
                if isinstance(block_or_eval_point, PyNomad.PyNomadEvalPoint)
                else [0] * len(eval_points)
            )
        for idx, eval_point in enumerate(eval_points):
            result_string = str(objectives[idx])
            if constraints.size:
                result_string += " " + " ".join(
                    str(value) for value in constraints[idx, :]
                )
            eval_point.setBBO(result_string.encode("UTF-8"))

        return (
            int(not np.isnan(objectives[0]))
            if isinstance(block_or_eval_point, PyNomad.PyNomadEvalPoint)
            else [int(not np.isnan(objective)) for objective in objectives]
        )

    def _get_parameters(  # noqa: C901, PLR0912
        self, normalized_constraints: NormalizedConstraints | None
    ) -> list[str]:
        dim = (
            self._config.variables.initial_values.size
            if self._config.variables.mask is None
            else self._config.variables.mask.sum()
        )
        parameters = [f"DIMENSION {dim}"]

        constraints = (
            0 if normalized_constraints is None else len(normalized_constraints.is_eq)
        )
        bb_output_type: str | None = "BB_OUTPUT_TYPE OBJ" + " EB" * constraints
        have_bb_max_block_size = False
        bb_input_type = None

        if self._config.optimizer.max_iterations is not None:
            parameters.append(f"MAX_ITERATIONS {self._config.optimizer.max_iterations}")

        if isinstance(self._config.optimizer.options, list):
            for option in self._config.optimizer.options:
                if option.strip().startswith("DIMENSION"):
                    msg = "Option Error: DIMENSION cannot be used"
                    raise ConfigError(msg)

                if (
                    self._config.optimizer.max_iterations is not None
                    and option.strip().startswith("MAX_ITERATIONS")
                ):
                    msg = (
                        "Option Error: MAX_ITERATIONS, maximum iterations "
                        "already configured"
                    )
                    raise ConfigError(msg)

                if option.strip().startswith("BB_INPUT_TYPE"):
                    bb_input_type = option.strip()
                elif self._config.variables.types is not None:
                    types = (
                        self._config.variables.types
                        if self._config.variables.mask is None
                        else self._config.variables.types[self._config.variables.mask]
                    )
                    bb_input_type = "BB_INPUT_TYPE ("
                    for item in types:
                        bb_input_type += " I" if item == VariableType.INTEGER else " R"
                    bb_input_type += " )"

                if option.strip().startswith("BB_OUTPUT_TYPE"):
                    if len(option.split()) != constraints + 2:
                        msg = (
                            "Option Error: BB_OUTPUT_TYPE specifies "
                            "incorrect number of outputs"
                        )
                        raise ConfigError(msg)
                    bb_output_type = None

                if option.strip().startswith("BB_MAX_BLOCK_SIZE"):
                    if self._config.optimizer.parallel is False:
                        msg = (
                            "Option Error: BB_MAX_BLOCK_SIZE may only be specified "
                            "if the parallel option is True"
                        )
                        raise ConfigError(msg)
                    have_bb_max_block_size = True

            parameters.extend(self._config.optimizer.options)

        if self._config.optimizer.parallel and have_bb_max_block_size is False:
            msg = (
                "Option Error: BB_MAX_BLOCK_SIZE must be specified "
                "if the parallel option is True"
            )
            raise ConfigError(msg)

        if bb_input_type is not None:
            parameters = [bb_input_type, *parameters]

        if bb_output_type is not None:
            parameters.append(bb_output_type)

        return parameters

    def _init_constraints(self) -> NormalizedConstraints | None:
        lower_bounds = []
        upper_bounds = []
        if self._config.nonlinear_constraints is not None:
            lower_bounds.append(self._config.nonlinear_constraints.lower_bounds)
            upper_bounds.append(self._config.nonlinear_constraints.upper_bounds)
        self._lin_coef: NDArray[np.float64] | None = None
        if self._config.linear_constraints is not None:
            self._lin_coef, lin_lower, lin_upper = get_masked_linear_constraints(
                self._config
            )
            lower_bounds.append(lin_lower)
            upper_bounds.append(lin_upper)
        if lower_bounds:
            normalized_constraints = NormalizedConstraints(
                np.concatenate(lower_bounds), np.concatenate(upper_bounds), flip=True
            )
            if np.any(normalized_constraints.is_eq):
                msg = "Equality constraints are not supported by NOMAD"
                raise ConfigError(msg)
            return normalized_constraints
        return None

    def _calculate_objective(
        self, variables: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        functions = self._get_functions(variables)
        if variables.ndim > 1:
            return functions[:, 0]
        return np.array(functions[0])

    def _calculate_constraints(
        self, variables: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        if self._normalized_constraints is None:
            return np.array([])
        if self._normalized_constraints.constraints is None:
            constraints = []
            if self._config.nonlinear_constraints is not None:
                functions = self._get_functions(variables)
                constraints.append(
                    (
                        functions[1:] if variables.ndim == 1 else functions[:, 1:]
                    ).transpose()
                )
            if self._lin_coef is not None:
                constraints.append(np.matmul(self._lin_coef, variables.transpose()))
            if constraints:
                self._normalized_constraints.set_constraints(
                    np.concatenate(constraints, axis=0)
                )
        assert self._normalized_constraints.constraints is not None
        return self._normalized_constraints.constraints.transpose()

    def _get_functions(self, variables: NDArray[np.float64]) -> NDArray[np.float64]:
        if (
            self._cached_variables is None
            or variables.shape != self._cached_variables.shape
            or not np.allclose(variables, self._cached_variables)
        ):
            self._cached_variables = None
            self._cached_function = None
            if self._normalized_constraints is not None:
                self._normalized_constraints.reset()
        if self._cached_function is None:
            self._cached_variables = variables.copy()
            function, _ = self._optimizer_callback(
                variables,
                return_functions=True,
                return_gradients=False,
            )
            self._cached_function = function.copy()
        return self._cached_function


class NomadOptimizerPlugin(OptimizerPlugin):
    """Nomad optimizer plugin class."""

    def create(
        self, config: EnOptConfig, optimizer_callback: OptimizerCallback
    ) -> NomadOptimizer:
        """Initialize the optimizer plugin.

        See the [ropt.plugins.optimizer.base.OptimizerPlugin][] abstract base class.

        # noqa
        """
        return NomadOptimizer(config, optimizer_callback)

    def is_supported(self, method: str) -> bool:
        """Check if a method is supported.

        See the [ropt.plugins.optimizer.base.OptimizerPlugin][] abstract base class.

        # noqa
        """
        return method.lower() in (_SUPPORTED_METHODS | {"default"})
