"""This module implements the NOMAD optimization plugin."""

from __future__ import annotations

import os
import sys
from contextlib import contextmanager
from typing import TYPE_CHECKING, Final, Generator

import numpy as np
import PyNomad
from ropt.enums import ConstraintType, VariableType
from ropt.exceptions import ConfigError
from ropt.plugins.optimizer.base import Optimizer, OptimizerCallback, OptimizerPlugin
from ropt.plugins.optimizer.utils import create_output_path, filter_linear_constraints

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import NDArray
    from ropt.config.enopt import EnOptConfig

_OUTPUT_FILE: Final = "optimizer_output"

_SUPPORTED_METHODS: Final = {"mads"}


class _Redirector:
    def __init__(self, output_file: Path | None) -> None:
        sys.stdout.flush()
        sys.stderr.flush()
        self._old_stdout = os.dup(1)
        self._old_stderr = os.dup(2)
        self._output_file = output_file
        self._new_stdout = (
            os.open(os.devnull, os.O_WRONLY)
            if output_file is None
            else os.open(output_file, os.O_WRONLY | os.O_CREAT)
        )

    @contextmanager
    def start(self) -> Generator[None, None, None]:
        try:
            sys.stdout.flush()
            sys.stderr.flush()
            os.dup2(self._new_stdout, 1)
            os.dup2(self._new_stdout, 2)
            yield
        finally:
            os.dup2(self._old_stdout, 1)
            os.dup2(self._old_stderr, 2)
            os.close(self._new_stdout)

    @contextmanager
    def stop(self) -> Generator[None, None, None]:
        try:
            if self._output_file is not None:
                os.fsync(self._new_stdout)
            os.dup2(self._old_stdout, 1)
            os.dup2(self._old_stderr, 2)
            yield
        finally:
            sys.stdout.flush()
            sys.stderr.flush()
            os.dup2(self._new_stdout, 1)
            os.dup2(self._new_stdout, 2)


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
        self._parameters = self._get_parameters()
        self._coefficients: NDArray[np.float64] | None = None
        self._rhs_values: NDArray[np.float64] | None = None
        self._cached_variables: NDArray[np.float64] | None = None
        self._cached_function: NDArray[np.float64] | None = None
        self._redirector: _Redirector

        _, _, self._method = self._config.optimizer.method.lower().rpartition("/")
        if self._method == "default":
            self._method = "mads"
        if self._method not in _SUPPORTED_METHODS:
            msg = f"NOMAD optimizer algorithm {self._method} is not supported"
            raise NotImplementedError(msg)

        self._get_constraints()

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

        variable_indices = self._config.variables.indices
        if variable_indices is not None:
            initial_values = initial_values[variable_indices]

        output_dir = self._config.optimizer.output_dir
        output_file: Path | None = None
        if output_dir is not None:
            output_file = create_output_path(_OUTPUT_FILE, output_dir, suffix=".txt")

        self._redirector = _Redirector(output_file)
        with self._redirector.start():
            PyNomad.optimize(
                self._evaluate,
                initial_values.tolist(),
                self._bounds[0],
                self._bounds[1],
                self._parameters,
            )

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
        variable_indices = self._config.variables.indices
        if variable_indices is not None:
            lower_bounds = lower_bounds[variable_indices]
            upper_bounds = upper_bounds[variable_indices]
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
        objectives = self._calculate_objective(variables)
        constraints = self._calculate_constraints(variables)
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

    def _get_parameters(self) -> list[str]:  # noqa: C901, PLR0912
        variable_indices = self._config.variables.indices
        dim = (
            self._config.variables.initial_values.size
            if variable_indices is None
            else variable_indices.size
        )
        parameters = [f"DIMENSION {dim}"]

        nonlinear = self._config.nonlinear_constraints
        non_linear_count = 0 if nonlinear is None else nonlinear.rhs_values.size
        linear = self._config.linear_constraints
        if linear is not None and variable_indices is not None:
            linear = filter_linear_constraints(linear, variable_indices)
        linear_count = 0 if linear is None else linear.rhs_values.size
        bb_output_type: str | None = "BB_OUTPUT_TYPE OBJ" + " EB" * (
            linear_count + non_linear_count
        )
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
                        if variable_indices is None
                        else self._config.variables.types[variable_indices]
                    )
                    bb_input_type = "BB_INPUT_TYPE ("
                    for item in types:
                        bb_input_type += " I" if item == VariableType.INTEGER else " R"
                    bb_input_type += " )"

                if option.strip().startswith("BB_OUTPUT_TYPE"):
                    if len(option.split()) != linear_count + non_linear_count + 2:
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

    def _get_constraints(self) -> None:
        nonlinear_config = self._config.nonlinear_constraints
        if nonlinear_config is not None and ConstraintType.EQ in nonlinear_config.types:
            msg = "Equality constraints are not supported by NOMAD"
            raise ConfigError(msg)

        linear_config = self._config.linear_constraints
        if linear_config is not None:
            if ConstraintType.EQ in linear_config.types:
                msg = "Equality constraints are not supported by NOMAD"
                raise ConfigError(msg)
            if self._config.variables.indices is not None:
                linear_config = filter_linear_constraints(
                    linear_config, self._config.variables.indices
                )
            self._coefficients = linear_config.coefficients.copy()
            self._rhs_values = linear_config.rhs_values.copy()
            self._coefficients[linear_config.types == ConstraintType.GE] *= -1.0
            self._rhs_values[linear_config.types == ConstraintType.GE] *= -1.0

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
        have_nonlinear = self._config.nonlinear_constraints is not None
        have_linear = self._coefficients is not None
        if have_nonlinear:
            functions = self._get_functions(variables)
            nonlinear_constraints = (
                functions[1:] if variables.ndim == 1 else functions[:, 1:]
            )
        if have_linear:
            assert self._coefficients is not None
            linear_constraints = (
                np.array(np.matmul(self._coefficients, variables) - self._rhs_values)
                if variables.ndim == 1
                else np.vstack(
                    [
                        np.matmul(self._coefficients, variables[idx, :])
                        - self._rhs_values
                        for idx in range(variables.shape[0])
                    ],
                )
            )
        if have_nonlinear and have_linear:
            return np.hstack((nonlinear_constraints, linear_constraints))
        if have_nonlinear:
            return nonlinear_constraints
        if have_linear:
            return linear_constraints
        return np.array([])

    def _get_functions(self, variables: NDArray[np.float64]) -> NDArray[np.float64]:
        if (
            self._cached_variables is None
            or variables.shape != self._cached_variables.shape
            or not np.allclose(variables, self._cached_variables)
        ):
            self._cached_variables = None
            self._cached_function = None
        if self._cached_function is None:
            self._cached_variables = variables.copy()
            with self._redirector.stop():
                function, _ = self._optimizer_callback(
                    variables,
                    return_functions=True,
                    return_gradients=False,
                )
            self._cached_function = function.copy()
        return self._cached_function


class NomadOptimizerPlugin(OptimizerPlugin):
    """Default filter transform plugin class."""

    def create(
        self, config: EnOptConfig, optimizer_callback: OptimizerCallback
    ) -> NomadOptimizer:
        """Initialize the optimizer plugin.

        See the [ropt.plugins.optimizer.base.OptimizerPlugin][] abstract base class.

        # noqa
        """
        return NomadOptimizer(config, optimizer_callback)

    def is_supported(self, method: str, *, explicit: bool) -> bool:  # noqa: ARG002
        """Check if a method is supported.

        See the [ropt.plugins.optimizer.base.OptimizerPlugin][] abstract base class.

        # noqa
        """
        return method.lower() in (_SUPPORTED_METHODS | {"default"})
