"""This module implements the NOMAD optimization plugin."""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, Final

import numpy as np
import PyNomad
from pydantic import Field
from ropt.config.options import OptionsSchemaModel
from ropt.enums import VariableType
from ropt.exceptions import StepAborted
from ropt.plugins.optimizer.base import Optimizer, OptimizerPlugin
from ropt.plugins.optimizer.utils import (
    NormalizedConstraints,
    get_masked_linear_constraints,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from ropt.config import EnOptConfig
    from ropt.optimization import OptimizerCallback

_SUPPORTED_METHODS: Final = {"mads"}
_DEFAULT_METHOD: Final = "mads"


class NomadOptimizer(Optimizer):
    """Nomad optimization backend for ropt.

    This class provides an interface to the `MADS` optimization algorithm from
    [`Nomad`](https://nomad-4-user-guide.readthedocs.io/en/latest/index.html),
    enabling their its within `ropt`.

    To select the `MADS` optimizer, set the `method` field within the
    [`optimizer`][ropt.config.OptimizerConfig] section of the
    [`EnOptConfig`][ropt.config.EnOptConfig] configuration object to
    `mads`. Most general options defined in the
    [`EnOptConfig`][ropt.config.EnOptConfig] object are supported. For
    algorithm-specific options, use the `options` dictionary within the
    [`optimizer`][ropt.config.OptimizerConfig] section.

    The table below lists the `MADS`-specific options that are supported. Click
    on the method name to consult the
    [`Nomad`](https://nomad-4-user-guide.readthedocs.io/en/latest/index.html)
    keyword documentation:

    --8<-- "nomad.md"
    """

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
        self._cached_variables: NDArray[np.float64] | None = None
        self._cached_function: NDArray[np.float64] | None = None
        self._exception: StepAborted | None = None

        _, _, self._method = self._config.optimizer.method.lower().rpartition("/")
        if self._method == "default":
            self._method = _DEFAULT_METHOD
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

        self._bounds = self._get_bounds()
        self._normalized_constraints = self._init_constraints(initial_values)
        self._parameters = self._get_parameters(self._normalized_constraints)

        PyNomad.optimize(
            self._evaluate,
            initial_values[self._config.variables.mask].tolist(),
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
        lower_bounds = self._config.variables.lower_bounds[self._config.variables.mask]
        upper_bounds = self._config.variables.upper_bounds[self._config.variables.mask]
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
        except StepAborted as exc:
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

    def _get_parameters(  # noqa: C901
        self, normalized_constraints: NormalizedConstraints | None
    ) -> list[str]:
        dim = self._config.variables.mask.sum()
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
                types = self._config.variables.types[self._config.variables.mask]
                bb_input_type = "BB_INPUT_TYPE ("
                for item in types:
                    bb_input_type += " I" if item == VariableType.INTEGER else " R"
                bb_input_type += " )"

                if option.strip().startswith("BB_OUTPUT_TYPE"):
                    if len(option.split()) != constraints + 2:
                        msg = "Option Error: BB_OUTPUT_TYPE specifies incorrect number of outputs"
                        raise ValueError(msg)
                    bb_output_type = None

                if option.strip().startswith("BB_MAX_BLOCK_SIZE"):
                    if self._config.optimizer.parallel is False:
                        msg = (
                            "Option Error: BB_MAX_BLOCK_SIZE may only be specified  "
                            "if the parallel option is True"
                        )
                        raise ValueError(msg)
                    have_bb_max_block_size = True

            parameters.extend(self._config.optimizer.options)

        if self._config.optimizer.parallel and have_bb_max_block_size is False:
            msg = (
                "Option Error: BB_MAX_BLOCK_SIZE must be specified "
                "if the parallel option is True"
            )
            raise ValueError(msg)

        if bb_input_type is not None:
            parameters = [bb_input_type, *parameters]

        if bb_output_type is not None:
            parameters.append(bb_output_type)

        return parameters

    def _get_constraint_bounds(
        self, nonlinear_bounds: tuple[NDArray[np.float64], NDArray[np.float64]] | None
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]] | None:
        bounds = []
        if nonlinear_bounds is not None:
            bounds.append(nonlinear_bounds)
        if self._linear_constraint_bounds is not None:
            bounds.append(self._linear_constraint_bounds)
        if bounds:
            lower_bounds, upper_bounds = zip(*bounds, strict=True)
            return np.concatenate(lower_bounds), np.concatenate(upper_bounds)
        return None

    def _init_constraints(
        self, initial_values: NDArray[np.float64]
    ) -> NormalizedConstraints | None:
        self._lin_coef: NDArray[np.float64] | None = None
        self._linear_constraint_bounds: (
            tuple[NDArray[np.float64], NDArray[np.float64]] | None
        ) = None
        if self._config.linear_constraints is not None:
            self._lin_coef, lin_lower, lin_upper = get_masked_linear_constraints(
                self._config, initial_values
            )
            self._linear_constraint_bounds = (lin_lower, lin_upper)
        nonlinear_bounds = (
            None
            if self._config.nonlinear_constraints is None
            else (
                self._config.nonlinear_constraints.lower_bounds,
                self._config.nonlinear_constraints.upper_bounds,
            )
        )
        if (bounds := self._get_constraint_bounds(nonlinear_bounds)) is not None:
            normalized_constraints = NormalizedConstraints(flip=True)
            normalized_constraints.set_bounds(*bounds)
            if np.any(normalized_constraints.is_eq):
                msg = "Equality constraints are not supported by NOMAD"
                raise ValueError(msg)
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
            callback_result = self._optimizer_callback(
                variables,
                return_functions=True,
                return_gradients=False,
            )
            function = callback_result.functions
            # The optimizer callback may change non-linear constraint bounds:
            if self._normalized_constraints is not None:
                bounds = self._get_constraint_bounds(
                    callback_result.nonlinear_constraint_bounds
                )
                assert bounds is not None
                self._normalized_constraints.set_bounds(*bounds)
            assert function is not None
            self._cached_function = function.copy()
        return self._cached_function


class NomadOptimizerPlugin(OptimizerPlugin):
    """Nomad optimizer plugin class."""

    @classmethod
    def create(
        cls, config: EnOptConfig, optimizer_callback: OptimizerCallback
    ) -> NomadOptimizer:
        """Initialize the optimizer plugin.

        See the [ropt.plugins.optimizer.base.OptimizerPlugin][] abstract base class.

        # noqa
        """
        return NomadOptimizer(config, optimizer_callback)

    @classmethod
    def is_supported(cls, method: str) -> bool:
        """Check if a method is supported.

        See the [ropt.plugins.optimizer.base.OptimizerPlugin][] abstract base class.

        # noqa
        """
        return method.lower() in (_SUPPORTED_METHODS | {"default"})

    @classmethod
    def validate_options(
        cls, method: str, options: dict[str, Any] | list[str] | None
    ) -> None:
        """Validate the options of a given method.

        See the [ropt.plugins.optimizer.base.OptimizerPlugin][] abstract base class.

        # noqa
        """
        if options is not None:
            if not isinstance(options, list):
                msg = "The Nomad optimizer options must be a list of strings"
                raise TypeError(msg)
            options_dict: dict[str, Any] = {}
            for option in options:
                split_option = re.split(r"\s+", option.strip(), maxsplit=1)
                options_dict[split_option[0]] = (
                    split_option[1]
                    if len(split_option) > 1 and split_option[1].strip()
                    else "yes"
                )
            *_, method = method.rpartition("/")
            OptionsSchemaModel.model_validate(_OPTIONS_SCHEMA).get_options_model(
                _DEFAULT_METHOD if method == "default" else method
            ).model_validate(options_dict)

            for option in options:
                if option.strip().startswith("BB_OUTPUT_TYPE"):
                    output_types = option.split()
                    if output_types[1] != "OBJ":
                        msg = (
                            "Option Error: First argument of BB_OUTPUT_TYPE must be OBJ"
                        )
                        raise ValueError(msg)
                    invalid_types = {
                        output_type
                        for output_type in output_types[2:]
                        if output_type not in {"EB", "F", "PB", "CSTR"}
                    }
                    if invalid_types:
                        msg = (
                            "Option Error: Invalid output type(s) in "
                            f"BB_OUTPUT_TYPE: {invalid_types}"
                        )
                        raise ValueError(msg)


_OPTIONS_SCHEMA: dict[str, Any] = {
    "methods": {
        "mads": {
            "options": {
                "BB_OUTPUT_TYPE": str,
                "BB_MAX_BLOCK_SIZE": Annotated[int, Field(gt=0)],
                "MAX_BB_EVAL": int,
                "MAX_EVAL": int,
                "SEED": int,
                "LH_SEARCH": str,
                "DISPLAY_ALL_EVAL": str,
                "DISPLAY_DEGREE": Annotated[int, Field(ge=0, le=3)],
                "DISPLAY_STATS": str,
            },
            "url": "https://nomad-4-user-guide.readthedocs.io/en/latest/Appendix.html#complete-list-of-parameters",
        },
    }
}


if __name__ == "__main__":
    from ropt.config.options import gen_options_table

    with Path("nomad.md").open("w", encoding="utf-8") as fp:
        fp.write(gen_options_table(_OPTIONS_SCHEMA))
