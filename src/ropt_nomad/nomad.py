"""This module implements the NOMAD optimization plugin."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, ClassVar, Final

import numpy as np
import PyNomad
from pydantic import Field
from ropt.backend import Backend
from ropt.backend.utils import (
    get_linear_constraints,
    get_nonlinear_equalities,
    resolve_verbosity,
    split_linear_constraints,
)
from ropt.config.options import OptionsSchemaModel
from ropt.enums import VariableType
from ropt.exceptions import UnsupportedError

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from ropt.config import BackendConfig
    from ropt.context import EnOptContext
    from ropt.core import OptimizerCallback
    from ropt.plugins import MethodSpec

_logger = logging.getLogger("ropt.backend.nomad")

_SUPPORTED_METHODS: Final = {"mads"}
_DEFAULT_METHOD: Final = "mads"
# The highest reporting level NOMAD accepts:
_MAX_DISPLAY_DEGREE: Final = 3


class NomadBackend(Backend):
    """Nomad optimization backend for ropt.

    This class provides an interface to the `MADS` optimization algorithm from
    [`Nomad`](https://nomad-4-user-guide.readthedocs.io/en/latest/index.html),
    enabling their its within `ropt`.

    !!! warning "This backend cannot run concurrently in-process"
        NOMAD keeps the state of a run inside the library rather than in
        anything it hands back, so a second run started while the first is still
        going corrupts it: NOMAD reports that a subproblem was not found, and
        both runs then hang. One optimization after another in the same process
        is fine; two at the same time are not. To use this backend alongside
        anything else, prefix the method with `external/` and it runs in a
        process of its own, through the
        [`external`][ropt.backend.external.ExternalBackend] backend.

    !!! note "Optimizer output"
        NOMAD reports its progress from its C++ implementation. How much it says
        follows the `verbose` setting of
        [`BackendConfig`][ropt.config.BackendConfig], which is mapped onto
        NOMAD's `DISPLAY_DEGREE`; supplying `DISPLAY_DEGREE` in `options`
        overrides it.

    To select the `MADS` optimizer, set the `method` field within the
    [`optimizer`][ropt.config.BackendConfig] section of the
    [`EnOptContext`][ropt.context.EnOptContext] configuration object to
    `mads`. Most general options defined in the
    [`EnOptContext`][ropt.context.EnOptContext] object are supported. For
    algorithm-specific options, use the `options` dictionary within the
    [`optimizer`][ropt.config.BackendConfig] section.

    The table below lists the `MADS`-specific options that are supported. Click
    on the method name to consult the
    [`Nomad`](https://nomad-4-user-guide.readthedocs.io/en/latest/index.html)
    keyword documentation:

    --8<-- "nomad.md"
    """

    methods: ClassVar[MethodSpec] = _SUPPORTED_METHODS | {"default"}

    def __init__(self, backend_config: BackendConfig) -> None:
        """Initialize the Nomad optimizer backend.

        Args:
            backend_config: The configuration for the backend, containing the
                            method name and options.
        """  # ruff: ignore[docstring-missing-exception]
        _, _, self._method = backend_config.method.lower().rpartition("/")
        if self._method == "default":
            self._method = _DEFAULT_METHOD
        if self._method not in _SUPPORTED_METHODS:
            msg = f"NOMAD optimizer algorithm '{self._method}' is not supported."
            raise UnsupportedError(msg)
        self._config = backend_config

    def init(
        self,
        context: EnOptContext,
        optimizer_callback: OptimizerCallback,
    ) -> None:
        """Initialize the optimizer implemented by the nomad plugin.

        See the [ropt.backend.Backend][] abstract base class.

        # noqa
        """
        self._context = context
        self._optimizer_callback = optimizer_callback
        self._cached_variables: NDArray[np.float64] | None = None
        self._cached_function: NDArray[np.float64] | None = None
        self._exception: Exception | None = None
        _logger.debug("Using NOMAD optimizer: %s", self._method)

    @property
    def is_parallel(self) -> bool:
        """Whether the current run is parallel.

        See the [ropt.backend.Backend][] abstract base class.

        # noqa
        """
        return self._config.parallel

    @property
    def bypasses_python_output(self) -> bool:
        """Whether the optimizer prints without going through Python.

        NOMAD reports its progress from its C++ implementation, so its output
        does not pass through `sys.stdout`.

        See the [ropt.backend.Backend][] abstract base class.

        # noqa
        """
        return True

    def start(self, initial_values: NDArray[np.float64]) -> None:
        """Start the optimization.

        See the [ropt.backend.Backend][] abstract base class.

        # noqa
        """
        self._cached_variables = None
        self._cached_function = None

        self._bounds = self._get_bounds()
        self._is_eq = self._init_constraints(initial_values)
        self._parameters = self._get_parameters(self._is_eq)

        PyNomad.optimize(
            self._evaluate,
            initial_values[self._context.variables.mask].tolist(),
            self._bounds[0],
            self._bounds[1],
            self._parameters,
        )
        if self._exception is not None:
            raise self._exception

    def validate_options(self) -> None:
        """Validate the options of a given method.

        See the [ropt.backend.Backend][] abstract base class.

        # noqa
        """  # ruff: ignore[docstring-missing-exception]
        if self._config.options is not None:
            if not isinstance(self._config.options, list):
                msg = "The Nomad optimizer options must be a list of strings"
                raise TypeError(msg)
            options_dict: dict[str, Any] = {}
            for option in self._config.options:
                split_option = re.split(r"\s+", option.strip(), maxsplit=1)
                options_dict[split_option[0]] = (
                    split_option[1]
                    if len(split_option) > 1 and split_option[1].strip()
                    else "yes"
                )
            OptionsSchemaModel.model_validate(_OPTIONS_SCHEMA).get_options_model(
                self._method
            ).model_validate(options_dict)

            for option in self._config.options:
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

    def _get_bounds(self) -> tuple[list[float], list[float]]:
        lower_bounds = self._context.variables.lower_bounds[
            self._context.variables.mask
        ]
        upper_bounds = self._context.variables.upper_bounds[
            self._context.variables.mask
        ]
        return lower_bounds.tolist(), upper_bounds.tolist()

    def _evaluate(
        self,
        block_or_eval_point: PyNomad.PyNomadEvalPoint | PyNomad.PyNomadBlock,
    ) -> int | list[int]:
        if self._exception is not None:
            return (
                0
                if isinstance(block_or_eval_point, PyNomad.PyNomadEvalPoint)
                else [0] * block_or_eval_point.size()
            )
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
        except Exception as exc:  # ruff: ignore[blind-except]
            _logger.warning("Evaluation failed: %s", exc)
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

    def _get_display_parameters(self, *, have_display_degree: bool) -> list[str]:
        # NOMAD reports at degree 2 unless told otherwise, so a parameter is
        # only added to turn that down or to set the level explicitly.
        if have_display_degree:
            return []
        level = resolve_verbosity(verbose=self._config.verbose)
        if level is None:
            return []
        return [f"DISPLAY_DEGREE {min(level, _MAX_DISPLAY_DEGREE)}"]

    def _get_parameters(  # ruff: ignore[complex-structure]
        self, is_eq: NDArray[np.bool_] | None
    ) -> list[str]:
        dim = self._context.variables.mask.sum()
        parameters = [f"DIMENSION {dim}"]

        constraints = 0 if is_eq is None else int(is_eq.size)
        bb_output_type: str | None = "BB_OUTPUT_TYPE OBJ" + " EB" * constraints
        have_bb_max_block_size = False
        have_display_degree = False

        if self._config.max_iterations is not None:
            parameters.append(f"MAX_ITERATIONS {self._config.max_iterations}")

        bb_input_type = None
        types = self._context.variables.types[self._context.variables.mask]
        if types is not None:
            bb_input_type = "BB_INPUT_TYPE ("
            for item in types:
                bb_input_type += " I" if item == VariableType.INTEGER else " R"
            bb_input_type += " )"

        if isinstance(self._config.options, list):
            for option in self._config.options:
                if option.strip().startswith("BB_OUTPUT_TYPE"):
                    if len(option.split()) != constraints + 2:
                        msg = "Option Error: BB_OUTPUT_TYPE specifies incorrect number of outputs"
                        raise ValueError(msg)
                    bb_output_type = None

                if option.strip().startswith("BB_MAX_BLOCK_SIZE"):
                    if self._config.parallel is False:
                        msg = (
                            "Option Error: BB_MAX_BLOCK_SIZE may only be specified  "
                            "if the parallel option is True"
                        )
                        raise ValueError(msg)
                    have_bb_max_block_size = True

            have_display_degree = any(
                option.strip().startswith("DISPLAY_DEGREE")
                for option in self._config.options
            )
            parameters.extend(self._config.options)

        parameters += self._get_display_parameters(
            have_display_degree=have_display_degree
        )

        if self._config.parallel and have_bb_max_block_size is False:
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

    def _init_constraints(
        self, initial_values: NDArray[np.float64]
    ) -> NDArray[np.bool_] | None:
        is_eq = get_nonlinear_equalities(self._context)
        self._nonlinear_constraint_count = 0 if is_eq is None else int(is_eq.size)
        self._linear_coefficients: NDArray[np.float64] | None = None
        self._linear_offsets: NDArray[np.float64] | None = None
        if self._context.linear_constraints is not None:
            coefficients, offsets, linear_is_eq = split_linear_constraints(
                *get_linear_constraints(self._context, initial_values)
            )
            self._linear_coefficients = coefficients
            self._linear_offsets = offsets
            is_eq = (
                linear_is_eq if is_eq is None else np.concatenate((is_eq, linear_is_eq))
            )
        if is_eq is None or is_eq.size == 0:
            return None
        if bool(is_eq.any()):
            msg = "Equality constraints are not supported by NOMAD"
            raise ValueError(msg)
        return is_eq

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
        if self._is_eq is None:
            return np.array([])
        blocks = []
        if self._nonlinear_constraint_count:
            functions = self._get_functions(variables)
            values = functions[1:] if variables.ndim == 1 else functions[:, 1:]
            blocks.append(np.atleast_2d(values).T if variables.ndim == 1 else values.T)
        if self._linear_coefficients is not None:
            assert self._linear_offsets is not None
            points = variables if variables.ndim > 1 else np.expand_dims(variables, 0)
            blocks.append(
                np.matmul(self._linear_coefficients, points.T)
                - self._linear_offsets[:, np.newaxis]
            )
        # NOMAD treats a constraint as satisfied when it is non-positive.
        return -np.concatenate(blocks, axis=0).transpose()

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
            callback_result = self._optimizer_callback(
                variables,
                return_functions=True,
                return_gradients=False,
            )
            function = callback_result.functions
            assert function is not None
            self._cached_function = function.copy()
        return self._cached_function


_OPTIONS_SCHEMA: dict[str, Any] = {
    "methods": {
        "mads": {
            "options": {
                "BB_INPUT_TYPE": str,
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

    Path("nomad.md").write_text(gen_options_table(_OPTIONS_SCHEMA), encoding="utf-8")
