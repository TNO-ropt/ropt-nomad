# A NOMAD optimizer plugin for ropt
This package installs a plugin for the [ropt](https://github.com/tno-ropt/ropt)
ensemble optimizer package, giving access to algorithms from the
[NOMAD](https://www.gerad.ca/en/software/nomad/) optimization package.


## Dependencies
This code has been tested with Python version 3.11 on linux.

The backend requires the [NOMAD](https://www.gerad.ca/en/software/nomad/)
optimizer, which will be installed as a dependency from PyPI.


## Installation
From PyPI:
```bash
pip install ropt-nomad
```


## Usage
An optimization by ropt using the plugin works mostly as any other optimization
run (see also the [ropt documentation](https://tno-ropt.github.io/ropt/)).
However, there are a few things to
consider:

1. Gradients are not used, any specifications relating to gradient calculations
   in ropt are ignored.
2. Some standard optimization parameters that can be specified in the
   optimization section are ignored, specifically:
    - `algorithm`
    - `tolerance`
3. Only inequality constraints are supported by `NOMAD`.
4. Linear and non-linear constraints are both supported. Linear constraints are
   not supported directly, but are internally converted to non-linear
   constraints.
5. Additional options can be passed as a list of strings via the `options` field
   in the optimizer configuration. You can use any option that is supported by
   the `NOMAD` python interface, with the following exceptions:
   - The `DIMENSION` and cannot be overridden.
   - The `MAX_ITERATIONS` keyword can only be used if the `max_iterations` field
     is not set in the optimizer configuration.
   - `BB_OUTPUT_TYPE` can be overridden, which is useful to change the type of
     constraint handling.


## Development
The `ropt-nomad` source distribution can be found on
[GitHub](https://github.com/tno-ropt/ropt-nomad). It uses a standard
`pyproject.toml` file, which contains build information and configuration
settings for various tools. A development environment can be set up with
compatible tools of your choice.

The `ropt-nomad` package uses [ruff](https://docs.astral.sh/ruff/) (for
formatting and linting), [mypy](https://www.mypy-lang.org/) (for static typing),
and [pytest](https://docs.pytest.org/en/stable/) (for running the test suite).
